// ctc/ctc-nnet-train.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//           2016  LingoChamp Feiteng

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "ctc/ctc-nnet-train.h"
#include "nnet2/nnet-nnet.h"
#include "thread/kaldi-thread.h"

// The CUDA implementation supports a maximum label length of 639 (timesteps are unlimited).
#define MAX_WARPCTC_LABEL_LENGTH 639

namespace kaldi {
namespace ctc {

class NnetCtcExampleBackgroundReader {
 public:
  NnetCtcExampleBackgroundReader(int32 minibatch_size,
                                 Nnet *nnet,
                                 SequentialNnetCtcExampleReader *reader, int32 max_frames):
    minibatch_size_(minibatch_size), nnet_(nnet), reader_(reader),
    finished_(false), max_frames_(max_frames), num_skiped_example_(0) {
    // When this class is created, it spawns a thread which calls ReadExamples()
    // in the background.
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);
    int32 ret;
    // below, Run is the static class-member function.
    if ((ret=pthread_create(&thread_, &pthread_attr,
                            Run, static_cast<void*>(this)))) {
      const char *c = strerror(ret);
      if (c == NULL) {
        c = "[NULL]";
      }
      KALDI_ERR << "Error creating thread, errno was: " << c;
    }
    // the following call is a signal that no-one is currently using the examples_ and
    // formatted_examples_ class members.
    consumer_semaphore_.Signal();
  }

  ~NnetCtcExampleBackgroundReader() {
    if (KALDI_PTHREAD_PTR(thread_) == 0)
      KALDI_ERR << "No thread to join.";
    if (pthread_join(thread_, NULL))
      KALDI_ERR << "Error rejoining thread.";
    KALDI_LOG << "max_frames = " << max_frames_
              << ", skiped " << num_skiped_example_ << " examples.";
  }

  // This will be called in a background thread.  It's responsible for
  // reading and formatting the examples.
  void ReadExamples() {
    KALDI_ASSERT(minibatch_size_ > 0);
    int32 minibatch_size = minibatch_size_;


    // Loop over minibatches...
    while (true) {
      // When the following call succeeds we interpret it as a signal that
      // we are free to write to the class-member variables examples_ and formatted_examples_.
      consumer_semaphore_.Wait();

      examples_.clear();
      examples_.reserve(minibatch_size);
      // Read the examples.
      for (; examples_.size() < minibatch_size
           && !reader_->Done(); reader_->Next()) {
        int32 num_frames = reader_->Value().NumFrames();
        int32 num_labels = reader_->Value().NumLabels();
        if (num_frames > max_frames_
            || num_labels > MAX_WARPCTC_LABEL_LENGTH) {
          num_skiped_example_++;
          continue;
        } else if (num_frames < 2 * num_labels + 1) {
          KALDI_WARN << "Too little feature frames.";
          num_skiped_example_++;
          continue;
        }
        examples_.push_back(reader_->Value());
      }

      // Format the examples as a single matrix.  The reason we do this here is
      // that it's a somewhat CPU-intensive operation (involves decompressing
      // the matrix), so we do it in a separate thread from the one that's
      // controlling the GPU (assuming we're using a GPU), so we can get better
      // GPU utilization.  If we have no GPU this doesn't hurt us.
      if (examples_.empty()) {
        formatted_examples_.Resize(0, 0);
        total_weight_ = 0.0;
      } else {
        FormatNnetInput(*nnet_, examples_, &formatted_examples_);
        total_weight_ = TotalNnetTrainingWeight(examples_);
      }

      bool finished = examples_.empty();

      // The following call alerts the main program thread (that calls
      // GetNextMinibatch() that it can how use the contents of
      // examples_ and formatted_examples_.
      producer_semaphore_.Signal();

      // If we just read an empty minibatch (because no more examples),
      // then return.
      if (finished)
        return;
    }
  }

  // this wrapper can be passed to pthread_create.
  static void* Run(void *ptr_in) {
    NnetCtcExampleBackgroundReader *ptr =
      reinterpret_cast<NnetCtcExampleBackgroundReader*>(ptr_in);
    ptr->ReadExamples();
    return NULL;
  }

  // This call makes available the next minibatch of input.  It returns
  // true if it got some, and false if there was no more available.
  // It is an error if you call this function after it has returned false.
  bool GetNextMinibatch(std::vector<NnetCtcExample> *examples,
                        Matrix<BaseFloat> *formatted_examples,
                        double *total_weight) {
    KALDI_ASSERT(!finished_);
    // wait until examples_ and formatted_examples_ have been created by
    // the background thread.
    producer_semaphore_.Wait();
    // the calls to swap and Swap are lightweight.
    examples_.swap(*examples);
    formatted_examples_.Swap(formatted_examples);
    *total_weight = total_weight_;

    // signal the background thread that it is now free to write
    // again to examples_ and formatted_examples_.
    consumer_semaphore_.Signal();

    if (examples->empty()) {
      finished_ = true;
      return false;
    } else {
      return true;
    }
  }

 private:
  int32 minibatch_size_;
  Nnet *nnet_;
  SequentialNnetCtcExampleReader *reader_;
  pthread_t thread_;

  std::vector<NnetCtcExample> examples_;
  Matrix<BaseFloat> formatted_examples_;
  double total_weight_;  // total weight, from TotalNnetTrainingWeight(examples_).
  // better to compute this in the background thread.

  Semaphore producer_semaphore_;
  Semaphore consumer_semaphore_;

  bool finished_;
  int32 max_frames_;
  int32 num_skiped_example_;
};



int64 TrainNnetSimple(const NnetSimpleTrainerConfig &config,
                      Nnet *nnet,
                      SequentialNnetCtcExampleReader *reader,
                      double *tot_weight_ptr,
                      double *tot_logprob_ptr) {
  int64 num_egs_processed = 0;
  double tot_weight = 0.0, tot_logprob = 0.0, tot_accuracy = 0.0;
  NnetCtcExampleBackgroundReader background_reader(
    config.minibatch_size,
    nnet, reader, config.max_allow_frames);
  KALDI_ASSERT(config.minibatches_per_phase > 0);
  KALDI_ASSERT(config.max_param_change > 0);
  KALDI_ASSERT(config.momentum >= 0 && config.momentum < 1.0);
  nnet2::Nnet *delta_nnet = NULL;
  if (config.momentum != 0) {
    KALDI_ASSERT(config.momentum >= 0 && config.momentum < 1);
    KALDI_LOG << "Nnet2 RNN momentum " << config.momentum << " training.";
    delta_nnet = new nnet2::Nnet(*nnet);
    KALDI_ASSERT(delta_nnet != NULL);
    bool treat_as_gradient = false;
    delta_nnet->SetZero(treat_as_gradient);
  }

  while (true) {
    // Iterate over phases.  A phase of training is just a certain number of
    // minibatches, and its only significance is that it's the periodicity with
    // which we print diagnostics.
    double tot_weight_this_phase = 0.0, tot_logprob_this_phase = 0.0,
           tot_accuracy_this_phase = 0.0;

    int32 i;
    for (i = 0; i < config.minibatches_per_phase; i++) {
      std::vector<NnetCtcExample> examples;
      Matrix<BaseFloat> examples_formatted;
      double minibatch_total_weight, minibatch_total_accuracy;
      if (!background_reader.GetNextMinibatch(&examples,
                                              &examples_formatted,
                                              &minibatch_total_weight))
        break;
      tot_logprob_this_phase += DoBackprop(*nnet, examples,
                                           &examples_formatted,
                                           (delta_nnet == NULL) ? nnet : delta_nnet,
                                           &minibatch_total_accuracy);

      if (delta_nnet != NULL) {
        // BaseFloat scale = (1.0 - config.momentum);
        // if (config.max_param_change != 0.0) {
        //   Vector<BaseFloat> dot_prod(delta_nnet->NumUpdatableComponents(), kSetZero);
        //   delta_nnet->ComponentDotProducts(*delta_nnet, &dot_prod);
        //   BaseFloat param_delta = std::sqrt(dot_prod.Sum()) * scale;
        //   if (param_delta > config.max_param_change) {
        //     if (param_delta - param_delta != 0.0) {
        //       KALDI_WARN << "Infinite parameter change, will not apply.";
        //       delta_nnet->SetZero(false);
        //     } else {
        //       scale *= config.max_param_change / param_delta;
        //       KALDI_LOG << "Parameter change too big: " << param_delta << " > "
        //                 << "--max-param-change=" << config.max_param_change
        //                 << ", scaling by " << config.max_param_change / param_delta;
        //     }
        //   }
        // }
        nnet->AddNnet(1.0, *delta_nnet);
        delta_nnet->Scale(config.momentum);
      }

      tot_weight_this_phase += minibatch_total_weight;
      tot_accuracy_this_phase += minibatch_total_accuracy;

      num_egs_processed += examples.size();
    }
    if (i != 0) {
      KALDI_VLOG(1) << "Training objective function (this phase) is "
                    << (tot_logprob_this_phase / tot_weight_this_phase) << " over "
                    << tot_weight_this_phase << " tokens.";
    }

    tot_weight += tot_weight_this_phase;
    tot_logprob += tot_logprob_this_phase;
    tot_accuracy += tot_accuracy_this_phase;

    if (tot_weight_this_phase != 0.0)
      KALDI_VLOG(1) << "tot_accuracy_this_phase = " <<
                    tot_accuracy_this_phase /
                    tot_weight_this_phase;
    if (i != config.minibatches_per_phase) {
      // did not get all the minibatches we wanted because no more input.
      // this is true if and only if we did "break" in the loop over i above.
      break;
    }
  }
  if (tot_weight == 0.0) {
    KALDI_WARN << "No data seen.";
  } else {
    KALDI_LOG << "Did backprop on " << tot_weight
              << " examples, average log-prob per frame is "
              << (tot_logprob / tot_weight);
    KALDI_LOG << "[this line is to be parsed by a script:] Accuracy = "
              << (tot_accuracy / tot_weight);
  }
  if (tot_weight_ptr) *tot_weight_ptr = tot_weight;
  if (tot_logprob_ptr) *tot_logprob_ptr = tot_logprob;
  return num_egs_processed;
}

}  // namespace ctc
}  // namespace kaldi
