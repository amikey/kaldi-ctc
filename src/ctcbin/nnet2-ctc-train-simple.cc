// ctcbin/nnet2-ctc-train-simple.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "ctc/ctc-nnet-train.h"
#include "ctc/ctc-transition-model.h"
#include "nnet2/am-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Train the neural network parameters with backprop and stochastic\n"
      "gradient descent using minibatches(CTC).  Training examples would be\n"
      "produced by nnet-ctc-get-egs.\n"
      "\n"
      "Usage:  nnet-ctc-train-simple [options] <model-in> <training-examples-in> <model-out>\n"
      "\n"
      "e.g.:\n"
      "nnet-ctc-train-simple 1.nnet ark:1.egs 2.nnet\n";

    bool binary_write = true;
    bool zero_stats = true;
    int32 srand_seed = 0;
    std::string use_gpu = "yes";
    NnetSimpleTrainerConfig train_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("zero-stats", &zero_stats, "If true, zero occupation "
                "counts stored with the neural net (only affects mixing up).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(relevant if you have layers of type AffineComponentPreconditioned "
                "with l2-penalty != 0.0");
    // po.Register("use-gpu", &use_gpu,
    //             "yes|no|optional|wait, only has effect if compiled with CUDA");

    train_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    srand(srand_seed);

    KALDI_LOG << "momentum = " << train_config.momentum;

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                examples_rspecifier = po.GetArg(2),
                nnet_wxfilename = po.GetArg(3);

    int64 num_examples;

    {
      CtcTransitionModel trans_model;
      nnet2::AmNnet am_nnet;
      {
        bool binary_read;
        Input ki(nnet_rxfilename, &binary_read);
        trans_model.Read(ki.Stream(), binary_read);
        am_nnet.Read(ki.Stream(), binary_read);
      }

      if (zero_stats) am_nnet.GetNnet().ZeroStats();

      SequentialNnetCtcExampleReader example_reader(examples_rspecifier);

      num_examples = TrainNnetSimple(train_config, &(am_nnet.GetNnet()),
                                     &example_reader);

      {
        Output ko(nnet_wxfilename, binary_write);
        trans_model.Write(ko.Stream(), binary_write);
        am_nnet.Write(ko.Stream(), binary_write);
      }
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


