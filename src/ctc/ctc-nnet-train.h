// ctc/ctc-nnet-train.h

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

#ifndef KALDI_CTC_CTC_TRAIN_H_
#define KALDI_CTC_CTC_TRAIN_H_

#include "nnet2/nnet-nnet.h"
#include "ctc/ctc-nnet-example.h"
#include "ctc/ctc-nnet-update.h"

namespace kaldi {
namespace ctc {

using kaldi::nnet2::Nnet;

struct NnetSimpleTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  BaseFloat momentum;
  BaseFloat max_param_change;
  int32 max_allow_frames;

  NnetSimpleTrainerConfig():
    minibatch_size(128),
    minibatches_per_phase(50),
    momentum(0.0),
    max_param_change(10.0),
    max_allow_frames(1000) { }

  void Register (OptionsItf *opts) {
    opts->Register("minibatch-size", &minibatch_size,
                   "Number of samples per minibatch of training data.");
    opts->Register("minibatches-per-phase", &minibatches_per_phase,
                   "Number of minibatches to wait before printing training-set "
                   "objective.");
    opts->Register("momentum", &momentum, "momentum constant to apply during "
                   "training (help stabilize update).  e.g. 0.9.  Note: we "
                   "automatically multiply the learning rate by (1-momenum) "
                   "so that the 'effective' learning rate is the same as "
                   "before (because momentum would normally increase the "
                   "effective learning rate by 1/(1-momentum))");
    opts->Register("max-param-change", &max_param_change, "The maximum change in"
                   "parameters allowed per minibatch, measured in Frobenius norm "
                   "over the entire model (change will be clipped to this value)");
    opts->Register("max-allow-frames", &max_allow_frames,
                   "The maximum frames length"
                   "parameters allowed per minibatch");
  }
};


/// Train on all the examples it can read from the reader.  This does training
/// in a single thread, but it uses a separate thread to read in the examples
/// and format the input data on the CPU; this saves us time when using GPUs.
/// Returns the number of examples processed.
/// Outputs to tot_weight and tot_logprob_per_frame, if non-NULL, the total
/// weight of the examples (typically equal to the number of examples) and the
/// total logprob objective function.
int64 TrainNnetSimple(const NnetSimpleTrainerConfig &config,
                      Nnet *nnet,
                      SequentialNnetCtcExampleReader *reader,
                      double *tot_weight = NULL,
                      double *tot_logprob = NULL);

}  // namespace ctc
}  // namespace kaldi

#endif
