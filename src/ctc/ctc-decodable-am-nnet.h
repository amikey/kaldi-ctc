// ctc/decodable-am-nnet.h

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

#ifndef KALDI_CTC_CTC_DECODABLE_AM_NNET_H_
#define KALDI_CTC_CTC_DECODABLE_AM_NNET_H_

#include <vector>
#include "base/kaldi-common.h"
#include "itf/decodable-itf.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute.h"
#include "ctc/ctc-transition-model.h"

namespace kaldi {
namespace ctc {

using kaldi::nnet2::AmNnet;

/// DecodableAmNnet is a decodable object that decodes
/// with a neural net acoustic model of type AmNnet.

class CtcDecodableAmNnet: public DecodableInterface {
 public:
  CtcDecodableAmNnet(const CtcTransitionModel &trans_model,
                     const AmNnet &am_nnet,
                     const CuMatrixBase<BaseFloat> &feats,
                     bool pad_input = true, // if !pad_input, the NumIndices()
                     // will be < feats.NumRows().
                     BaseFloat prob_scale = 1.0, BaseFloat blank_threshold = 1.0);

  // Note, frames are numbered from zero.  But tid is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return log_probs_(frame, trans_model_.TransitionIdToPdf(tid));
  }

  virtual int32 NumFramesReady() const {
    return log_probs_.NumRows();
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const {
    return am_nnet_.NumPdfs();
  }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 protected:
  const CtcTransitionModel &trans_model_;
  const AmNnet &am_nnet_;
  Matrix<BaseFloat>
  log_probs_;  // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.

  KALDI_DISALLOW_COPY_AND_ASSIGN(CtcDecodableAmNnet);
};

/// This version of DecodableAmNnet is intended for a version of the decoder
/// that processes different utterances with multiple threads.  It needs to do
/// the computation in a different place than the initializer, since the
/// initializer gets called in the main thread of the program.

class CtcDecodableAmNnetParallel: public DecodableInterface {
 public:
  CtcDecodableAmNnetParallel(const CtcTransitionModel &trans_model,
                             const AmNnet &am_nnet,
                             const CuMatrix<BaseFloat> *feats,
                             bool pad_input = true,
                             BaseFloat prob_scale = 1.0):
    trans_model_(trans_model),
    am_nnet_(am_nnet), feats_(feats),
    pad_input_(pad_input), prob_scale_(prob_scale) {
    KALDI_ASSERT(trans_model_.NumPdfs() == am_nnet_.NumPdfs());
    KALDI_ASSERT(feats_ != NULL);
  }

  void Compute();

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    if (feats_) Compute();  // this function sets feats_ to NULL.
    return log_probs_(frame, trans_model_.TransitionIdToPdf(tid));
  }

  int32 NumFramesReady() const {
    if (feats_) {
      if (pad_input_) {
        return feats_->NumRows();
      } else {
        int32 ans = feats_->NumRows() - am_nnet_.GetNnet().LeftContext() -
                    am_nnet_.GetNnet().RightContext();
        if (ans < 0) ans = 0;
        return ans;
      }
    } else {
      return log_probs_.NumRows();
    }
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const {
    return am_nnet_.NumPdfs();
  }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
  ~CtcDecodableAmNnetParallel() {
    delete feats_;
  }
 protected:
  const CtcTransitionModel &trans_model_;
  const AmNnet &am_nnet_;
  // actually not really probabilities, since we divide
  CuMatrix<BaseFloat> log_probs_;
  // by the prior -> they won't sum to one.
  const CuMatrix<BaseFloat> *feats_;
  bool pad_input_;
  BaseFloat prob_scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(CtcDecodableAmNnetParallel);
};


}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CTC_DECODABLE_AM_NNET_H_
