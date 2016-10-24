// ctc/cctc-transition-model.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)
//                 2016  LingoChamp Feiteng

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


#ifndef KALDI_CTC_CTC_TRANSITION_MODEL_H_
#define KALDI_CTC_CTC_TRANSITION_MODEL_H_

#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace ctc {

// CTC means Connectionist Temporal Classification, see the paper by Graves et
// al.  CCTC means context-dependent CTC, it's an extension of the original model.
//

// Ctc corresponds to context-dependent connectionist temporal classification;
// it's an extension to the original model in which we make the output
// probabilities dependent on the previously emitted symbols.
// This class includes everything but the neural net; is is similar in
// function to the TransitionModel class in the HMM-based approach.
class CtcTransitionModel {
 public:
  CtcTransitionModel() { }

  CtcTransitionModel(const ContextDependencyInterface &ctx_dep,
                     const HmmTopology &hmm_topo): trans_model_(ctx_dep, hmm_topo) {}

  int32 NumPhones() const {
    return trans_model_.NumPhones();
  }

  int32 NumPdfs() const {
    return trans_model_.NumPdfs() + 1;
  }

  int32 TransitionIdToPdf(int32 tid) const {
    KALDI_ASSERT(tid >= 1 && tid <= NumGraphLabels());
    if (tid == 1)  // this's blank
      return 0;
    else
      return trans_model_.TransitionIdToPdf(tid - 1) + 1;  // shift 1(blank)
  }

  // Graph-labels are numbered from 1 to NumGraphLabels()
  int32 NumGraphLabels() const {
    return trans_model_.NumTransitionIds() + 1;
  }

  // Maps graph-label to phone (i.e. the predicted phone, or 0 for blank).
  int32 GraphLabelToPhone(int32 tid) const {
    KALDI_ASSERT(tid >= 1 && tid <= NumGraphLabels());
    if (tid == 1)
      return 0;
    return trans_model_.TransitionIdToPhone(tid - 1);
  }

  // should be called after GraphLabelToPhone() > 0
  bool IsSelfLoop(int32 tid) const {
    KALDI_ASSERT(tid >= 1 && tid <= NumGraphLabels());
    return trans_model_.IsSelfLoop(tid - 1);
  }

  void Write(std::ostream &os, bool binary) const {
    trans_model_.Write(os, binary);
  }

  void Read(std::istream &is, bool binary) {
    trans_model_.Read(is, binary);
  }

  std::string Info() const {
    std::ostringstream ostr;
    ostr << "num-phones: " << NumPhones() << "\n";
    return ostr.str();
  }

 protected:
  TransitionModel trans_model_;  // context dependent phone model for CTC
};

}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_TRANSITION_MODEL_H_
