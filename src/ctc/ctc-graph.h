// ctc/cctc-graph.h

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


#ifndef KALDI_CTC_CTC_GRAPH_H_
#define KALDI_CTC_CTC_GRAPH_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "ctc/ctc-transition-model.h"
#include "fstext/deterministic-fst.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {
namespace ctc {


/**  This function adds one to all the phones to the FST and adds self-loops
     for the the optional blank symbols to all of its states.
     @param [in,out]          The FST we modify.  At entry, the symbols on its input side
                              must be phones in the range [1, num_phones]
                              the output-side symbols will be left as they are.
                              If this represents a decoding graph you'll probably
                              want to have determinized this with disambiguation symbols
                              in place, then removed the disambiguation symbols and minimized
                              it.

                              What this function does is to add 1 to all nonzero
                              input symbols on arcs (to convert phones to
                              phones-plus-one), then at each state of the
                              modified FST, add a self-loop with a 1
                              (blank-plus-one) on the input and 0 (epsilon) on
                              the output.
*/
void ShiftPhonesAndAddBlanks(fst::StdVectorFst *fst,
                             bool add_phone_loop = false);

/** This is a Debug function
*/
void CtcGraphInfo(const TransitionModel &trans_model,
                  const fst::StdVectorFst &fst);

/** This is a Ctc version of the function DeterminizeLatticePhonePrunedWrapper,
    declared in ../lat/determinize-lattice-pruned.h.  It can be used
    as a top-level interface to all the determinization code.  It's destructive
    of its input.
*/
bool DeterminizeLatticePhonePrunedWrapperCtc(
  const CtcTransitionModel &trans_model,
  fst::MutableFst<kaldi::LatticeArc> *ifst,
  double prune,
  fst::MutableFst<kaldi::CompactLatticeArc> *ofst,
  fst::DeterminizeLatticePhonePrunedOptions opts);


}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CTC_GRAPH_H_
