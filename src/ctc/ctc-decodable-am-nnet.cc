// ctc/ctc-decodable-am-nnet.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)
//                2016   LingoChamp Feiteng

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

#include "ctc/ctc-decodable-am-nnet.h"


namespace kaldi {
namespace ctc {

using kaldi::nnet2::Nnet;

CtcDecodableAmNnet::CtcDecodableAmNnet(const CtcTransitionModel &trans_model,
                                       const AmNnet &am_nnet,
                                       const CuMatrixBase<BaseFloat> &feats,
                                       bool pad_input,
                                       BaseFloat prob_scale,
                                       BaseFloat blank_threshold):
  trans_model_(trans_model), am_nnet_(am_nnet) {
  // Note: we could make this more memory-efficient by doing the
  // computation in smaller chunks than the whole utterance, and not
  // storing the whole thing.  We'll leave this for later.
  int32 num_rows = feats.NumRows() -
                   (pad_input ? 0 : am_nnet.GetNnet().LeftContext() +
                    am_nnet.GetNnet().RightContext());
  if (num_rows <= 0) {
    KALDI_WARN << "Input with " << feats.NumRows()
               << " rows will produce "
               << "empty output.";
    return;
  }
  CuMatrix<BaseFloat> log_probs(num_rows, am_nnet.NumPdfs());

  // the following function is declared in nnet-compute.h
  KALDI_ASSERT(feats.NumCols() == feats.Stride());
  NnetComputation(am_nnet.GetNnet(), feats, pad_input, &log_probs);

  if (blank_threshold < 1.0) {
    std::vector<int32> keep;
    for (int32 i = 0; i < log_probs.NumRows(); ++i) {
      if (log_probs(i, 0) < blank_threshold)
        keep.push_back(i);
    }
    if (keep.size() != log_probs.NumRows()) {
      if (keep.size() == 0) {
        KALDI_WARN << "No Frame will be keeped(try larger blank_threshold), don't skip blank.";
      } else {
        CuMatrix<BaseFloat> keep_log_probs(keep.size(), NumIndices());
        keep_log_probs.CopyRows(log_probs, CuArray<int32>(keep));
        log_probs = keep_log_probs;
      }
    }
  }

  log_probs.ApplyFloor(1.0e-10);  // Avoid log of zero which leads to NaN.
  log_probs.ApplyLog();

  if (am_nnet.Priors().Dim()) {
    CuVector<BaseFloat> priors(am_nnet.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model_.NumPdfs());
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs.AddVecToRows(-1.0, priors);
  }

// apply probability scale.
  log_probs.Scale(prob_scale);
// Transfer the log-probs to the CPU for faster access by the
// decoding process.
  log_probs_.Swap(&log_probs);
}

void CtcDecodableAmNnetParallel::Compute() {
  log_probs_.Resize(feats_->NumRows(), am_nnet_.NumPdfs());
  NnetComputation(am_nnet_.GetNnet(), *feats_, pad_input_, &log_probs_);

  log_probs_.ApplyFloor(1.0e-20);  // Avoid log of zero which leads to NaN.
  log_probs_.ApplyLog();

  if (am_nnet_.Priors().Dim()) {
    CuVector<BaseFloat> priors(am_nnet_.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model_.NumPdfs());
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs_.AddVecToRows(-1.0, priors);
  }
  // apply probability scale.
  log_probs_.Scale(prob_scale_);
  delete feats_;
  feats_ = NULL;
}

}  // namespace ctc
}  // namespace kaldi
