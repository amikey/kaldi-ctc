// nnet2bin/nnet-get-egs.cc

// Copyright 2016  LingoChamp Feiteng

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "ctc/ctc-nnet-example.h"

namespace kaldi {
namespace ctc {


static void ProcessFile(const MatrixBase<BaseFloat> &feats,
                        const std::vector<int32> &pdf_post,
                        const std::string &utt_id,
                        int32 left_context,
                        int32 right_context,
                        int32 const_feat_dim,
                        int64 *num_utts_written,
                        int64 *num_egs_written,
                        NnetCtcExampleWriter *example_writer) {
  KALDI_ASSERT(feats.NumRows() >= static_cast<int32>(pdf_post.size()));
  int32 feat_dim = feats.NumCols();
  KALDI_ASSERT(const_feat_dim < feat_dim);
  int32 basic_feat_dim = feat_dim - const_feat_dim;

  int32 this_num_frames = feats.NumRows();

  int32 tot_frames = left_context + this_num_frames + right_context;
  NnetCtcExample eg;
  Matrix<BaseFloat> input_frames(tot_frames, basic_feat_dim);

  eg.left_context = left_context;
  eg.spk_info.Resize(const_feat_dim);

  // Set up "input_frames".
  for (int32 j = -left_context; j < this_num_frames + right_context; j++) {
    int32 t2 = j;
    if (t2 < 0) t2 = 0;
    if (t2 >= feats.NumRows()) t2 = feats.NumRows() - 1;
    if (j == 0) {
      SubMatrix<BaseFloat> src(feats, 0, this_num_frames, 0, basic_feat_dim),
                dest(input_frames, left_context, this_num_frames, 0, basic_feat_dim);
      dest.CopyFromMat(src);
      j += this_num_frames - 1;
    } else {
      SubVector<BaseFloat> src(feats.Row(t2), 0, basic_feat_dim),
                dest(input_frames, j + left_context);
      dest.CopyFromVec(src);
    }
  }

  if (const_feat_dim > 0) {
    eg.spk_info.AddRowSumMat(1.0 / this_num_frames, feats);
  }

  eg.labels = pdf_post;
  eg.input_frames = input_frames;  // Copy to CompressedMatrix.

  *num_utts_written += 1;
  *num_egs_written += 1;

  example_writer->Write(utt_id, eg);
}


} // namespace ctc
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Get full-utterence examples of data for neural network training.\n"
      "note that this does not randomize the order of utterence.\n"
      "\n"
      "Usage:  nnet-ctc-get-egs [options] <features-rspecifier> "
      "<pdf-id-rspecifier> <training-examples-out>\n"
      "\n"
      "An example [where $feats expands to the actual features]:\n"
      "nnet-ctc-get-egs --left-context=0 --right-context=0 \"$feats\" \\\n"
      "  \"ark:gunzip -c exp/nnet/ali.1.gz\" ark:- \n"
      "Note: the --left-context and --right-context would be derived from\n"
      "the output of nnet-info.";

    const int32 max_label_length = 639;

    int32 left_context = 0, right_context = 0, const_feat_dim = 0;
    bool is_ctc = true;

    ParseOptions po(usage);
    po.Register("ctc", &is_ctc, "Wether get egs for CTC training.");
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("const-feat-dim", &const_feat_dim, "If specified, the last "
                "const-feat-dim dimensions of the feature input are treated as "
                "constant over the context window (so are not spliced)");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
                pdf_post_rspecifier = po.GetArg(2),
                examples_wspecifier = po.GetArg(3);

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessInt32VectorReader pdf_post_reader(pdf_post_rspecifier);
    NnetCtcExampleWriter example_writer(examples_wspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_utts_written = 0, num_egs_written = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!pdf_post_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        const std::vector<int32> &pdf_post = pdf_post_reader.Value(key);
        if (pdf_post.size() != feats.NumRows() || is_ctc) {
          if (!is_ctc) {
            KALDI_WARN << "Posterior has wrong size " << pdf_post.size()
                       << " versus " << feats.NumRows();
            num_err++;
          } else {
            if (2 * pdf_post.size() + 1 > feats.NumRows()) {
              KALDI_WARN << "CTC: label has wrong size: (2 * " << pdf_post.size()
                         << " + 1) > " << feats.NumRows();
              continue;
            } else if (pdf_post.size() > max_label_length) {
              KALDI_WARN << "CTC: label has wrong size: " << pdf_post.size()
                         << " > 639.";
              continue;
            }
          }
        }
        ProcessFile(feats, pdf_post, key,
                    left_context, right_context,
                    const_feat_dim, &num_utts_written, &num_egs_written,
                    &example_writer);
        num_done++;
      }
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples, "
              << " with " << num_utts_written << " egs in total; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 || num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
