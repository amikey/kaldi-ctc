// ctcbin/nnet2-ctc-latgen-faster.cc

// Copyright 2009-2012   Microsoft Corporation
//                       Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen
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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/kaldi-fst-io.h"

#include "nnet2/am-nnet.h"

#include "decoder/decoder-wrappers.h"
#include "ctc/ctc-decodable-am-nnet.h"
#include "ctc/ctc-decoder-wrappers.h"
#include "ctc/ctc-nnet-example.h"

#include "base/timer.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;

    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
      "Generate lattices using neural net model.\n"
      "Usage: nnet-latgen-faster [options] <nnet-in> <fst-in|fsts-rspecifier> <features-rspecifier>"
      " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = true;
    BaseFloat blank_threshold = 0.98;
    BaseFloat acoustic_scale = 1.0;
    int32 frame_subsampling_factor = 0, frame_shift = 0;

    LatticeFasterDecoderConfig config;
    std::string use_gpu = "yes";

    std::string word_syms_filename;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("blank-threshold", &blank_threshold,
                "If blank prob bigger than blank_threshold, skip this frame(faster), set 1.0 to avoid skip any frame.");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("frame-shift", &frame_shift, "Allows you to shift time values "
                "in the supervision data (excluding iVector data) - useful in "
                "augmenting data.  Note, the outputs will remain at the closest "
                "exact multiples of the frame subsampling factor");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor,
                "the frame subsampling factor");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string model_in_filename = po.GetArg(1),
                fst_in_str = po.GetArg(2),
                feature_rspecifier = po.GetArg(3),
                lattice_wspecifier = po.GetArg(4),
                words_wspecifier = po.GetOptArg(5),
                alignment_wspecifier = po.GetOptArg(6);

    CtcTransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatCuMatrixReader feature_reader(feature_rspecifier);

      // Input FST is just one FST, not a table of FSTs.
      VectorFst<StdArc> *decode_fst = fst::ReadFstKaldi(fst_in_str);

      {
        LatticeFasterDecoder decoder(*decode_fst, config);

        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          const CuMatrix<BaseFloat> &features (feature_reader.Value());
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
          bool pad_input = true;
          CuMatrix<BaseFloat> feats;
          if (frame_subsampling_factor > 1) {
            Matrix<BaseFloat> features_cpu(features);
            FrameSubsamplingShiftFeatureTimes(frame_subsampling_factor, frame_shift,
                                              features_cpu);
            feats.Resize(features_cpu.NumRows(), features_cpu.NumCols(), kUndefined,
                         kStrideEqualNumCols);
            feats.CopyFromMat(features_cpu);
          } else {
            feats.Resize(features.NumRows(), features.NumCols(), kUndefined,
                         kStrideEqualNumCols);
            feats.CopyFromMat(features);
          }

          CtcDecodableAmNnet nnet_decodable(trans_model,
                                            am_nnet,
                                            feats,
                                            pad_input,
                                            acoustic_scale,
                                            blank_threshold);
          double like;
          if (DecodeUtteranceLatticeFasterCtc(
                decoder, nnet_decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_success++;
          } else num_fail++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatCuMatrixReader feature_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_fail++;
          continue;
        }
        const CuMatrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }

        LatticeFasterDecoder decoder(fst_reader.Value(), config);

        bool pad_input = true;
        CuMatrix<BaseFloat> feats;
        if (frame_subsampling_factor > 1) {
          Matrix<BaseFloat> features_cpu(features);
          FrameSubsamplingShiftFeatureTimes(frame_subsampling_factor, frame_shift,
                                            features_cpu);
          feats.Resize(features_cpu.NumRows(), features_cpu.NumCols(), kUndefined,
                       kStrideEqualNumCols);
          feats.CopyFromMat(features_cpu);
        } else {
          feats.Resize(features.NumRows(), features.NumCols(), kUndefined,
                       kStrideEqualNumCols);
          feats.CopyFromMat(features);
        }

        CtcDecodableAmNnet nnet_decodable(trans_model,
                                          am_nnet,
                                          feats,
                                          pad_input,
                                          acoustic_scale);
        double like;
        if (DecodeUtteranceLatticeFasterCtc(
              decoder, nnet_decodable, trans_model, word_syms, utt,
              acoustic_scale, determinize, allow_partial, &alignment_writer,
              &words_writer, &compact_lattice_writer, &lattice_writer,
              &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          num_success++;
        } else num_fail++;
      }
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count<<" frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
