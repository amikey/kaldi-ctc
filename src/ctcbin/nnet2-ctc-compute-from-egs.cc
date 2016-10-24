// nnet2bin/nnet-compute-from-egs.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"
#include "ctc/ctc-nnet-example.h"
#include "ctc/ctc-transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Does the neural net computation, taking as input the nnet-training examples\n"
      "(typically an archive with the extension .egs), ignoring the labels; it\n"
      "outputs as a matrix the result.  Used mostly for debugging.\n"
      "\n"
      "Usage:  nnet-ctc-compute-from-egs [options] <model-in> <egs-rspecifier> "
      "<feature-wspecifier>\n"
      "e.g.:  nnet-ctc-compute-from-egs final.mdl egs.10.1.ark ark:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId("yes");
#endif

    std::string model_in_filename = po.GetArg(1),
                examples_rspecifier = po.GetArg(2),
                features_or_loglikes_wspecifier = po.GetArg(3);

    CtcTransitionModel trans_model;
    nnet2::AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    int64 num_egs = 0;

    SequentialNnetCtcExampleReader example_reader(examples_rspecifier);
    BaseFloatMatrixWriter writer(features_or_loglikes_wspecifier);

    nnet2::Nnet nnet = am_nnet.GetNnet();
    int32 left_context = nnet.LeftContext();

    for (; !example_reader.Done(); example_reader.Next()) {
      const NnetCtcExample &eg = example_reader.Value();
      int32 start_offset = eg.left_context - left_context;
      int32 basic_dim = eg.input_frames.NumCols(),
            spk_dim = eg.spk_info.Dim(), dim = basic_dim + spk_dim;
      int32 context = eg.input_frames.NumRows() - start_offset;
      Matrix<BaseFloat> input_frames(eg.input_frames),
             input_block(context, dim);
      input_block.Range(0, context, 0, basic_dim).CopyFromMat(
        input_frames.Range(start_offset, context, 0, basic_dim));
      if (spk_dim != 0) {
        input_block.Range(0, context, basic_dim, spk_dim).CopyRowsFromVec(
          eg.spk_info);
      }
      CuMatrix<BaseFloat> gpu_input_block(input_block.NumRows(),
                                          input_block.NumCols(), kUndefined,
                                          kStrideEqualNumCols);
      gpu_input_block.CopyFromMat(input_block);
      CuMatrix<BaseFloat> gpu_output_block(gpu_input_block.NumRows() - left_context -
                                           nnet.RightContext(),
                                           nnet.OutputDim());
      bool pad_input = false;
      NnetComputation(dynamic_cast<nnet2::Nnet &>(nnet), gpu_input_block, pad_input,
                      &gpu_output_block);

      writer.Write("global", Matrix<BaseFloat>(gpu_output_block));
      num_egs++;
    }

    KALDI_LOG << "Processed " << num_egs << " examples.";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return (num_egs == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


