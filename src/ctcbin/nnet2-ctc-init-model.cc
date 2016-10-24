// ctcbin/ctc-init-transition-model.cc

// Copyright       2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "ctc/ctc-transition-model.h"
#include "nnet2/am-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;

    const char *usage =
      "Initialize CCTC transition-model object.\n"
      "\n"
      "Usage:  ctc-init-transition-model [options] <tree-in> <topo-file> <nnet> <ctc-model-out>\n"
      "e.g.:\n"
      "ctc-init-model tree topo nnet dir/0.mdl\n";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId("yes");
#endif

    std::string tree_filename = po.GetArg(1),
                topo_filename = po.GetArg(2),
                raw_nnet_rxfilename  = po.GetArg(3),
                ctc_model_wxfilename = po.GetArg(4);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);
    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);
    CtcTransitionModel ctc_trans_model(ctx_dep, topo);

    // default priors: only divide the blank label posterior by a constant value(9)
    Vector<BaseFloat> priors(ctc_trans_model.NumPdfs(), kSetZero);
    priors.Set(1.0);  // - log(1.0) = 0
    priors(0) = 9;    // - log(p/9) = - (log(p) - log(9))

    nnet2::Nnet nnet;
    bool binary;
    Input ki(raw_nnet_rxfilename, &binary);
    nnet.Read(ki.Stream(), binary);

    nnet2::AmNnet am_nnet(nnet);
    am_nnet.SetPriors(priors);

    Output ko(ctc_model_wxfilename, binary_write);
    ctc_trans_model.Write(ko.Stream(), binary_write);
    am_nnet.Write(ko.Stream(), binary_write);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

