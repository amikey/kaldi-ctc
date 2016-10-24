// ctcbin/ctc-make-decoding-graph.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "fst/fstlib.h"
#include "ctc/ctc-graph.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
      "Executes the last stages of creating a CTC decoding graph,\n"
      "given an LG.fst on the input, e.g. min(det(L o G))).\n"
      "\n"
      "Usage:  ctc-make-decoding-graph <in-fst> <out-fst>\n"
      "E.g:  fstrmsymbols data/lang/phones/disambig.int LG.fst | \\\n"
      "         ctc-make-decoding-graph - CTC.fst\n";

    bool add_phone_loop = false;
    ParseOptions po(usage);
    po.Register("add-phone-loop", &add_phone_loop,
                "Add phone loop in CTC decoding graph.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_rxfilename = po.GetOptArg(1),
                fst_wxfilename = po.GetOptArg(2);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_rxfilename);

    ctc::ShiftPhonesAndAddBlanks(fst, add_phone_loop);

    WriteFstKaldi(*fst, fst_wxfilename);

    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

