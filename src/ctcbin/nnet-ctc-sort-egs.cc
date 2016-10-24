// ctcbin/nnet-ctc-sort-egs.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "ctc/ctc-nnet-example.h"

using kaldi::ctc::NnetCtcExample;

bool SortNnetCtcExample(const std::pair<std::string, NnetCtcExample*> &e1,
                        const std::pair<std::string, NnetCtcExample*> &e2) {
  return e1.second->NumFrames() < e2.second->NumFrames();
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Copy examples (typically single frames) for neural network training,\n"
      "from the input to output, but sort the order.  This program will keep\n"
      "all of the examples in memory at once, unless you use the --buffer-size option\n"
      "\n"
      "Usage:  nnet-ctc-sort-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
      "\n"
      "nnet-ctc-sort-egs --srand=1 ark:train.egs ark:sortd.egs\n";

    int32 buffer_size = 0;
    int32 srand_seed = 0;
    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("buffer-size", &buffer_size, "If >0, size of a buffer we use "
                "to do limited-memory partial sort.  Otherwise, do "
                "full sort.");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
                examples_wspecifier = po.GetArg(2);

    int64 num_done = 0;

    std::vector<std::pair<std::string, NnetCtcExample*> > egs;

    SequentialNnetCtcExampleReader example_reader(examples_rspecifier);
    NnetCtcExampleWriter example_writer(examples_wspecifier);
    int32 num_read = 0;

    if (buffer_size == 0) {  // Do full randomization
      // Putting in an extra level of indirection here to avoid excessive
      // computation and memory demands when we have to resize the vector.

      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(std::make_pair(example_reader.Key(),
                                     new NnetCtcExample(example_reader.Value())));

      std::sort(egs.begin(), egs.end(), SortNnetCtcExample);
      num_read = egs.size();
    } else {
      KALDI_ASSERT(buffer_size > 0);
      egs.resize(buffer_size,
                 std::pair<std::string, NnetCtcExample*>("",
                     static_cast<NnetCtcExample *>(NULL)));
      for (; !example_reader.Done(); example_reader.Next(), num_read++) {
        if (num_read > 0 && (num_read % buffer_size == 0)) {
          std::sort(egs.begin(), egs.end(), SortNnetCtcExample);
          for (size_t i = 0; i < egs.size(); i++) {
            example_writer.Write(egs[i].first, *(egs[i].second));
            num_done++;
          }
          num_read = 0;
        }

        int32 index = num_read;
        if (egs[index].second == NULL) {
          egs[index] = std::make_pair(example_reader.Key(),
                                      new NnetCtcExample(example_reader.Value()));
        } else {
          egs[index].first = example_reader.Key();
          *(egs[index].second) = example_reader.Value();
        }
      }
    }

    for (size_t i = 0; i < egs.size(); i++) {
      if (egs[i].second != NULL) {
        if (i < num_read) {
          example_writer.Write(egs[i].first, *(egs[i].second));
          num_done++;
        }
        delete egs[i].second;
      }
    }

    KALDI_LOG << "sortd order of " << num_done
              << " neural-network training examples "
              << (buffer_size ? "using a buffer (partial sort)" : "");

    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


