// ctc/ctc-nnet-example.cc

// Copyright 2012-2013   Johns Hopkins University (author: Daniel Povey)
//                2014   Vimal Manohar
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

#include "ctc/ctc-nnet-example.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace ctc {

void NnetCtcExample::Write(std::ostream &os, bool binary) const {
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<NnetCtcExample>");

  // just wrote <Labels>.
  WriteToken(os, binary, "<Labels>");
  WriteIntegerVector(os, binary, labels);
  WriteToken(os, binary, "<InputFrames>");
  input_frames.Write(os, binary);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context);
  WriteToken(os, binary, "<SpkInfo>");
  spk_info.Write(os, binary);
  WriteToken(os, binary, "</NnetCtcExample>");
}

void NnetCtcExample::Read(std::istream &is, bool binary) {
  // Note: weight, label, input_frames, left_context and spk_info are members.
  // This is a struct.
  ExpectToken(is, binary, "<NnetCtcExample>");

  ExpectToken(is, binary,  "<Labels>");
  ReadIntegerVector(is, binary, &labels);
  ExpectToken(is, binary, "<InputFrames>");
  input_frames.Read(is, binary);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context);
  ExpectToken(is, binary, "<SpkInfo>");
  spk_info.Read(is, binary);
  ExpectToken(is, binary, "</NnetCtcExample>");
}

static bool nnet_example_warned_left = false,
            nnet_example_warned_right = false;

void NnetCtcExample::SetLabels(const std::vector<int32> &alignment) {
  KALDI_ASSERT(alignment.size() <= input_frames.NumRows());
  labels = alignment;
}


bool SortNnetCtcExampleByNumFrames(const
                                   std::pair<std::string, NnetCtcExample>
                                   &e1,
                                   const std::pair<std::string, NnetCtcExample> &e2) {
  return e1.second.NumFrames() < e2.second.NumFrames();
}

void FrameSubsamplingShiftFeatureTimes(int32 frame_subsampling_factor,
                                       int32 frame_shift,
                                       Matrix<BaseFloat> &feature) {
  Matrix<BaseFloat> full_src(feature);
  std::vector<int32> indexs;
  for (int i = 0; i + frame_shift < full_src.NumRows();
       i += frame_subsampling_factor) {
    indexs.push_back(i + frame_shift);
  }
  if (indexs.size() == 0)
    return;

  feature.Resize(indexs.size(), full_src.NumCols());
  feature.CopyRows(full_src, &(indexs[0]));
}

void FrameSubsamplingShiftNnetCtcExampleTimes(
    int32 frame_subsampling_factor,
    int32 frame_shift,
    NnetCtcExample *eg) {
  if (frame_subsampling_factor <= 1)
    return;
  KALDI_ASSERT(frame_shift < frame_subsampling_factor);
  Matrix<BaseFloat> full_src(eg->input_frames);
  FrameSubsamplingShiftFeatureTimes(frame_subsampling_factor,
                                    frame_shift,
                                    full_src);
  eg->input_frames = full_src;
}

void ExamplesRepository::AcceptExamples(
  std::vector<NnetCtcExample> *examples) {
  KALDI_ASSERT(!examples->empty());
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  examples_.swap(*examples);
  full_semaphore_.Signal();
}

void ExamplesRepository::ExamplesDone() {
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  done_ = true;
  full_semaphore_.Signal();
}

bool ExamplesRepository::ProvideExamples(
  std::vector<NnetCtcExample> *examples) {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal();  // Increment the semaphore so
    // the call by the next thread will not block.
    return false;  // no examples to return-- all finished.
  } else {
    KALDI_ASSERT(!examples_.empty() && examples->empty());
    examples->swap(examples_);
    empty_semaphore_.Signal();
    return true;
  }
}


void DiscriminativeNnetCtcExample::Write(std::ostream &os,
    bool binary) const {
  // Note: weight, num_ali, den_lat, input_frames, left_context and spk_info are
  // members.  This is a struct.
  WriteToken(os, binary, "<DiscriminativeNnetCtcExample>");
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight);
  WriteToken(os, binary, "<NumAli>");
  WriteIntegerVector(os, binary, num_ali);
  if (!WriteCompactLattice(os, binary, den_lat)) {
    // We can't return error status from this function so we
    // throw an exception.
    KALDI_ERR << "Error writing CompactLattice to stream";
  }
  WriteToken(os, binary, "<InputFrames>");
  { // Note: this can be read as a regular matrix.
    CompressedMatrix cm(input_frames);
    cm.Write(os, binary);
  }
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context);
  WriteToken(os, binary, "<SpkInfo>");
  spk_info.Write(os, binary);
  WriteToken(os, binary, "</DiscriminativeNnetCtcExample>");
}

void DiscriminativeNnetCtcExample::Read(std::istream &is,
                                        bool binary) {
  // Note: weight, num_ali, den_lat, input_frames, left_context and spk_info are
  // members.  This is a struct.
  ExpectToken(is, binary, "<DiscriminativeNnetCtcExample>");
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight);
  ExpectToken(is, binary, "<NumAli>");
  ReadIntegerVector(is, binary, &num_ali);
  CompactLattice *den_lat_tmp = NULL;
  if (!ReadCompactLattice(is, binary, &den_lat_tmp)
      || den_lat_tmp == NULL) {
    // We can't return error status from this function so we
    // throw an exception.
    KALDI_ERR << "Error reading CompactLattice from stream";
  }
  den_lat = *den_lat_tmp;
  delete den_lat_tmp;
  ExpectToken(is, binary, "<InputFrames>");
  input_frames.Read(is, binary);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context);
  ExpectToken(is, binary, "<SpkInfo>");
  spk_info.Read(is, binary);
  ExpectToken(is, binary, "</DiscriminativeNnetCtcExample>");
}

void DiscriminativeNnetCtcExample::Check() const {
  KALDI_ASSERT(weight > 0.0);
  KALDI_ASSERT(!num_ali.empty());
  int32 num_frames = static_cast<int32>(num_ali.size());


  std::vector<int32> times;
  int32 num_frames_den = CompactLatticeStateTimes(den_lat, &times);
  KALDI_ASSERT(num_frames == num_frames_den);
  KALDI_ASSERT(input_frames.NumRows() >= left_context + num_frames);
}


}  // namespace ctc
}  // namespace kaldi
