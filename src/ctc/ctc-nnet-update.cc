// ctc/ctc-nnet-update.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)
//           2014   Xiaohui Zhang
//           2016   LingoChamp Feiteng

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

#include <numeric>

#include "util/edit-distance.h"
#include "ctc/ctc-nnet-update.h"

extern "C" {
#include "ctc.h"
}

#define WARPCTC_SAFE_CALL(fun) \
{ \
  ctcStatus_t ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "ctcStatus_t " << ret << " : \"" << ctcGetStatusString(ret) << "\" returned from '" << #fun << "'"; \
  } \
}

namespace kaldi {
namespace ctc {

using kaldi::nnet2::Component;
using kaldi::nnet2::ChunkInfo;

static void CopyToVector(const std::vector<int32> &src,
                         Vector<BaseFloat> &vec) {
  vec.Resize(src.size());
  for (int32 i = 0; i < src.size(); i++) {
    vec(i) = src[i];
  }
}

NnetCtcUpdater::NnetCtcUpdater(const Nnet &nnet,
                               Nnet *nnet_to_update):
  nnet_(nnet), nnet_to_update_(nnet_to_update) {
}

void NnetCtcUpdater::FormatInput(const std::vector<NnetCtcExample>
                                 &data) {
  forward_data_.resize(nnet_.NumComponents() + 1);
  Matrix<BaseFloat> input;
  FormatNnetInput(nnet_, data, &input);
  {
    forward_data_[0].Resize(input.NumRows(), input.NumCols(), kSetZero,
                            kStrideEqualNumCols);
    forward_data_[0].CopyFromMat(input);

    int32 num_splice = 1 + nnet_.RightContext() + nnet_.LeftContext();
    nnet_.ComputeChunkInfo(num_splice,
                           forward_data_[0].NumRows() / num_splice,
                           &chunk_info_out_);
    nnet_.SetMiniBatch(data.size());
  }
}

double NnetCtcUpdater::ComputeForMinibatch(
  const std::vector<NnetCtcExample> &data,
  double *tot_accuracy) {
  FormatInput(data);
  Propagate();
  CuMatrix<BaseFloat> tmp_deriv;
  double tot_objf = 0.0;
  bool ans = ComputeObjfAndDeriv(data, &tmp_deriv, &tot_objf,
                                 tot_accuracy);
  if (ans && nnet_to_update_ != NULL)
    Backprop(&tmp_deriv);  // this is summed (after weighting), not averaged.

  return tot_objf;
}


// form of ComputeForMinibatch for when the input data has
// already been formatted as a single matrix.
double NnetCtcUpdater::ComputeForMinibatch(const
    std::vector<NnetCtcExample>
    &data,
    Matrix<BaseFloat> *formatted_data,
    double *tot_accuracy) {
  {
    // accept the formatted input.  This replaces the call to FormatInput().
    int32 mini_batch = data.size();
    KALDI_ASSERT(formatted_data->NumRows() % mini_batch == 0 &&
                 formatted_data->NumCols() == nnet_.InputDim());

    forward_data_.resize(nnet_.NumComponents() + 1);
    {
      forward_data_[0].Resize(formatted_data->NumRows(),
                              formatted_data->NumCols(),
                              kSetZero,
                              kStrideEqualNumCols);
      forward_data_[0].CopyFromMat(*formatted_data);

      int32 num_splice = 1 + nnet_.RightContext() + nnet_.LeftContext();
      nnet_.ComputeChunkInfo(num_splice,
                             forward_data_[0].NumRows() / num_splice,
                             &chunk_info_out_);
      nnet_.SetMiniBatch(mini_batch);
    }
  }
  Propagate();
  CuMatrix<BaseFloat> tmp_deriv;
  double tot_objf = 0;
  bool ans = ComputeObjfAndDeriv(data, &tmp_deriv, &tot_objf,
                                 tot_accuracy);
  if (ans && nnet_to_update_ != NULL)
    Backprop(&tmp_deriv);  // this is summed (after weighting), not averaged.
  return tot_objf;
}

void NnetCtcUpdater::GetOutput(CuMatrix<BaseFloat> *output) {
  int32 num_components = nnet_.NumComponents();
  KALDI_ASSERT(forward_data_.size() == nnet_.NumComponents() + 1);
  *output = forward_data_[num_components];
}

void NnetCtcUpdater::Propagate() {
  int32 num_components = nnet_.NumComponents();
  {
    const Nnet &nnet = nnet_;
    static int32 num_times_printed = 0;
    for (int32 c = 0; c < num_components; c++) {
      const Component &component = nnet.GetComponent(c);
      const CuMatrix<BaseFloat> &input = forward_data_[c];
      CuMatrix<BaseFloat> &output = forward_data_[c+1];
      output.Resize(chunk_info_out_[c+1].NumRows(),
                    chunk_info_out_[c+1].NumCols(),
                    kSetZero,
                    kStrideEqualNumCols);
      KALDI_ASSERT(input.NumCols() == input.Stride());
      KALDI_ASSERT(output.NumCols() == output.Stride());
      component.Propagate(chunk_info_out_[c], chunk_info_out_[c+1], input,
                          &output);
      // If we won't need the output of the previous layer for
      // backprop, delete it to save memory.
      bool need_last_output =
        (c>0 && nnet.GetComponent(c-1).BackpropNeedsOutput()) ||
        component.BackpropNeedsInput();
      if (g_kaldi_verbose_level >= 3 && num_times_printed < 100) {
        KALDI_VLOG(3) << "Stddev of data for component " << c
                      << " for this minibatch is "
                      << (TraceMatMat(forward_data_[c], forward_data_[c], kTrans) /
                          (forward_data_[c].NumRows() * forward_data_[c].NumCols()));
        num_times_printed++;
      }
      if (!need_last_output)
        forward_data_[c].Resize(0, 0);  // We won't need this data.
    }
  }
}

bool NnetCtcUpdater::ComputeObjfAndDeriv(
  const std::vector<NnetCtcExample> &data,
  CuMatrix<BaseFloat> *deriv, double *tot_objf,
  double *tot_accuracy) const {
  BaseFloat tot_weight = 0.0;
  int32 num_components = nnet_.NumComponents();
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(output.NumCols() == output.Stride());

  int32 mini_batch = data.size();
  // prepare warp-ctc inputs
  std::vector<int32> flat_labels;
  std::vector<int32> label_lengths(mini_batch);
  std::vector<int32> input_lengths(mini_batch);
  int32 alphabet_size = nnet_.OutputDim();

  int32 ignore_frames = data[0].left_context + nnet_.RightContext();

  for (int32 m = 0; m < mini_batch; m++) {
    flat_labels.insert(flat_labels.end(), data[m].labels.begin(),
                       data[m].labels.end());
    label_lengths[m] = data[m].labels.size();
    input_lengths[m] = data[m].NumFrames() - ignore_frames;
  }

  KALDI_ASSERT(flat_labels.size() == std::accumulate(
                 label_lengths.begin(),
                 label_lengths.end(), 0));

  ctcComputeInfo info;
  info.loc = CTC_GPU;
  CU_SAFE_CALL(cudaStreamCreate(&(info.stream)));
  size_t gpu_alloc_bytes;
  WARPCTC_SAFE_CALL(get_workspace_size(label_lengths.data(),
                                       input_lengths.data(),
                                       alphabet_size, mini_batch, info,
                                       &gpu_alloc_bytes));

  char *ctc_gpu_workspace;
  CU_SAFE_CALL(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes));
  KALDI_VLOG(1) << "gpu_alloc_bytes(Mb) = " << gpu_alloc_bytes /
                1024.0 / 1024;
  Vector<BaseFloat> costs(mini_batch, kSetZero);
  if (deriv != NULL) {
    deriv->Resize(output.NumRows(), nnet_.OutputDim(), kSetZero,
                  kStrideEqualNumCols);  // sets to zero.
    WARPCTC_SAFE_CALL(compute_ctc_loss(output.Data(), deriv->Data(),
                                       flat_labels.data(), label_lengths.data(),
                                       input_lengths.data(),
                                       alphabet_size,
                                       mini_batch,
                                       costs.Data(),
                                       ctc_gpu_workspace,
                                       info));
    BaseFloat sum = deriv->Sum();
    KALDI_ASSERT(sum == sum
                 && "Error in this batch, deriv sum is inf/nan.");
  } else {
    WARPCTC_SAFE_CALL(compute_ctc_loss(output.Data(), NULL,
                                       flat_labels.data(), label_lengths.data(),
                                       input_lengths.data(),
                                       alphabet_size,
                                       mini_batch,
                                       costs.Data(),
                                       ctc_gpu_workspace,
                                       info));
  }

  KALDI_VLOG(1) << "compute_ctc_loss costs = " << costs;
  CU_SAFE_CALL(cudaFree(ctc_gpu_workspace));
  CU_SAFE_CALL(cudaStreamDestroy(info.stream));

  if (tot_accuracy != NULL) {
    *tot_accuracy = ComputeTotAccuracy(data, &tot_weight);
    KALDI_ASSERT(tot_weight == flat_labels.size());
  }
  KALDI_ASSERT(costs.Sum() == costs.Sum());

  *tot_objf = costs.Sum();

  return true;
}

double NnetCtcUpdater::ComputeTotAccuracy(
  const std::vector<NnetCtcExample> &data ,
  BaseFloat *tot_weight) const {
  const int32 blank_id = 0;
  BaseFloat tot_accuracy = 0.0, err_num = 0, tot_num = 0;
  int32 num_components = nnet_.NumComponents();
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_VLOG(2) << "---------------\n"  << "Nnet output: " << output;

  CuArray<int32> best_pdf(output.NumRows());
  std::vector<int32> best_pdf_cpu;
  output.FindRowMaxId(&best_pdf);
  best_pdf.CopyToVec(&best_pdf_cpu);

  int32 minibatch = data.size();
  KALDI_ASSERT(output.NumRows() % minibatch == 0);
  std::vector<std::vector<int32> > hyp_labels(minibatch);

  for (int32 m = 0; m < minibatch; m++) {
    tot_num += data[m].labels.size();
    KALDI_ASSERT(std::find(data[m].labels.begin(), data[m].labels.end(),
                           blank_id) == data[m].labels.end());
    for (int32 i = 0; i < data[m].NumFrames(); i++) {
      hyp_labels[m].push_back(best_pdf_cpu[i * minibatch + m]);
    }
  }

  Vector<BaseFloat> vec;

  // remove blank, and uniq labels
  for (int32 m = 0; m < minibatch; m++) {
    int32 i = 1, j = 1;
    std::vector<int32> &hyp = hyp_labels[m];
    CopyToVector(hyp, vec);
    KALDI_VLOG(2) << "Origin Hyp: " << vec;
    while (j < data[m].NumFrames()) {
      if (hyp[j] != hyp[j-1] && hyp[j] != blank_id) {
        hyp[i] = hyp[j];
        i++;
      }
      j++;
    }
    hyp.resize(i);  // at least one label
    CopyToVector(data[m].labels, vec);
    KALDI_VLOG(2) << "Train: " << vec;
    CopyToVector(hyp, vec);
    KALDI_VLOG(2) << "Hyp  : " << vec;
    err_num += LevenshteinEditDistance(data[m].labels, hyp);
    KALDI_VLOG(2) << "Err  : " << err_num;
  }

  tot_accuracy = tot_num - err_num;
  if (tot_weight != NULL)
    *tot_weight = tot_num;

  return tot_accuracy;
}


void NnetCtcUpdater::Backprop(CuMatrix<BaseFloat> *deriv) const {
  // We assume ComputeObjfAndDeriv has already been called.

  deriv->Scale(-1);
  const Nnet &nnet = nnet_;
  for (int32 c = nnet.NumComponents() - 1;
       c >= nnet.FirstUpdatableComponent(); c--) {
    const Component &component = nnet.GetComponent(c);
    KALDI_VLOG(1) << component.Info();
    Component *component_to_update = (nnet_to_update_ == NULL ? NULL :
                                      &(nnet_to_update_->GetComponent(c)));
    const CuMatrix<BaseFloat> &input = forward_data_[c],
                               &output = forward_data_[c+1];
    KALDI_ASSERT(input.NumCols() == input.Stride());
    KALDI_ASSERT(output.NumCols() == output.Stride());
    KALDI_ASSERT(deriv->NumCols() == deriv->Stride());
    CuMatrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols(),
                                    kSetZero,
                                    kStrideEqualNumCols);
    const CuMatrix<BaseFloat> &output_deriv(*deriv);
    component.Backprop(chunk_info_out_[c], chunk_info_out_[c+1], input,
                       output,
                       output_deriv, component_to_update,
                       &input_deriv);
    KALDI_VLOG(1) << "input_deriv.NumCols() = " << input_deriv.NumCols()
                  << ", input_deriv.Stride() = " << input_deriv.Stride();
    input_deriv.Swap(deriv);
  }
}


void FormatNnetInput(const Nnet &nnet,
                     const std::vector<NnetCtcExample> &data,
                     Matrix<BaseFloat> *input_mat) {
  KALDI_ASSERT(data.size() > 0);
  int32 num_splice = 1 + nnet.RightContext() + nnet.LeftContext();
  KALDI_VLOG(1) << "num_splice = " << num_splice;
  KALDI_ASSERT(data[0].input_frames.NumRows() >= num_splice);

  int32 feat_dim = data[0].input_frames.NumCols(),
        spk_dim = data[0].spk_info.Dim(),
        tot_dim = feat_dim + spk_dim;  // we append these at the neural net
  // input... note, spk_dim might be 0.
  KALDI_ASSERT(tot_dim == nnet.InputDim());
  int32 left_context = data[0].left_context;
  KALDI_ASSERT(left_context >= nnet.LeftContext());
  int32 ignore_frames = left_context - nnet.LeftContext();  // If
  // the NnetCtcExample has more left-context than we need, ignore some.
  // this may happen in settings where we increase the amount of context during
  // training, e.g. by adding layers that require more context.

  int32 num_chunks = data.size();
  int32 max_num_frames = 0;

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    int32 this_num_frames = data[chunk].input_frames.NumRows() -
                            num_splice -
                            ignore_frames + 1;
    if (this_num_frames > max_num_frames)
      max_num_frames = this_num_frames;
  }
  KALDI_VLOG(1) << "MiniBatch: " << num_chunks << ", max_num_frames is "
                <<
                max_num_frames;

  // important here, we use 'kSetZero, kStrideEqualNumCols'
  Matrix<BaseFloat> tmp_feat(max_num_frames,
                             num_chunks * tot_dim * num_splice,
                             kSetZero, kStrideEqualNumCols);
  int32 feat_dim_offset = 0;
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    Matrix<BaseFloat> full_src(data[chunk].input_frames);
    int32 this_num_frames = data[chunk].NumFrames() - num_splice -
                            ignore_frames +
                            1;
    for (int32 s = 0; s < num_splice; s++) {
      SubMatrix<BaseFloat> dest(tmp_feat,
                                0, this_num_frames,
                                feat_dim_offset, feat_dim);

      SubMatrix<BaseFloat> src(full_src, ignore_frames + s, this_num_frames,
                               0,
                               feat_dim);

      dest.CopyFromMat(src);
      feat_dim_offset += feat_dim;
      if (spk_dim != 0) {
        SubMatrix<BaseFloat> spk_dest(tmp_feat,
                                      0, this_num_frames,
                                      feat_dim_offset, spk_dim);
        spk_dest.CopyRowsFromVec(data[chunk].spk_info);
        feat_dim_offset += spk_dim;
      }
    }
  }
  // Resize()
  input_mat->Resize(max_num_frames * num_splice * num_chunks, tot_dim,
                    kSetZero, kStrideEqualNumCols);
  memcpy(input_mat->Data(), tmp_feat.Data(),
         sizeof(BaseFloat) * tmp_feat.NumRows() * tmp_feat.NumCols());
  KALDI_VLOG(1) << "InputMat: NumRows = " << input_mat->NumRows() <<
                ", NumCols = " <<
                input_mat->NumCols();
  KALDI_VLOG(5) << "InputMat: " << *input_mat;
}

double ComputeNnetObjf(const Nnet &nnet,
                       const std::vector<NnetCtcExample> &examples,
                       double *tot_accuracy) {
  NnetCtcUpdater updater(nnet, NULL);
  return updater.ComputeForMinibatch(examples, tot_accuracy);
}

double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetCtcExample> &examples,
                  Nnet *nnet_to_update,
                  double *tot_accuracy) {
  try {
    NnetCtcUpdater updater(nnet, nnet_to_update);
    return updater.ComputeForMinibatch(examples, tot_accuracy);
  } catch(...) {
    KALDI_LOG << "Error doing backprop.";
    throw;
  }
}

// version of DoBackprop that takes already-formatted examples.
double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetCtcExample> &examples,
                  Matrix<BaseFloat> *examples_formatted,
                  Nnet *nnet_to_update,
                  double *tot_accuracy) {
  try {
    KALDI_ASSERT(examples_formatted->NumCols() ==
                 examples_formatted->Stride());
    NnetCtcUpdater updater(nnet, nnet_to_update);
    return updater.ComputeForMinibatch(examples,
                                       examples_formatted,
                                       tot_accuracy);
  } catch(...) {
    KALDI_LOG << "Error doing backprop.";
    throw;
  }
}

BaseFloat TotalNnetTrainingWeight(const std::vector<NnetCtcExample>
                                  &egs) {
  double ans = 0.0;
  for (size_t i = 0; i < egs.size(); i++)
    ans += egs[i].labels.size();  // assume no blank in labels
  return ans;
}

double ComputeNnetObjf(
  const Nnet &nnet,
  const std::vector<NnetCtcExample> &validation_set,
  int32 batch_size,
  double *tot_accuracy) {
  double tot_accuracy_tmp;
  if (tot_accuracy)
    *tot_accuracy = 0.0;
  std::vector<NnetCtcExample> batch;
  batch.reserve(batch_size);
  double tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set.size()));
         i++) {
      batch.push_back(validation_set[i]);
    }
    tot_objf += ComputeNnetObjf(nnet, batch,
                                tot_accuracy != NULL ? &tot_accuracy_tmp : NULL);
    if (tot_accuracy)
      *tot_accuracy += tot_accuracy_tmp;
  }
  return tot_objf;
}

}  // namespace ctc
}  // namespace kaldi
