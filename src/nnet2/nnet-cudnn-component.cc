// nnet2/nnet-cudnn-component.cc

// Copyright 2016  LingoChamp Feiteng


#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
#include <numeric>
#include <functional>
#include <iomanip>

#include <cudnn.h>

#include "nnet2/nnet-cudnn-component.h"
#include "cudamatrix/cudnn-recurrent.h"

namespace kaldi {
namespace nnet2 {

static void PrintParameterStats(std::ostringstream &os,
                                const std::string &name,
                                const CuVector<BaseFloat> &params,
                                bool include_mean = false) {
  os << std::setprecision(4);
  os << ", " << name << '-';
  if (include_mean) {
    BaseFloat mean = params.Sum() / params.Dim(),
              stddev = std::sqrt(VecVec(params, params) / params.Dim() - mean * mean);
    os << "{mean,stddev}=" << mean << ',' << stddev;
  } else {
    BaseFloat rms = std::sqrt(VecVec(params, params) / params.Dim());
    os << "rms=" << rms;
  }
  os << std::setprecision(6);  // restore the default precision.
}

static void PrintParameterStats(std::ostringstream &os,
                                const std::string &name,
                                const CuMatrix<BaseFloat> &params,
                                bool include_mean = false) {

  os << std::setprecision(4);
  os << ", " << name << '-';
  int32 dim = params.NumRows() * params.NumCols();
  if (include_mean) {
    BaseFloat mean = params.Sum() / dim,
              stddev = std::sqrt(TraceMatMat(params, params, kTrans) / dim -
                                 mean * mean);
    os << "{mean,stddev}=" << mean << ',' << stddev;
  } else {
    BaseFloat rms = std::sqrt(TraceMatMat(params, params, kTrans) / dim);
    os << "rms=" << rms;
  }
  os << std::setprecision(6);  // restore the default precision.
}

std::string CuDNNRecurrentComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", input-dim=" << input_dim_
         << ", hidden-dim=" << hidden_dim_
         << ", num-layers=" << num_layers_
         << ", max-seq-length=" << max_seq_length_
         << ", mini-batch=" << mini_batch_;
  if (bidirectional_)
    stream << ", BIDIRECTIONAL";
  else
    stream << ", UNIDIRECTIONAL";
  PrintParameterStats(stream, "filter-params", filter_params_, true);
  return stream.str();
}

void CuDNNRecurrentComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;

  ok = ok && ParseFromString("learning-rate", &args, &learning_rate_);
  ok = ok && ParseFromString("num-layers", &args, &num_layers_);
  ok = ok && ParseFromString("input-dim", &args, &input_dim_);
  ok = ok && ParseFromString("output-dim", &args, &hidden_dim_);
  ok = ok && ParseFromString("rnn-mode", &args, &rnn_mode_);
  ok = ok && ParseFromString("bidirectional", &args, &bidirectional_);
  ok = ok && ParseFromString("max-seq-length", &args, &max_seq_length_);

  ParseFromString("param-stddev", &args, &param_stddev_);
  ParseFromString("bias-stddev", &args, &bias_stddev_);
  ParseFromString("clip-gradient", &args, &clip_gradient_);

  if (!ok) {
    KALDI_ERR << "Bad initializer " << orig_args;
  }
  int32 mini_batch = 0;
  if (ParseFromString("mini-batch", &args, &mini_batch)) {
    KALDI_ASSERT(mini_batch > 0);
    Init(mini_batch);
  } else {
    Init(1, max_seq_length_);
  }
}

void CuDNNRecurrentComponent::InitMiniBatch(int32 mini_batch, int32 seq_length) const {
  Init(mini_batch, seq_length);
}

void CuDNNRecurrentComponent::Init(int32 mini_batch, int32 seq_length) const {
  if ((mini_batch_ == mini_batch) && (seq_length == 0))
    return;
  KALDI_LOG << "Init RNN.";

  if (mini_batch_ != 0) {
    DestroyDescriptors();
    mini_batch_ = 0;
  }
  if (mini_batch <= 0) {
    KALDI_WARN << "mini_batch is " << mini_batch;
    return;
  }

  mini_batch_ = mini_batch;
  KALDI_ASSERT(mini_batch_ > 0);
  if (seq_length != 0) {
    max_seq_length_ = seq_length;
  }
  KALDI_ASSERT(max_seq_length_ > 0);
  KALDI_LOG << Info();
  // -------------------------
  // Set up inputs and outputs
  // -------------------------
  hx_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
             1));
  cx_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
             1));
  dhx_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
              1));
  dcx_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
              1));

  hy_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
             1));
  cy_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
             1));
  dhy_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
              1));
  dcy_.Resize(num_layers_ * hidden_dim_ * mini_batch_ * (bidirectional_ ? 2 :
              1));

  x_desc_ = new cudnnTensorDescriptor_t[max_seq_length_];
  y_desc_ = new cudnnTensorDescriptor_t[max_seq_length_];
  dx_desc_ = new cudnnTensorDescriptor_t[max_seq_length_];
  dy_desc_ = new cudnnTensorDescriptor_t[max_seq_length_];

  int dimA[3];
  int strideA[3];
  KALDI_ASSERT(cudnn::GetDataType() == CUDNN_DATA_FLOAT);
  // In this example dimA[1] is constant across the whole sequence
  // This isn't required, all that is required is that it does not increase.
  for (int i = 0; i < max_seq_length_; i++) {
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&x_desc_[i]));
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&y_desc_[i]));
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dx_desc_[i]));
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dy_desc_[i]));

    dimA[0] = mini_batch_;
    dimA[1] = input_dim_;
    dimA[2] = 1;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(x_desc_[i], cudnn::GetDataType(),
                    3,
                    dimA, strideA));
    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(dx_desc_[i], cudnn::GetDataType(),
                    3,
                    dimA, strideA));

    dimA[0] = mini_batch_;
    dimA[1] = bidirectional_ ? (hidden_dim_ * 2) : hidden_dim_;
    dimA[2] = 1;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(y_desc_[i], cudnn::GetDataType(),
                    3,
                    dimA, strideA));
    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(dy_desc_[i], cudnn::GetDataType(),
                    3,
                    dimA, strideA));
  }

  dimA[0] = num_layers_ * (bidirectional_ ? 2 : 1);
  dimA[1] = mini_batch_;
  dimA[2] = hidden_dim_;

  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;

  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&hx_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&cx_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&hy_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&cy_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dhx_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dcx_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dhy_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&dcy_desc_));

  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(hx_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(cx_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(hy_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(cy_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(dhx_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(dcx_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(dhy_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(dcy_desc_, cudnn::GetDataType(), 3,
                  dimA, strideA));

  // -------------------------
  // Set up the dropout descriptor (needed for the RNN descriptor)
  // -------------------------
  unsigned long long seed = 1337ull; // Pick a seed
  CUDNN_SAFE_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_));

  // How much memory does dropout need for states?
  // These states are used to generate random numbers internally
  // and should not be freed until the RNN descriptor is no longer used
  size_t state_size;
  CUDNN_SAFE_CALL(cudnnDropoutGetStatesSize(
                    CuDevice::Instantiate().GetCudnnHandle(), &state_size));
  KALDI_ASSERT(state_size > 0);
  KALDI_ASSERT(state_size % sizeof(BaseFloat) == 0);
  dropout_states_.Resize(state_size / sizeof(BaseFloat));

  CUDNN_SAFE_CALL(cudnnSetDropoutDescriptor(dropout_desc_,
                  CuDevice::Instantiate().GetCudnnHandle(),
                  0, dropout_states_.Data(),
                  state_size, seed));
  KALDI_VLOG(5) << "dropout_states_: " << dropout_states_;
  // -------------------------
  // Set up the RNN descriptor
  // -------------------------
  CUDNN_SAFE_CALL(cudnnCreateRNNDescriptor(&rnn_desc_));
  cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
  if      (rnn_mode_ == 0) rnn_mode = CUDNN_RNN_RELU;
  else if (rnn_mode_ == 1) rnn_mode = CUDNN_RNN_TANH;
  else if (rnn_mode_ == 2) rnn_mode = CUDNN_LSTM;
  else if (rnn_mode_ == 3) rnn_mode = CUDNN_GRU;
  else {
    KALDI_ERR << "rnn_mode_ = " << rnn_mode_ << ", should in [0, 1, 2, 3].";
  }

  CUDNN_SAFE_CALL(cudnnSetRNNDescriptor(rnn_desc_, hidden_dim_, num_layers_,
                                        dropout_desc_,
                                        CUDNN_LINEAR_INPUT,
                                        bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                        rnn_mode, cudnn::GetDataType()));

  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&w_desc_));
  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&dw_desc_));

  size_t weights_size;
  CUDNN_SAFE_CALL(cudnnGetRNNParamsSize(
                    CuDevice::Instantiate().GetCudnnHandle(),
                    rnn_desc_, x_desc_[0], &weights_size, cudnn::GetDataType()));

  KALDI_ASSERT(weights_size % sizeof(BaseFloat) == 0);
  int dimW[3];
  dimW[0] =  weights_size / sizeof(BaseFloat);
  dimW[1] = 1;
  dimW[2] = 1;
  // TODO check CUDNN_TENSOR_NCHW
  CUDNN_SAFE_CALL(cudnnSetFilterNdDescriptor(w_desc_, cudnn::GetDataType(),
                  CUDNN_TENSOR_NCHW, 3, dimW));
  CUDNN_SAFE_CALL(cudnnSetFilterNdDescriptor(dw_desc_, cudnn::GetDataType(),
                  CUDNN_TENSOR_NCHW, 3, dimW));

  // Initialise weights
  filter_params_grad_.Resize(weights_size / sizeof(BaseFloat), kSetZero);

  // -------------------------
  // Set up work space and reserved memory
  // -------------------------
  size_t work_size;
  size_t reserve_size;

  // Need for every pass
  CUDNN_SAFE_CALL(cudnnGetRNNWorkspaceSize(
                    CuDevice::Instantiate().GetCudnnHandle(),
                    rnn_desc_, max_seq_length_, x_desc_, &work_size));
  // Only needed in training, shouldn't be touched between passes.
  CUDNN_SAFE_CALL(cudnnGetRNNTrainingReserveSize(
                    CuDevice::Instantiate().GetCudnnHandle(),
                    rnn_desc_, max_seq_length_, x_desc_, &reserve_size));
  KALDI_LOG << "work_size = " << work_size
            << "\nreserve_size = " << reserve_size;

  KALDI_ASSERT(work_size % sizeof(BaseFloat) == 0);
  KALDI_ASSERT(reserve_size % sizeof(BaseFloat) == 0);
  KALDI_ASSERT(work_size > 0);
  KALDI_ASSERT(reserve_size > 0);

  work_space_.Resize(work_size / sizeof(BaseFloat), kSetZero);
  work_space_size_ = work_size;
  reserve_space_.Resize(reserve_size / sizeof(BaseFloat), kSetZero);
  reserve_space_size_ = reserve_size;


  if (filter_params_.Dim() != weights_size / sizeof(BaseFloat)) {
    KALDI_ASSERT(filter_params_.Dim() == 0);
    KALDI_LOG << "Initialise weights.";
    filter_params_.Resize(weights_size / sizeof(BaseFloat));
    // filter_params_.SetRandn();
    // if (param_stddev_ == 0)
    //   param_stddev_ = 1.0 / (input_dim_ * hidden_dim_);
    // filter_params_.Scale(param_stddev_);

    // Weights
    int numLinearLayers = 0;
    if (rnn_mode == CUDNN_RNN_RELU || rnn_mode == CUDNN_RNN_TANH) {
      numLinearLayers = 2;
    } else if (rnn_mode == CUDNN_LSTM) {
      numLinearLayers = 8;
    } else if (rnn_mode == CUDNN_GRU) {
      numLinearLayers = 6;
    }

    for (int32 layer = 0; layer < num_layers_ * (bidirectional_ ? 2 : 1); layer++) {
      for (int32 linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
        cudnnFilterDescriptor_t linLayerMatDesc;
        CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&linLayerMatDesc));
        float *linLayerMat;

        CUDNN_SAFE_CALL(cudnnGetRNNLinLayerMatrixParams(CuDevice::Instantiate().GetCudnnHandle(),
                        rnn_desc_,
                        layer,
                        x_desc_[0],
                        w_desc_,
                        filter_params_.Data(),
                        linLayerID,
                        linLayerMatDesc,
                        (void**)&linLayerMat));

        cudnnDataType_t dataType;
        cudnnTensorFormat_t format;
        int nbDims;
        int filterDimA[3];
        CUDNN_SAFE_CALL(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                        3,
                        &dataType,
                        &format,
                        &nbDims,
                        filterDimA));
        int32 num_params = filterDimA[0] * filterDimA[1] * filterDimA[2];
        KALDI_LOG << "layer " << layer << ", linLayerID " << linLayerID
                  << ", FilterDimA(" << filterDimA[0] << ", " << filterDimA[1] << ", " << filterDimA[2] << ")";
        KALDI_ASSERT(num_params > 0);
        Vector<BaseFloat> linear(num_params);
        linear.SetRandn();
        linear.Scale(param_stddev_);
        KALDI_ASSERT(linear.Dim() == num_params);
        CU_SAFE_CALL(cudaMemcpy(linLayerMat, linear.Data(),
                                sizeof(BaseFloat) * linear.Dim(), cudaMemcpyHostToDevice));

        CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(linLayerMatDesc));


        cudnnFilterDescriptor_t linLayerBiasDesc;
        CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
        float *linLayerBias;

        CUDNN_SAFE_CALL(cudnnGetRNNLinLayerBiasParams(CuDevice::Instantiate().GetCudnnHandle(),
                        rnn_desc_,
                        layer,
                        x_desc_[0],
                        w_desc_,
                        filter_params_.Data(),
                        linLayerID,
                        linLayerBiasDesc,
                        (void**)&linLayerBias));

        CUDNN_SAFE_CALL(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                        3,
                        &dataType,
                        &format,
                        &nbDims,
                        filterDimA));
        num_params = filterDimA[0] * filterDimA[1] * filterDimA[2];
        KALDI_LOG << "layer " << layer << ", linLayerID " << linLayerID
                  << ", BiasDimA(" << filterDimA[0] << ", " << filterDimA[1] << ", " << filterDimA[2] << ")";
        KALDI_ASSERT(num_params > 0);
        linear.Resize(num_params);
        linear.SetRandn();
        linear.Set(bias_stddev_);
        CU_SAFE_CALL(cudaMemcpy(linLayerBias, linear.Data(), sizeof(BaseFloat) * linear.Dim(),
                                cudaMemcpyHostToDevice));

        CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
      }
    }
    BaseFloat sum = filter_params_.Sum();
    KALDI_ASSERT(sum == sum);
  } else {
    KALDI_LOG << "Already has weights.";
  }

}

void CuDNNRecurrentComponent::GetFilterBias(int32 layer, int linLayerID, Vector<BaseFloat> &filter,
    Vector<BaseFloat> &bias) {
  KALDI_ASSERT(layer >= 0 && layer < num_layers_ * (bidirectional_ ? 2 : 1));
  KALDI_ASSERT(linLayerID < 8);
  cudnnFilterDescriptor_t linLayerMatDesc;
  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&linLayerMatDesc));
  float *linLayerMat;

  CUDNN_SAFE_CALL(cudnnGetRNNLinLayerMatrixParams(CuDevice::Instantiate().GetCudnnHandle(),
                  rnn_desc_,
                  layer,
                  x_desc_[0],
                  w_desc_,
                  filter_params_.Data(),
                  linLayerID,
                  linLayerMatDesc,
                  (void**)&linLayerMat));

  cudnnDataType_t dataType;
  cudnnTensorFormat_t format;
  int nbDims;
  int filterDimA[3];
  CUDNN_SAFE_CALL(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                  3,
                  &dataType,
                  &format,
                  &nbDims,
                  filterDimA));
  int32 num_params = filterDimA[0] * filterDimA[1] * filterDimA[2];
  KALDI_VLOG(2) << "layer " << layer << ", linLayerID " << linLayerID
                << ", FilterDimA(" << filterDimA[0] << ", " << filterDimA[1] << ", " << filterDimA[2] << ")";
  filter.Resize(num_params);
  CU_SAFE_CALL(cudaMemcpy(filter.Data(), linLayerMat,
                          sizeof(BaseFloat) * filter.Dim(), cudaMemcpyDeviceToHost));

  CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(linLayerMatDesc));


  cudnnFilterDescriptor_t linLayerBiasDesc;
  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
  float *linLayerBias;

  CUDNN_SAFE_CALL(cudnnGetRNNLinLayerBiasParams(CuDevice::Instantiate().GetCudnnHandle(),
                  rnn_desc_,
                  layer,
                  x_desc_[0],
                  w_desc_,
                  filter_params_.Data(),
                  linLayerID,
                  linLayerBiasDesc,
                  (void**)&linLayerBias));

  CUDNN_SAFE_CALL(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                  3,
                  &dataType,
                  &format,
                  &nbDims,
                  filterDimA));
  num_params = filterDimA[0] * filterDimA[1] * filterDimA[2];
  KALDI_VLOG(2) << "layer " << layer << ", linLayerID " << linLayerID
                << ", BiasDimA(" << filterDimA[0] << ", " << filterDimA[1] << ", " << filterDimA[2] << ")";
  bias.Resize(num_params);
  CU_SAFE_CALL(cudaMemcpy(bias.Data(), linLayerBias, sizeof(BaseFloat) * bias.Dim(),
                          cudaMemcpyDeviceToHost));

  CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
}


CuDNNRecurrentComponent::CuDNNRecurrentComponent() :
  UpdatableComponent(),
  input_dim_(0), hidden_dim_(0), num_layers_(0), max_seq_length_(2000),
  bidirectional_(true), is_gradient_(false),
  rnn_mode_(2), mini_batch_(0), param_stddev_(0.02), bias_stddev_(0.2),
  clip_gradient_(5.0) {
}

void CuDNNRecurrentComponent::SetBufferZero() const {
  hx_.SetZero();
  cx_.SetZero();
  hy_.SetZero();
  cy_.SetZero();
  dhx_.SetZero();
  dcx_.SetZero();
  dhy_.SetZero();
  dcy_.SetZero();
  work_space_.SetZero();
  reserve_space_.SetZero();
  filter_params_grad_.SetZero();
}

void CuDNNRecurrentComponent::Propagate(const ChunkInfo &in_info,
                                        const ChunkInfo &out_info,
                                        const CuMatrixBase<BaseFloat> &in,
                                        CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == in.Stride());
  KALDI_ASSERT(out->NumCols() == out->Stride());

  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  if (mini_batch_ == 0) {
    InitMiniBatch(1);
  }

  int seq_length = in.NumRows() / mini_batch_;
  KALDI_ASSERT(in.NumCols() == input_dim_);
  KALDI_ASSERT(in.NumRows() % mini_batch_ == 0);
  KALDI_ASSERT(out->NumCols() == hidden_dim_ * (bidirectional_ ? 2 : 1));

  KALDI_VLOG(2) << Info();
  if (seq_length > max_seq_length_) {
    KALDI_VLOG(1) << "Reinit Descriptors; new seq length = " << seq_length;
    InitMiniBatch(mini_batch_, seq_length);
  }

  SetBufferZero();

  if (mini_batch_ == 1) {
    cudnn::RecurrentForwardInference(
      CuDevice::Instantiate().GetCudnnHandle(),
      rnn_desc_, seq_length,
      x_desc_, in.Data(),
      hx_desc_, hx_.Data(), cx_desc_, cx_.Data(),
      w_desc_, filter_params_.Data(),
      y_desc_, out->Data(),
      hy_desc_, hy_.Data(), cy_desc_, cy_.Data(),
      work_space_.Data(), work_space_size_);
  } else {
    cudnn::RecurrentForwardTraining(
      CuDevice::Instantiate().GetCudnnHandle(),
      rnn_desc_, seq_length,
      x_desc_, in.Data(),
      hx_desc_, hx_.Data(), cx_desc_, cx_.Data(),
      w_desc_, filter_params_.Data(),
      y_desc_, out->Data(),
      hy_desc_, hy_.Data(), cy_desc_, cy_.Data(),
      work_space_.Data(), work_space_size_, reserve_space_.Data(),
      reserve_space_size_);
  }
}

void CuDNNRecurrentComponent::Backprop(const ChunkInfo &in_info,
                                       const ChunkInfo &out_info,
                                       const CuMatrixBase<BaseFloat> &in, // in_value
                                       const CuMatrixBase<BaseFloat> &out, // out_value
                                       const CuMatrixBase<BaseFloat> &out_deriv,
                                       Component *to_update_in,
                                       CuMatrix<BaseFloat> *in_deriv) const {
  int seq_length = in.NumRows() / mini_batch_;
  KALDI_ASSERT(mini_batch_ > 0);
  KALDI_ASSERT(seq_length > 0 && seq_length <= max_seq_length_);
  KALDI_ASSERT(in.NumCols() == InputDim());

  KALDI_ASSERT(in.NumCols() == in.Stride());
  KALDI_ASSERT(out.NumCols() == out.Stride());
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride());

  BaseFloat f = filter_params_.Sum();
  KALDI_ASSERT(f == f);
  if (in_deriv != NULL) {
    cudnn::RecurrentBackwardData(
      CuDevice::Instantiate().GetCudnnHandle(),
      rnn_desc_, seq_length,
      y_desc_, out.Data(), dy_desc_, out_deriv.Data(),
      dhy_desc_, dhy_.Data(), dcy_desc_, dcy_.Data(),
      w_desc_, filter_params_.Data(),
      hx_desc_, hx_.Data(), cx_desc_, cx_.Data(),
      dx_desc_, in_deriv->Data(),
      dhx_desc_, dhx_.Data(), dcx_desc_, dcx_.Data(),
      work_space_.Data(), work_space_size_,
      reserve_space_.Data(), reserve_space_size_);
  }

  if (to_update_in != NULL) {
    KALDI_ASSERT(in_deriv != NULL);
    KALDI_VLOG(1) << "out_deriv: Min = " << out_deriv.Min()
                  << ", Max = " << out_deriv.Max();
    KALDI_ASSERT(filter_params_grad_.Sum() == 0);
    cudnn::RecurrentBackwardWeights(CuDevice::Instantiate().GetCudnnHandle(),
                                    rnn_desc_, seq_length,
                                    x_desc_, in.Data(), hx_desc_, hx_.Data(), y_desc_, out.Data(),
                                    work_space_.Data(), work_space_size_, dw_desc_, filter_params_grad_.Data(),
                                    reserve_space_.Data(), reserve_space_size_);
    KALDI_VLOG(1) << "filter_params_grad_: min = " << filter_params_grad_.Min()
                  << ", max = " << filter_params_grad_.Max();
    filter_params_grad_.ApplyFloor(-clip_gradient_);
    filter_params_grad_.ApplyCeiling(clip_gradient_);

    CuDNNRecurrentComponent *to_update =
      dynamic_cast<CuDNNRecurrentComponent*>(to_update_in);
    KALDI_ASSERT(to_update != NULL);
    to_update->Update(filter_params_grad_);
  }
}

void CuDNNRecurrentComponent::Update(const CuVector<BaseFloat> &filter_params_grad) {
  filter_params_.AddVec(learning_rate_, filter_params_grad);
}

void CuDNNRecurrentComponent::DestroyDescriptors() const {
  KALDI_ASSERT(mini_batch_ > 0);
  for (int32 i = 0; i < max_seq_length_; i++) {
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(x_desc_[i]));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dx_desc_[i]));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(y_desc_[i]));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dy_desc_[i]));
  }
  delete x_desc_;
  delete dx_desc_;
  delete y_desc_;
  delete dy_desc_;

  CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(w_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(dw_desc_));

  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(hx_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(cx_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(hy_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(cy_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dhx_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dcx_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dhy_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(dcy_desc_));

  CUDNN_SAFE_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyRNNDescriptor(rnn_desc_));
}

CuDNNRecurrentComponent::~CuDNNRecurrentComponent() {
  if (mini_batch_ != 0)
    DestroyDescriptors();
}

CuDNNRecurrentComponent::CuDNNRecurrentComponent(const CuDNNRecurrentComponent
    &other): UpdatableComponent(other) {
  mini_batch_ = 0;
  input_dim_ = other.input_dim_;
  hidden_dim_ = other.hidden_dim_;
  num_layers_ = other.num_layers_;
  bidirectional_ = other.bidirectional_;
  rnn_mode_ = other.rnn_mode_;
  max_seq_length_ = other.max_seq_length_;
  filter_params_ = other.filter_params_;
  is_gradient_ = other.is_gradient_;
  clip_gradient_ = other.clip_gradient_;

  param_stddev_ = other.param_stddev_;
  bias_stddev_ = other.bias_stddev_;

  // Init(other.mini_batch_, max_seq_length_);
}

Component * CuDNNRecurrentComponent::Copy() const {
  return new CuDNNRecurrentComponent(*this);
}

void CuDNNRecurrentComponent::Read(std::istream &is, bool binary) {
  // ExpectToken(is, binary, "<CuDNNRecurrentComponent>");
  ExpectToken(is, binary, "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "<ClipGradient>");
  ReadBasicType(is, binary, &clip_gradient_);
  ExpectToken(is, binary, "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<HiddenDim>");
  ReadBasicType(is, binary, &hidden_dim_);
  ExpectToken(is, binary, "<NumLayers>");
  ReadBasicType(is, binary, &num_layers_);
  ExpectToken(is, binary, "<Bidirectional>");
  ReadBasicType(is, binary, &bidirectional_);
  ExpectToken(is, binary, "<RNNMode>");
  ReadBasicType(is, binary, &rnn_mode_);
  ExpectToken(is, binary, "<MaxSeqLength>");
  ReadBasicType(is, binary, &max_seq_length_);
  ExpectToken(is, binary, "<FilterParams>");
  filter_params_.Read(is, binary);
  ExpectToken(is, binary, "</CuDNNRecurrentComponent>");
}

void CuDNNRecurrentComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<CuDNNRecurrentComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<ClipGradient>");
  WriteBasicType(os, binary, clip_gradient_);
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<HiddenDim>");
  WriteBasicType(os, binary, hidden_dim_);
  WriteToken(os, binary, "<NumLayers>");
  WriteBasicType(os, binary, num_layers_);
  WriteToken(os, binary, "<Bidirectional>");
  WriteBasicType(os, binary, bidirectional_);
  WriteToken(os, binary, "<RNNMode>");
  WriteBasicType(os, binary, rnn_mode_);
  WriteToken(os, binary, "<MaxSeqLength>");
  WriteBasicType(os, binary, max_seq_length_);
  WriteToken(os, binary, "<FilterParams>");
  filter_params_.Write(os, binary);
  WriteToken(os, binary, "</CuDNNRecurrentComponent>");
}

void CuDNNRecurrentComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    is_gradient_ = true;
  }
  filter_params_.SetZero();
  SetBufferZero();
}

BaseFloat CuDNNRecurrentComponent::DotProduct(const UpdatableComponent
    &other_in)
const {
  const CuDNNRecurrentComponent *other =
    dynamic_cast<const CuDNNRecurrentComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return VecVec(filter_params_, other->filter_params_);
}

int32 CuDNNRecurrentComponent::NumParameters() const {
  return filter_params_.Dim();
}

void CuDNNRecurrentComponent::Scale(BaseFloat scale) {
  filter_params_.Scale(scale);
}

void CuDNNRecurrentComponent::Add(BaseFloat alpha,
                                  const UpdatableComponent &other_in) {
  const CuDNNRecurrentComponent *other =
    dynamic_cast<const CuDNNRecurrentComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  filter_params_.AddVec(alpha, other->filter_params_);
}

void CuDNNRecurrentComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->CopyFromVec(filter_params_);
}

void CuDNNRecurrentComponent::UnVectorize(const VectorBase<BaseFloat>
    &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  filter_params_.CopyFromVec(params);
}

void CuDNNRecurrentComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_bias_params(filter_params_);
  temp_bias_params.SetRandn();
  filter_params_.AddVec(stddev, temp_bias_params);
}


void ClipGradientComponent::Read(std::istream &is, bool binary) {
  // ExpectToken(is, binary, "<ClipGradientComponent>");
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<ClippingThreshold>");
  ReadBasicType(is, binary, &clipping_threshold_);
  ExpectToken(is, binary, "<NormBasedClipping>");
  ReadBasicType(is, binary, &norm_based_clipping_);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<SelfRepairClippedProportionThreshold>") {
    ReadBasicType(is, binary, &self_repair_clipped_proportion_threshold_);
    ExpectToken(is, binary, "<SelfRepairTarget>");
    ReadBasicType(is, binary, &self_repair_target_);
    ExpectToken(is, binary, "<SelfRepairScale>");
    ReadBasicType(is, binary, &self_repair_scale_);
    ExpectToken(is, binary, "<NumElementsClipped>");
  } else {
    self_repair_clipped_proportion_threshold_ = 1.0;
    self_repair_target_ = 0.0;
    self_repair_scale_ = 0.0;
    KALDI_ASSERT(token == "<NumElementsClipped>");
  }
  ReadBasicType(is, binary, &num_clipped_);
  ExpectToken(is, binary, "<NumElementsProcessed>");
  ReadBasicType(is, binary, &count_);
  ReadToken(is, binary, &token);
  if (token == "<NumSelfRepaired>") {
    ReadBasicType(is, binary, &num_self_repaired_);
    ExpectToken(is, binary, "<NumBackpropped>");
    ReadBasicType(is, binary, &num_backpropped_);
    ExpectToken(is, binary, "</ClipGradientComponent>");
  } else {
    num_self_repaired_ = 0;
    num_backpropped_ = 0;
    KALDI_ASSERT(token == "</ClipGradientComponent>");
  }
}

void ClipGradientComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ClipGradientComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<ClippingThreshold>");
  WriteBasicType(os, binary, clipping_threshold_);
  WriteToken(os, binary, "<NormBasedClipping>");
  WriteBasicType(os, binary, norm_based_clipping_);
  WriteToken(os, binary, "<SelfRepairClippedProportionThreshold>");
  WriteBasicType(os, binary, self_repair_clipped_proportion_threshold_);
  WriteToken(os, binary, "<SelfRepairTarget>");
  WriteBasicType(os, binary, self_repair_target_);
  WriteToken(os, binary, "<SelfRepairScale>");
  WriteBasicType(os, binary, self_repair_scale_);
  WriteToken(os, binary, "<NumElementsClipped>");
  WriteBasicType(os, binary, num_clipped_);
  WriteToken(os, binary, "<NumElementsProcessed>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<NumSelfRepaired>");
  WriteBasicType(os, binary, num_self_repaired_);
  WriteToken(os, binary, "<NumBackpropped>");
  WriteBasicType(os, binary, num_backpropped_);
  WriteToken(os, binary, "</ClipGradientComponent>");
}

std::string ClipGradientComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", norm-based-clipping="
         << (norm_based_clipping_ ? "true" : "false")
         << ", clipping-threshold=" << clipping_threshold_
         << ", clipped-proportion="
         << (count_ > 0 ? static_cast<BaseFloat>(num_clipped_)/count_ : 0);
  if (self_repair_scale_ != 0.0)
    stream << ", self-repair-clipped-proportion-threshold="
           << self_repair_clipped_proportion_threshold_
           << ", self-repair-target=" << self_repair_target_
           << ", self-repair-scale=" << self_repair_scale_;
  return stream.str();
}

void ClipGradientComponent::Init(int32 dim,
                                 BaseFloat clipping_threshold,
                                 bool norm_based_clipping,
                                 BaseFloat self_repair_clipped_proportion_threshold,
                                 BaseFloat self_repair_target,
                                 BaseFloat self_repair_scale,
                                 int32 num_clipped,
                                 int32 count,
                                 int32 num_self_repaired,
                                 int32 num_backpropped)  {
  KALDI_ASSERT(clipping_threshold >= 0 && dim > 0 &&
      self_repair_clipped_proportion_threshold >= 0.0 && 
      self_repair_target >= 0.0 && self_repair_scale >= 0.0);
  dim_ = dim;
  norm_based_clipping_ = norm_based_clipping;
  clipping_threshold_ = clipping_threshold;
  self_repair_clipped_proportion_threshold_ =
      self_repair_clipped_proportion_threshold;
  self_repair_target_ = self_repair_target;
  self_repair_scale_ = self_repair_scale;
  num_clipped_ = num_clipped;
  count_ = count;
  num_self_repaired_ = num_self_repaired;
  num_backpropped_ = num_backpropped;
}

void ClipGradientComponent::InitFromString(std::string args) {
  std::string orig_args(args);

  bool ok = true;
  int32 dim = 0;
  ok = ok && ParseFromString("dim", &args, &dim);

  bool norm_based_clipping = false;
  BaseFloat clipping_threshold = 15.0;
  BaseFloat self_repair_clipped_proportion_threshold = 0.01;
  BaseFloat self_repair_target = 0.0;
  BaseFloat self_repair_scale = 1.0;
  ParseFromString("clipping-threshold", &args, &clipping_threshold);
  ParseFromString("norm-based-clipping", &args, &norm_based_clipping);
  ParseFromString("self-repair-clipped-proportion-threshold", &args,
                &self_repair_clipped_proportion_threshold);
  ParseFromString("self-repair-target", &args,
                &self_repair_target);
  ParseFromString("self-repair-scale", &args, &self_repair_scale);
  if (!ok || !args.empty() ||
      clipping_threshold < 0 || dim <= 0 ||
      self_repair_clipped_proportion_threshold < 0.0 || 
      self_repair_target < 0.0 || self_repair_scale < 0.0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, clipping_threshold, norm_based_clipping,
       self_repair_clipped_proportion_threshold,
       self_repair_target,
       self_repair_scale, 0, 0, 0, 0);
}

void ClipGradientComponent::Propagate(const ChunkInfo &in_info,
                                 const ChunkInfo &out_info,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  // TODO clip the activations of memory cells to range [-50, 50]
  out->CopyFromMat(in);
}


void ClipGradientComponent::Backprop(const ChunkInfo &in_info,
                             const ChunkInfo &out_info,
                             const CuMatrixBase<BaseFloat> &in_value,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             Component *to_update_in, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrix<BaseFloat> *in_deriv) const {
  // the following statement will do nothing if in_deriv and out_deriv have same
  // memory.
  in_deriv->CopyFromMat(out_deriv);

  ClipGradientComponent *to_update =
      dynamic_cast<ClipGradientComponent*>(to_update_in);

  if (clipping_threshold_ > 0) {
    if (norm_based_clipping_) {
      // each row in the derivative matrix, which corresponds to one sample in
      // the mini-batch, is scaled to have a max-norm of clipping_threshold_
      CuVector<BaseFloat> clipping_scales(in_deriv->NumRows());
      clipping_scales.AddDiagMat2(pow(clipping_threshold_, -2), *in_deriv,
                                  kNoTrans, 0.0);
     // now clipping_scales contains the squared (norm of each row divided by
     //  clipping_threshold)
      int32 num_not_scaled = clipping_scales.ApplyFloor(1.0);
     // now clipping_scales contains min(1,
     //    squared-(norm/clipping_threshold))
      if (num_not_scaled != clipping_scales.Dim()) {
        clipping_scales.ApplyPow(-0.5);
        // now clipping_scales contains max(1,
        //       clipping_threshold/vector_norm)
        in_deriv->MulRowsVec(clipping_scales);
        if (to_update != NULL)
          to_update->num_clipped_ += (clipping_scales.Dim() - num_not_scaled);
       }
      if (to_update != NULL)
        to_update->count_ += clipping_scales.Dim();
    } else {
      // each element of the derivative matrix, is clipped to be below the
      // clipping_threshold_
      in_deriv->ApplyCeiling(clipping_threshold_);
      in_deriv->ApplyFloor(-1 * clipping_threshold_);
    }

    if (to_update != NULL) {
      to_update->num_backpropped_ += 1;
      RepairGradients(in_value, in_deriv, to_update);
    }
  }
}

// This function will add a self-repair term to in-deriv, attempting to shrink
// the maginitude of the input towards self_repair_target_.
// This term is proportional to [-(input vector - self_repair_target_)].
// The avarage magnitude of this term is equal to
// [self_repair_scale_ * clipped_proportion * average norm of input derivative].
// We use norm of input derivative when computing the magnitude so that it is
// comparable to the magnitude of input derivative, especially when the gradient
// explosion is actually happening.
void ClipGradientComponent::RepairGradients(
    const CuMatrixBase<BaseFloat> &in_value,
    CuMatrixBase<BaseFloat> *in_deriv, ClipGradientComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;
  if (self_repair_clipped_proportion_threshold_ >= 1.0 ||
      self_repair_scale_ == 0.0 || count_ == 0 ||
      RandUniform() > repair_probability)
    return;

  KALDI_ASSERT(self_repair_target_ >= 0.0 && self_repair_scale_ > 0.0);

  BaseFloat clipped_proportion =
    (count_ > 0 ? static_cast<BaseFloat>(num_clipped_) / count_ : 0);
  // in-deriv would be modified only when clipped_proportion exceeds the
  // threshold
  if (clipped_proportion <= self_repair_clipped_proportion_threshold_)
    return;

  to_update->num_self_repaired_ += 1;
  if (to_update->num_self_repaired_ == 1)
    KALDI_LOG << "ClipGradientComponent's "
              << "self-repair was activated as the first time at the "
              << to_update->num_backpropped_
              << "-th call of Backprop() in this training job.";

  // sign_mat = sign(in_value), i.e.,
  // An element in sign_mat is 1 if its corresponding element in in_value > 0,
  // or -1 otherwise
  CuMatrix<BaseFloat> sign_mat(in_value);
  sign_mat.ApplyHeaviside();
  sign_mat.Scale(2.0);
  sign_mat.Add(-1.0);

  // repair_mat =
  // floor(abs(in_value) - self_repair_target_, 0) .* sign(in_value)
  CuMatrix<BaseFloat> repair_mat(in_value);
  repair_mat.ApplyPowAbs(1.0);
  repair_mat.Add(-self_repair_target_);
  repair_mat.ApplyFloor(0.0);
  repair_mat.MulElements(sign_mat);

  // magnitude =
  // self_repair_scale_ * clipped_proportion * average norm of in-deriv
  CuVector<BaseFloat> in_deriv_norm_vec(in_deriv->NumRows());
  in_deriv_norm_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 0.0);
  in_deriv_norm_vec.ApplyPow(0.5);
  double in_deriv_norm_sum = in_deriv_norm_vec.Sum();
  BaseFloat magnitude = self_repair_scale_ * clipped_proportion *
                        (in_deriv_norm_sum / in_deriv_norm_vec.Dim());
 
  CuVector<BaseFloat> repair_mat_norm_vec(repair_mat.NumRows());
  repair_mat_norm_vec.AddDiagMat2(1.0, repair_mat, kNoTrans, 0.0);
  repair_mat_norm_vec.ApplyPow(0.5);
  double repair_mat_norm_sum = repair_mat_norm_vec.Sum();
  double scale = 0.0;
  if (repair_mat_norm_sum != 0.0)
    scale = magnitude / (repair_mat_norm_sum / repair_mat_norm_vec.Dim());
  // repair_mat is scaled so that on average the rows have the norm
  // (magnitude / repair_probability). This will give higher magnitude of
  // self-repair to input vectors that have larger absolute value, which tend to
  // be those that are diverging.
  in_deriv->AddMat(-scale / repair_probability, repair_mat);
  CuVector<BaseFloat> in_deriv_repaired_norm_vec(in_deriv->NumRows());
  in_deriv_repaired_norm_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 0.0);
  in_deriv_repaired_norm_vec.ApplyPow(0.5);
  // scale in_deriv to have the same norm as that before adding the self-repair
  // term, in order to avoid increase of the norm caused by self-repair,
  // which may incur more clip of gradient and thus more self-repair
  double in_deriv_repaired_norm_sum = in_deriv_repaired_norm_vec.Sum();
  if (in_deriv_repaired_norm_sum != 0.0)
    in_deriv->Scale(in_deriv_norm_sum / in_deriv_repaired_norm_sum);
}

void ClipGradientComponent::ZeroStats()  {
  count_ = 0.0;
  num_clipped_ = 0.0;
  num_self_repaired_ = 0;
  num_backpropped_ = 0;
}

void ClipGradientComponent::Scale(BaseFloat scale) {
  count_ *= scale;
  num_clipped_ *= scale;
}

void ClipGradientComponent::Add(BaseFloat alpha, const Component &other_in) {
  const ClipGradientComponent *other =
      dynamic_cast<const ClipGradientComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  count_ += alpha * other->count_;
  num_clipped_ += alpha * other->num_clipped_;
}

}  // namespace nnet2
}  // namespace kaldi

#endif  // HAVE_CUDA && HAVE_CUDNN
