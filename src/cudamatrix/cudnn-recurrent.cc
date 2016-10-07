// cudamatrix/cudnn-recurrent.h

// Copyright 2016  LingoChamp Feiteng


#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
#include "cudamatrix/cudnn-recurrent.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {
namespace cudnn {

void RecurrentForwardInference(
  cudnnHandle_t handle, const cudnnRNNDescriptor_t &rnnDesc, const int seqLength,
  const cudnnTensorDescriptor_t * xDesc, const void * x,
  const cudnnTensorDescriptor_t &hxDesc, const void * hx,
  const cudnnTensorDescriptor_t &cxDesc, const void * cx,
  const cudnnFilterDescriptor_t &wDesc, const void * w,
  const cudnnTensorDescriptor_t *yDesc, void * y,
  const cudnnTensorDescriptor_t &hyDesc, void * hy,
  const cudnnTensorDescriptor_t &cyDesc, void * cy,
  void * workspace, size_t workSpaceSizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnRNNForwardInference(handle, rnnDesc, seqLength,
      xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy,
      workspace, workSpaceSizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void RecurrentForwardTraining(
  cudnnHandle_t handle, const cudnnRNNDescriptor_t &rnnDesc, const int seqLength,
  const cudnnTensorDescriptor_t *xDesc, const void * x,
  const cudnnTensorDescriptor_t &hxDesc, const void * hx,
  const cudnnTensorDescriptor_t &cxDesc, const void * cx,
  const cudnnFilterDescriptor_t &wDesc, const void * w,
  const cudnnTensorDescriptor_t *yDesc, void * y,
  const cudnnTensorDescriptor_t &hyDesc, void * hy,
  const cudnnTensorDescriptor_t &cyDesc, void * cy,
  void * workspace, size_t workSpaceSizeInBytes,
  void * reserveSpace, size_t reserveSpaceSizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnRNNForwardTraining(handle, rnnDesc, seqLength,
      xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy,
      workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void RecurrentBackwardData(
  cudnnHandle_t handle, const cudnnRNNDescriptor_t &rnnDesc, const int seqLength,
  const cudnnTensorDescriptor_t * yDesc, const void * y,
  const cudnnTensorDescriptor_t * dyDesc, const void * dy,
  const cudnnTensorDescriptor_t &dhyDesc, const void * dhy,
  const cudnnTensorDescriptor_t &dcyDesc, const void * dcy,
  const cudnnFilterDescriptor_t &wDesc, const void * w,
  const cudnnTensorDescriptor_t &hxDesc, const void * hx,
  const cudnnTensorDescriptor_t &cxDesc, const void * cx,
  const cudnnTensorDescriptor_t * dxDesc, void * dx,
  const cudnnTensorDescriptor_t &dhxDesc, void * dhx,
  const cudnnTensorDescriptor_t &dcxDesc, void * dcx,
  void * workspace, size_t workSpaceSizeInBytes,
  const void * reserveSpace, size_t reserveSpaceSizeInBytes ) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnRNNBackwardData(handle, rnnDesc, seqLength,
      yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w,
      hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx,
      workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void RecurrentBackwardWeights(
  cudnnHandle_t handle, const cudnnRNNDescriptor_t &rnnDesc, const int seqLength,
  const cudnnTensorDescriptor_t * xDesc, const void * x,
  const cudnnTensorDescriptor_t &hxDesc, const void * hx,
  const cudnnTensorDescriptor_t * yDesc, const void * y,
  const void * workspace, size_t workSpaceSizeInBytes,
  const cudnnFilterDescriptor_t &dwDesc, void * dw,
  const void * reserveSpace, size_t reserveSpaceSizeInBytes ) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnRNNBackwardWeights(handle, rnnDesc, seqLength,
      xDesc, x, hxDesc, hx, yDesc, y,
      workspace, workSpaceSizeInBytes,
      dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void GetRecurrentWorkspaceSize(cudnnHandle_t handle,
                              const cudnnRNNDescriptor_t &rnnDesc,
                              const int seqLength,
                              const cudnnTensorDescriptor_t *xDesc,
                              size_t *sizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength,
      xDesc, sizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void GetRecurrentTrainingReserveSize(cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t &rnnDesc,
                                    const int seqLength,
                                    const cudnnTensorDescriptor_t *xDesc,
                                    size_t *sizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength,
      xDesc, sizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void GetRecurrentParamsSize(cudnnHandle_t handle,
                           const cudnnRNNDescriptor_t &rnnDesc,
                           const cudnnTensorDescriptor_t &xDesc,
                           size_t *sizeInBytes,
                           cudnnDataType_t dataType) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetRNNParamsSize(handle, rnnDesc,
      xDesc, sizeInBytes, dataType));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void GetRecurrentLinLayerMatrixParams(cudnnHandle_t handle,
                           const cudnnRNNDescriptor_t &rnnDesc,
                           const int layer,
                           const cudnnTensorDescriptor_t &xDesc,
                           const cudnnFilterDescriptor_t &wDesc,
                           const void * w,
                           const int linLayerID,
                           cudnnFilterDescriptor_t &linLayerMatDesc,
                           void ** linLayerMat) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, layer,
      xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void GetRecurrentLinLayerBiasParams(cudnnHandle_t handle,
                           const cudnnRNNDescriptor_t &rnnDesc,
                           const int layer,
                           const cudnnTensorDescriptor_t &xDesc,
                           const cudnnFilterDescriptor_t &wDesc,
                           const void * w,
                           const int linLayerID,
                           cudnnFilterDescriptor_t &linLayerBiasDesc,
                           void ** linLayerBias) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, layer,
      xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

}  // namespace cudnn
}  // namespace kaldi

#endif  // HAVE_CUDA && HAVE_CUDNN
