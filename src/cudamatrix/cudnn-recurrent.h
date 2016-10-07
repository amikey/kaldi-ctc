// cudamatrix/cudnn-recurrent.h

// Copyright 2016  LingoChamp Feiteng


#ifndef KALDI_CUDAMATRIX_CUDNN_RECURRENT_H_
#define KALDI_CUDAMATRIX_CUDNN_RECURRENT_H_

#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
#include "cudamatrix/cu-matrix.h"
#include "cudnn.h"


namespace kaldi {
namespace cudnn {

  void RecurrentForwardInference(cudnnHandle_t handle,
                                  const cudnnRNNDescriptor_t &rnnDesc,
                                  const int seqLength,
                                  const cudnnTensorDescriptor_t * xDesc,
                                  const void * x,
                                  const cudnnTensorDescriptor_t &hxDesc,
                                  const void * hx,
                                  const cudnnTensorDescriptor_t &cxDesc,
                                  const void * cx,
                                  const cudnnFilterDescriptor_t &wDesc,
                                  const void * w,
                                  const cudnnTensorDescriptor_t *yDesc,
                                  void * y,
                                  const cudnnTensorDescriptor_t &hyDesc,
                                  void * hy,
                                  const cudnnTensorDescriptor_t &cyDesc,
                                  void * cy,
                                  void * workspace,
                                  size_t workSpaceSizeInBytes);

  void RecurrentForwardTraining(cudnnHandle_t handle,
                                 const cudnnRNNDescriptor_t &rnnDesc,
                                 const int seqLength,
                                 const cudnnTensorDescriptor_t *xDesc,
                                 const void * x,
                                 const cudnnTensorDescriptor_t &hxDesc,
                                 const void * hx,
                                 const cudnnTensorDescriptor_t &cxDesc,
                                 const void * cx,
                                 const cudnnFilterDescriptor_t &wDesc,
                                 const void * w,
                                 const cudnnTensorDescriptor_t *yDesc,
                                 void * y,
                                 const cudnnTensorDescriptor_t &hyDesc,
                                 void * hy,
                                 const cudnnTensorDescriptor_t &cyDesc,
                                 void * cy,
                                 void * workspace,
                                 size_t workSpaceSizeInBytes,
                                 void * reserveSpace,
                                 size_t reserveSpaceSizeInBytes);

  void RecurrentBackwardData(cudnnHandle_t handle,
                              const cudnnRNNDescriptor_t &rnnDesc,
                              const int seqLength,
                              const cudnnTensorDescriptor_t * yDesc,
                              const void * y,
                              const cudnnTensorDescriptor_t * dyDesc,
                              const void * dy,
                              const cudnnTensorDescriptor_t &dhyDesc,
                              const void * dhy,
                              const cudnnTensorDescriptor_t &dcyDesc,
                              const void * dcy,
                              const cudnnFilterDescriptor_t &wDesc,
                              const void * w,
                              const cudnnTensorDescriptor_t &hxDesc,
                              const void * hx,
                              const cudnnTensorDescriptor_t &cxDesc,
                              const void * cx,
                              const cudnnTensorDescriptor_t * dxDesc,
                              void * dx,
                              const cudnnTensorDescriptor_t &dhxDesc,
                              void * dhx,
                              const cudnnTensorDescriptor_t &dcxDesc,
                              void * dcx,
                              void * workspace,
                              size_t workSpaceSizeInBytes,
                              const void * reserveSpace,
                              size_t reserveSpaceSizeInBytes);


  void RecurrentBackwardWeights(cudnnHandle_t handle,
                                 const cudnnRNNDescriptor_t &rnnDesc,
                                 const int seqLength,
                                 const cudnnTensorDescriptor_t * xDesc,
                                 const void * x,
                                 const cudnnTensorDescriptor_t &hxDesc,
                                 const void * hx,
                                 const cudnnTensorDescriptor_t * yDesc,
                                 const void * y,
                                 const void * workspace,
                                 size_t workSpaceSizeInBytes,
                                 const cudnnFilterDescriptor_t &dwDesc,
                                 void * dw,
                                 const void * reserveSpace,
                                 size_t reserveSpaceSizeInBytes);

  void GetRecurrentWorkspaceSize(cudnnHandle_t handle,
                                const cudnnRNNDescriptor_t &rnnDesc,
                                const int seqLength,
                                const cudnnTensorDescriptor_t *xDesc,
                                size_t *sizeInBytes);

  void GetRecurrentTrainingReserveSize(cudnnHandle_t handle,
                                      const cudnnRNNDescriptor_t &rnnDesc,
                                      const int seqLength,
                                      const cudnnTensorDescriptor_t *xDesc,
                                      size_t *sizeInBytes);

  void GetRecurrentParamsSize(cudnnHandle_t handle,
                             const cudnnRNNDescriptor_t &rnnDesc,
                             const cudnnTensorDescriptor_t &xDesc,
                             size_t *sizeInBytes,
                             cudnnDataType_t dataType);

  void GetRecurrentLinLayerMatrixParams(cudnnHandle_t handle,
                             const cudnnRNNDescriptor_t &rnnDesc,
                             const int layer,
                             const cudnnTensorDescriptor_t &xDesc,
                             const cudnnFilterDescriptor_t &wDesc,
                             const void * w,
                             const int linLayerID,
                             cudnnFilterDescriptor_t &linLayerMatDesc,
                             void ** linLayerMat);

  void GetRecurrentLinLayerBiasParams(cudnnHandle_t handle,
                             const cudnnRNNDescriptor_t &rnnDesc,
                             const int layer,
                             const cudnnTensorDescriptor_t &xDesc,
                             const cudnnFilterDescriptor_t &wDesc,
                             const void * w,
                             const int linLayerID,
                             cudnnFilterDescriptor_t &linLayerBiasDesc,
                             void ** linLayerBias);

}  // namespace cudnn
}  // namespace kaldi
#endif  // HAVE_CUDA && HAVE_CUDNN
#endif
