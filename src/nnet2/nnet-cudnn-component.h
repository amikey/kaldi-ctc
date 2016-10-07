// nnet2/nnet-cudnn-component.h

// Copyright 2016  LingoChamp Feiteng


#ifndef KALDI_NNET2_NNET_CUDNN_SIMPLE_COMPONENT_H_
#define KALDI_NNET2_NNET_CUDNN_SIMPLE_COMPONENT_H_

#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
#include <vector>

#include "nnet2/nnet-component.h"
#include "cudamatrix/cudnn-utils.h"

namespace kaldi {
namespace nnet2 {

class CuDNNRecurrentComponent: public UpdatableComponent {
 public:
  virtual int32 InputDim() const {
    return input_dim_;
  }
  virtual int32 OutputDim() const {
    return hidden_dim_ * (bidirectional_ ? 2 : 1);
  };

  virtual std::string Info() const;
  virtual void InitFromString(std::string args);

  CuDNNRecurrentComponent();  // use Init to really initialize.

  virtual std::string Type() const {
    return "CuDNNRecurrentComponent";
  }

  virtual bool BackpropNeedsInput() const {
    return true;
  }

  virtual bool BackpropNeedsOutput() const {
    return true;
  }

  virtual int32 GetParameterDim() const {
    return filter_params_.Dim();
  }

  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &,  // in_value
                        const CuMatrixBase<BaseFloat> &,  // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrix<BaseFloat> *in_deriv) const;

  void InitMiniBatch(int32 mini_batch, int32 seq_length = 0) const;
  void Init(int32 mini_batch, int32 seq_length = 0) const;
  void DestroyDescriptors() const;

  // constructor using another component
  explicit CuDNNRecurrentComponent(const CuDNNRecurrentComponent &component);

  ~CuDNNRecurrentComponent();

  virtual Component *Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const UpdatableComponent &other);
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void PerturbParams(BaseFloat stddev);
  void GetFilterBias(int32 layer, int linLayerID, Vector<BaseFloat> &filter, Vector<BaseFloat> &bias);

 private:
  void Update(const CuVector<BaseFloat> &filter_params_grad);

  void SetBufferZero() const;

  int32 input_dim_;
  int32 hidden_dim_;
  int32 num_layers_;

  mutable int32 max_seq_length_;

  // cudnnRNNInputMode_t input_mode_;
  bool bidirectional_;
  bool is_gradient_;

  mutable cudnnRNNDescriptor_t rnn_desc_;
  int32 rnn_mode_;

  mutable int32 mini_batch_;
  mutable BaseFloat param_stddev_;
  mutable BaseFloat bias_stddev_;

  mutable CuVector<BaseFloat> work_space_;
  mutable CuVector<BaseFloat> reserve_space_;
  mutable uint32 work_space_size_;
  mutable uint32 reserve_space_size_;

  mutable CuVector<BaseFloat> hx_;
  mutable CuVector<BaseFloat> cx_;
  mutable CuVector<BaseFloat> dhx_;
  mutable CuVector<BaseFloat> dcx_;

  mutable CuVector<BaseFloat> hy_;
  mutable CuVector<BaseFloat> cy_;
  mutable CuVector<BaseFloat> dhy_;
  mutable CuVector<BaseFloat> dcy_;

  // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
  mutable cudnnTensorDescriptor_t *x_desc_, *y_desc_, *dx_desc_, *dy_desc_;
  mutable cudnnTensorDescriptor_t hx_desc_, cx_desc_;
  mutable cudnnTensorDescriptor_t hy_desc_, cy_desc_;
  mutable cudnnTensorDescriptor_t dhx_desc_, dcx_desc_;
  mutable cudnnTensorDescriptor_t dhy_desc_, dcy_desc_;

  mutable cudnnDropoutDescriptor_t dropout_desc_;
  mutable CuVector<BaseFloat> dropout_states_;

  mutable cudnnFilterDescriptor_t w_desc_, dw_desc_;
  mutable CuVector<BaseFloat> filter_params_;
  mutable CuVector<BaseFloat> filter_params_grad_;
  BaseFloat clip_gradient_;

  const CuDNNRecurrentComponent &operator = (const CuDNNRecurrentComponent
      &other); // Disallow.
};


// modified from nnet3
// ClipGradientComponent just duplicates its input, but clips gradients
// during backpropagation if they cross a predetermined threshold.
// This component will be used to prevent gradient explosion problem in
// recurrent neural networks
class ClipGradientComponent: public Component {
 public:
  ClipGradientComponent(int32 dim, BaseFloat clipping_threshold,
                        bool norm_based_clipping,
                        BaseFloat self_repair_clipped_proportion_threshold,
                        BaseFloat self_repair_target,
                        BaseFloat self_repair_scale,
                        int32 num_clipped,
                        int32 count,
                        int32 num_self_repaired,
                        int32 num_backpropped) {
    Init(dim, clipping_threshold, norm_based_clipping,
         self_repair_clipped_proportion_threshold,
         self_repair_target,
         self_repair_scale,
         num_clipped, count,
         num_self_repaired, num_backpropped);}

  ClipGradientComponent(): dim_(0), clipping_threshold_(-1),
    norm_based_clipping_(false),
    self_repair_clipped_proportion_threshold_(1.0),
    self_repair_target_(0.0),
    self_repair_scale_(0.0),
    num_clipped_(0), count_(0),
    num_self_repaired_(0), num_backpropped_(0) { }

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void InitFromString(std::string args);
  void Init(int32 dim, BaseFloat clipping_threshold, bool norm_based_clipping,
            BaseFloat self_repair_clipped_proportion_threshold,
            BaseFloat self_repair_target,
            BaseFloat self_repair_scale,
            int32 num_clipped, int32 count,
            int32 num_self_repaired, int32 num_backpropped);

  virtual std::string Type() const { return "ClipGradientComponent"; }

  virtual void ZeroStats();

  virtual Component* Copy() const {
    return new ClipGradientComponent(dim_,
                                     clipping_threshold_,
                                     norm_based_clipping_,
                                     self_repair_clipped_proportion_threshold_,
                                     self_repair_target_,
                                     self_repair_scale_,
                                     num_clipped_,
                                     count_,
                                     num_self_repaired_,
                                     num_backpropped_);}
  using Component::Propagate; // to avoid name hiding
  virtual void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrix<BaseFloat> *in_deriv) const;
  virtual bool BackpropNeedsOutput() const { return false; }
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.
  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
  virtual ~ClipGradientComponent() {
    if (num_self_repaired_ > 0)
      KALDI_LOG << "ClipGradientComponent's"
                << " self-repair was activated " << num_self_repaired_
                << " time(s) out of " << num_backpropped_
                << " times of calling Backprop() in this training job.";
  }
 private:
  int32 dim_;  // input/output dimension
  BaseFloat clipping_threshold_;  // threshold to be used for clipping
                                  // could correspond to max-row-norm (if
                                  // norm_based_clipping_ == true) or
                                  // max-absolute-value (otherwise)
  bool norm_based_clipping_;  // if true the max-row-norm will be clipped
                              // else element-wise absolute value clipping is
                              // done

  // some configuration values relating to self-repairing.
  BaseFloat self_repair_clipped_proportion_threshold_; // the threshold of
                                                       // clipped-proportion
                                                       // for self-repair to be
                                                       // activated
  BaseFloat self_repair_target_; // the target value towards which self-repair
                                 // is trying to set for in-deriv
  BaseFloat self_repair_scale_;  // constant scaling the self-repair vector
  
  // this function is called from Backprop code, and only does something if the
  // self-repair-scale config value is set and the current clipped proportion
  // exceeds the threshold. What it does is to add a term to in-deriv that
  // forces the input to the ClipGradientComponent to be close to some small
  // value (e.g., 0.0 or 0.5, depending on what the input is, e.g.,
  // Sigmoid or Tanh or Affine). The hope is that if the input is forced to be
  // small, the parameters on the path will also tend to be small, which may
  // help tamp down the divergence caused by gradient explosion.
  void RepairGradients(const CuMatrixBase<BaseFloat> &in_value,
                       CuMatrixBase<BaseFloat> *in_deriv,
                       ClipGradientComponent *to_update) const;

  ClipGradientComponent &operator =
      (const ClipGradientComponent &other); // Disallow.

 protected:
  // variables to store stats
  // An element corresponds to rows of derivative matrix, when
  // norm_based_clipping_ is true,
  // else it corresponds to each element of the derivative matrix
  // Note: no stats are stored when norm_based_clipping_ is false
  int32 num_clipped_;  // number of elements which were clipped
  int32 count_;  // number of elements which were processed
  int32 num_self_repaired_; // number of times self-repair is activated
  int32 num_backpropped_; //number of times backprop is called

};

}  // namespace nnet2
}  // namespace kaldi

#endif  // HAVE_CUDA && HAVE_CUDNN
#endif  // KALDI_NNET2_NNET_CUDNN_SIMPLE_COMPONENT_H_
