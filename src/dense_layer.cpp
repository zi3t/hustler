#include "hustler/dense_layer.hpp"

#include <stdexcept>
#include <utility>

DenseLayer::DenseLayer(Tensor weights, Tensor bias)
    : weights_(std::move(weights)), bias_(std::move(bias)) {
  if (weights_.rank() != 2 || bias_.rank() != 1 ||
      weights_.shape()[1] != bias_.shape()[0]) {
    throw std::invalid_argument(
        "dense layer requires rank-2 weights and matching rank-1 bias");
  }
}

Tensor DenseLayer::forward(const Tensor &input) {
  if (input.rank() != 2 || input.shape()[1] != weights_.shape()[0]) {
    throw std::invalid_argument("dense layer input shape does not match weights");
  }

  cached_input_ = input;
  return input.matmul(weights_).add_rowwise(bias_);
}

Tensor DenseLayer::backward(const Tensor &gradient_output) {
  if (!cached_input_) {
    throw std::logic_error("dense layer backward called before forward");
  }
  if (gradient_output.rank() != 2 ||
      gradient_output.shape()[0] != cached_input_->shape()[0] ||
      gradient_output.shape()[1] != weights_.shape()[1]) {
    throw std::invalid_argument("dense layer gradient output shape is invalid");
  }

  gradient_weights_ = cached_input_->transpose().matmul(gradient_output);
  gradient_bias_ = gradient_output.sum_rows();
  return gradient_output.matmul(weights_.transpose());
}

void DenseLayer::apply_gradients(float learning_rate) {
  if (learning_rate <= 0.0f) {
    throw std::invalid_argument("learning rate must be positive");
  }
  if (!gradient_weights_ || !gradient_bias_) {
    throw std::logic_error("dense layer gradients have not been calculated");
  }

  weights_ = weights_ - (*gradient_weights_ * learning_rate);
  bias_ = bias_ - (*gradient_bias_ * learning_rate);
  gradient_weights_.reset();
  gradient_bias_.reset();
}

const Tensor &DenseLayer::weights() const noexcept { return weights_; }

const Tensor &DenseLayer::bias() const noexcept { return bias_; }
