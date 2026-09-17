#pragma once

#include "hustler/tensor.hpp"

#include <optional>

class DenseLayer {
public:
  DenseLayer(Tensor weights, Tensor bias);

  Tensor forward(const Tensor &input);
  Tensor backward(const Tensor &gradient_output);
  void apply_gradients(float learning_rate);

  const Tensor &weights() const noexcept;
  const Tensor &bias() const noexcept;

private:
  Tensor weights_;
  Tensor bias_;
  std::optional<Tensor> cached_input_;
  std::optional<Tensor> gradient_weights_;
  std::optional<Tensor> gradient_bias_;
};
