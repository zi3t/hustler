#include "hustler/loss.hpp"

#include <stdexcept>

float MeanSquaredError::forward(const Tensor &prediction,
                                const Tensor &target) const {
  const Tensor difference = prediction - target;
  float total = 0.0f;
  for (const float value : difference.values()) {
    total += value * value;
  }
  return total / static_cast<float>(difference.size());
}

Tensor MeanSquaredError::backward(const Tensor &prediction,
                                  const Tensor &target) const {
  if (prediction.size() == 0) {
    throw std::invalid_argument("mean squared error requires a non-empty tensor");
  }
  return (prediction - target) *
         (2.0f / static_cast<float>(prediction.size()));
}
