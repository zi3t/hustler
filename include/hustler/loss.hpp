#pragma once

#include "hustler/tensor.hpp"

class MeanSquaredError {
public:
  float forward(const Tensor &prediction, const Tensor &target) const;
  Tensor backward(const Tensor &prediction, const Tensor &target) const;
};
