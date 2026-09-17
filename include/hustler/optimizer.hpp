#pragma once

#include "hustler/dense_layer.hpp"

class SGD {
public:
  explicit SGD(float learning_rate);

  void step(DenseLayer &layer) const;

private:
  float learning_rate_;
};
