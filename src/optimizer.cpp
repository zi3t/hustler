#include "hustler/optimizer.hpp"

#include <stdexcept>

SGD::SGD(float learning_rate) : learning_rate_(learning_rate) {
  if (learning_rate_ <= 0.0f) {
    throw std::invalid_argument("learning rate must be positive");
  }
}

void SGD::step(DenseLayer &layer) const { layer.apply_gradients(learning_rate_); }
