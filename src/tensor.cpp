#include "hustler/tensor.hpp"
#include <stdexcept>
#include <utility>

Tensor::Tensor(std::vector<float> input_values,
               std::vector<std::size_t> input_shape)
    : values(std::move(input_values)), shape(std::move(input_shape)) {
  std::size_t count = 1;
  for (std::size_t dimension : shape) {
    count *= dimension;
  }
  if (count != values.size()) {
    throw std::invalid_argument("value count does not match shape");
  }
}
