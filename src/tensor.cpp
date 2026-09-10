#include "hustler/tensor.hpp"

#include <stdexcept>
#include <utility>

Tensor::Tensor(std::vector<float> values, std::vector<std::size_t> shape)
    : values_(std::move(values)), shape_(std::move(shape)),
      strides_(shape_.size()) {
  std::size_t count = 1;
  for (std::size_t i = shape_.size(); i-- > 0;) {
    strides_[i] = count;
    count *= shape_[i];
  }

  if (count != values_.size()) {
    throw std::invalid_argument("value count does not match shape");
  }
}

std::size_t Tensor::rank() const noexcept { return shape_.size(); }

std::size_t Tensor::size() const noexcept { return values_.size(); }

const std::vector<std::size_t> &Tensor::shape() const noexcept {
  return shape_;
}

float Tensor::at(const std::vector<std::size_t> &indices) const {
  if (indices.size() != rank()) {
    throw std::out_of_range("index count does not match tensor rank");
  }

  std::size_t offset = 0;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("tensor index out of range");
    }
    offset += indices[i] * strides_[i];
  }

  return values_[offset];
}
