#include "hustler/tensor.hpp"

#include <algorithm>
#include <functional>
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

Tensor Tensor::operator+(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("tensor shapes do not match");
  }

  std::vector<float> result(size());
  std::transform(values_.begin(), values_.end(), other.values_.begin(),
                 result.begin(), std::plus<>{});
  return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator*(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("tensor shapes do not match");
  }

  std::vector<float> result(size());
  std::transform(values_.begin(), values_.end(), other.values_.begin(),
                 result.begin(), std::multiplies<>{});
  return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator*(float scalar) const {
  std::vector<float> result(size());
  std::transform(values_.begin(), values_.end(), result.begin(),
                 [scalar](float value) { return value * scalar; });
  return Tensor(std::move(result), shape_);
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (rank() != 2 || other.rank() != 2) {
    throw std::invalid_argument("matmul requires rank-2 tensors");
  }

  const std::size_t rows = shape_[0];
  const std::size_t shared = shape_[1];
  const std::size_t columns = other.shape_[1];
  if (shared != other.shape_[0]) {
    throw std::invalid_argument("matmul inner dimensions do not match");
  }

  std::vector<float> result(rows * columns, 0.0f);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < columns; ++col) {
      for (std::size_t k = 0; k < shared; ++k) {
        result[row * columns + col] +=
            values_[row * shared + k] * other.values_[k * columns + col];
      }
    }
  }

  return Tensor(std::move(result), {rows, columns});
}

Tensor Tensor::transpose() const {
  if (rank() != 2) {
    throw std::invalid_argument("transpose requires a rank-2 tensor");
  }

  const std::size_t rows = shape_[0];
  const std::size_t columns = shape_[1];
  std::vector<float> result(size());
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      result[column * rows + row] = values_[row * columns + column];
    }
  }

  return Tensor(std::move(result), {columns, rows});
}

Tensor Tensor::add_rowwise(const Tensor &row) const {
  if (rank() != 2 || row.rank() != 1 || shape_[1] != row.shape_[0]) {
    throw std::invalid_argument(
        "add_rowwise requires a matrix and a matching rank-1 row tensor");
  }

  std::vector<float> result(size());
  for (std::size_t index = 0; index < size(); ++index) {
    result[index] = values_[index] + row.values_[index % shape_[1]];
  }
  return Tensor(std::move(result), shape_);
}

Tensor Tensor::sum_rows() const {
  if (rank() != 2) {
    throw std::invalid_argument("sum_rows requires a rank-2 tensor");
  }

  const std::size_t rows = shape_[0];
  const std::size_t columns = shape_[1];
  std::vector<float> result(columns, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      result[column] += values_[row * columns + column];
    }
  }
  return Tensor(std::move(result), {columns});
}

Tensor Tensor::map(const std::function<float(float)> &function) const {
  std::vector<float> result(size());
  std::transform(values_.begin(), values_.end(), result.begin(), function);
  return Tensor(std::move(result), shape_);
}
