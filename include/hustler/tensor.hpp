#pragma once

#include <cstddef>
#include <functional>
#include <vector>

class Tensor {
public:
  Tensor(std::vector<float> values, std::vector<std::size_t> shape);

  std::size_t rank() const noexcept;
  std::size_t size() const noexcept;
  const std::vector<std::size_t> &shape() const noexcept;
  float at(const std::vector<std::size_t> &indices) const;

  Tensor operator+(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator*(float scalar) const;
  Tensor matmul(const Tensor &other) const;
  Tensor transpose() const;
  Tensor add_rowwise(const Tensor &row) const;
  Tensor sum_rows() const;
  Tensor map(const std::function<float(float)> &function) const;

private:
  std::vector<float> values_;
  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
};
