#pragma once

#include <cstddef>
#include <vector>

class Tensor {
public:
  Tensor(std::vector<float> values, std::vector<std::size_t> shape);

  std::size_t rank() const noexcept;
  std::size_t size() const noexcept;
  const std::vector<std::size_t> &shape() const noexcept;
  float at(const std::vector<std::size_t> &indices) const;

private:
  std::vector<float> values_;
  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
};