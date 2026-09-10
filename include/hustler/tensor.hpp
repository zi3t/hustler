#pragma once

#include <cstddef>
#include <vector>

class Tensor {
private:
  std::vector<float> values;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> computed_strides;

public:
  Tensor(std::vector<float> values, std::vector<std::size_t> shape);
};