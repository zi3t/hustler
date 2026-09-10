#include "hustler/tensor.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

int main() {
  Tensor tensor({1.9f, 2.8f, 3.7f, 4.6f, 5.5f, 6.4f}, {2, 3});
  const std::vector<std::size_t> expected_shape{2, 3};

  assert(tensor.rank() == 2);
  assert(tensor.size() == 6);
  assert(tensor.shape() == expected_shape);
  assert(tensor.at({0, 0}) == 1.9f);
  assert(tensor.at({1, 2}) == 6.4f);

  bool rejected = false;
  try {
    Tensor invalid({1.9f, 2.8f, 3.7f, 4.6f, 5.5f}, {2, 3});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    tensor.at({1, 3});
  } catch (const std::out_of_range &) {
    rejected = true;
  }
  assert(rejected);
}
