#include "hustler/tensor.hpp"

#include <cassert>
#include <stdexcept>

int main() {
  Tensor valid({1.9f, 2.8f, 3.7f, 4.6f, 5.5f, 6.4f}, {2, 3});

  bool rejected = false;

  try {
    Tensor invalid({1.9f, 3.7f, 4.6f, 5.5f, 6.4f}, {2, 3});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }

  assert(rejected);
  return 0;
}