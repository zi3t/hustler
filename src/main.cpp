#include "hustler/dense_layer.hpp"
#include "hustler/loss.hpp"
#include "hustler/optimizer.hpp"
#include "hustler/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>
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

  DenseLayer layer(Tensor({0.0f}, {1, 1}), Tensor({0.0f}, {1}));
  MeanSquaredError mse;
  SGD optimizer(0.1f);
  const Tensor training_inputs({0.0f, 1.0f, 2.0f, 3.0f}, {4, 1});
  const Tensor training_targets({1.0f, 3.0f, 5.0f, 7.0f}, {4, 1});
  const float initial_loss =
      mse.forward(layer.forward(training_inputs), training_targets);

  for (int iteration = 0; iteration < 200; ++iteration) {
    const Tensor prediction = layer.forward(training_inputs);
    const Tensor gradient_output = mse.backward(prediction, training_targets);
    (void)layer.backward(gradient_output);
    optimizer.step(layer);
  }

  const Tensor trained_prediction = layer.forward(training_inputs);
  const float final_loss = mse.forward(trained_prediction, training_targets);
  assert(final_loss < 0.0001f);
  assert(std::fabs(layer.weights().at({0, 0}) - 2.0f) < 0.01f);
  assert(std::fabs(layer.bias().at({0}) - 1.0f) < 0.01f);
  std::cout << "Training complete: loss " << initial_loss << " -> "
            << final_loss << ", weight=" << layer.weights().at({0, 0})
            << ", bias=" << layer.bias().at({0}) << '\n';

  rejected = false;
  try {
    tensor.at({1, 3});
  } catch (const std::out_of_range &) {
    rejected = true;
  }
  assert(rejected);

  const Tensor left({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  const Tensor right({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  const Tensor sum = left + right;
  const Tensor product = left * right;

  assert(sum.at({0, 0}) == 6.0f);
  assert(sum.at({1, 1}) == 12.0f);
  assert(product.at({0, 0}) == 5.0f);
  assert(product.at({1, 1}) == 32.0f);
  assert(left.at({0, 0}) == 1.0f);
  assert(right.at({0, 0}) == 5.0f);

  const Tensor flat({1.0f, 2.0f, 3.0f, 4.0f}, {4});
  rejected = false;
  try {
    (void)(left + flat);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    (void)(left * flat);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  const Tensor matrix_a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const Tensor matrix_b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  const Tensor matrix_product = matrix_a.matmul(matrix_b);
  const std::vector<std::size_t> product_shape{2, 2};

  assert(matrix_product.shape() == product_shape);
  assert(matrix_product.at({0, 0}) == 58.0f);
  assert(matrix_product.at({0, 1}) == 64.0f);
  assert(matrix_product.at({1, 0}) == 139.0f);
  assert(matrix_product.at({1, 1}) == 154.0f);

  const Tensor transposed = matrix_a.transpose();
  assert(transposed.shape() == std::vector<std::size_t>({3, 2}));
  assert(transposed.at({0, 1}) == 4.0f);
  assert(transposed.at({2, 1}) == 6.0f);

  const Tensor bias({10.0f, 20.0f}, {2});
  const Tensor biased = matrix_product.add_rowwise(bias);
  assert(biased.at({0, 0}) == 68.0f);
  assert(biased.at({1, 1}) == 174.0f);

  const Tensor row_sum = matrix_product.sum_rows();
  assert(row_sum.shape() == std::vector<std::size_t>({2}));
  assert(row_sum.at({0}) == 197.0f);
  assert(row_sum.at({1}) == 218.0f);

  const Tensor doubled = matrix_product * 2.0f;
  assert(doubled.at({0, 0}) == 116.0f);

  const Tensor relu = Tensor({-1.0f, 0.0f, 2.0f}, {3}).map(
      [](float value) { return std::max(0.0f, value); });
  assert(relu.at({0}) == 0.0f);
  assert(relu.at({2}) == 2.0f);

  rejected = false;
  try {
    (void)matrix_a.matmul(left);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    (void)flat.transpose();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    (void)matrix_a.add_rowwise(left);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    (void)flat.matmul(flat);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}
