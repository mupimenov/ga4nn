#include <cmath>

#include "gtest/gtest.h"
#include "neuron.hpp"

#include "mock_neuron.hpp"

using namespace ga4nn;

TEST(input_neuron, default_ctor_and_default_values) {
  neuron::ptr n(new input_neuron);

  EXPECT_EQ(0, n->link_count());
  EXPECT_FALSE(n->get_link(0));
  EXPECT_TRUE(n->activated());
  EXPECT_DOUBLE_EQ(0.0, n->get_output());
  EXPECT_TRUE(n->get_weights().empty());
}

TEST(input_neuron, set_then_get_output) {
  neuron::ptr n(new input_neuron);

  const double v1 = 0.1234;
  n->set_output(v1);
  n->compute();

  EXPECT_DOUBLE_EQ(n->get_output(), v1);
  EXPECT_TRUE(n->activated());

  const double v2 = 0.987;
  n->set_output(v2);
  n->compute();

  EXPECT_DOUBLE_EQ(n->get_output(), v2);
  EXPECT_TRUE(n->activated());
}

TEST(sigmoid_neuron, default_ctor_and_default_values) {
  neuron::ptr n(new sigmoid_neuron);

  EXPECT_EQ(0, n->link_count());
  EXPECT_FALSE(n->get_link(0));
  EXPECT_FALSE(n->activated());
  EXPECT_DOUBLE_EQ(0.0, n->get_output());
  EXPECT_TRUE(n->get_weights().empty());
}

TEST(sigmoid_neuron, compute) {
  using ::testing::Return;

  neuron::ptr n(new sigmoid_neuron);
  mock_neuron *m = new mock_neuron;

  const double weight = 1.0;
  n->create_link(neuron::ptr(m), weight, false);

  EXPECT_EQ(n->link_count(), 1);

  const double input = 1.0;
  EXPECT_CALL(*m, get_output()).WillRepeatedly(Return(input));

  EXPECT_DOUBLE_EQ(n->get_link(0)->neuron_back->get_output(), input);

  n->compute();

  double sum = input * weight;
  double output = sum / (1 + std::abs(sum));
  EXPECT_DOUBLE_EQ(n->get_output(), output > 0.5? output: 0.0);

  EXPECT_EQ(n->activated(), output > 0.5);
}
