#include "gmock/gmock.h"
#include "neuron.hpp"

class mock_neuron : public ga4nn::neuron {
public:
  MOCK_CONST_METHOD0(activated, bool());
  MOCK_METHOD0(compute, void());
  MOCK_CONST_METHOD0(get_output, double());
  MOCK_METHOD1(set_output, void(double));
};
