/*
COPYRIGHT (c) 2016 Mikhail Pimenov

MIT License

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "neuron_factory.hpp"

namespace ga4nn {
input_neuron_factory::input_neuron_factory() {}
neuron::ptr input_neuron_factory::create_neuron() {
  return neuron::ptr(new input_neuron);
}

sigmoid_neuron_factory::sigmoid_neuron_factory() {}
neuron::ptr sigmoid_neuron_factory::create_neuron() {
  return neuron::ptr(new sigmoid_neuron);
}

linear_neuron_factory::linear_neuron_factory() {}
neuron::ptr linear_neuron_factory::create_neuron() {
  return neuron::ptr(new linear_neuron);
}

feedback_neuron_factory::feedback_neuron_factory() : d_hystory_len(1) {}
neuron::ptr feedback_neuron_factory::create_neuron() {
  return neuron::ptr(new feedback_neuron(d_hystory_len++));
}
}
