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
#include "layer.hpp"

namespace ga4nn {
layer::~layer() {}

size_t layer::neuron_count() const { return m_neuron.size(); }

neuron::ptr layer::get_neuron(size_t index) const {
  if (index >= m_neuron.size())
    return neuron::ptr();
  return m_neuron[index];
}

size_t layer::connection_count() const { return m_connection.size(); }

layer::connection::ptr layer::get_connection(size_t index) const {
  if (index >= m_connection.size())
    return connection::ptr();
  return m_connection[index];
}

void layer::set_weights(std::vector<double>::const_iterator &first,
                        const std::vector<double>::const_iterator &last) {
  for (size_t i = 0; (i < m_neuron.size()) && (first != last); ++i)
    m_neuron[i]->set_weights(first, last);
}

std::vector<double> layer::get_weights() const {
  std::vector<double> weights;
  for (size_t i = 0; i < m_neuron.size(); ++i) {
    std::vector<double> neuron_weights = m_neuron[i]->get_weights();
    std::copy(neuron_weights.begin(), neuron_weights.end(),
              std::back_inserter(weights));
  }
  return weights;
}

void layer::compute() {
  for (size_t i = 0; i < m_neuron.size(); ++i)
    m_neuron[i]->compute();
}

void layer::set_outputs(std::vector<double>::const_iterator &first,
                        const std::vector<double>::const_iterator &last) {
  for (size_t i = 0; (i < m_neuron.size()) && (first != last); ++i) {
    m_neuron[i]->set_output(*first);
    ++first;
  }
}

std::vector<double> layer::get_outputs() const {
  std::vector<double> outputs;
  for (size_t i = 0; i < m_neuron.size(); ++i)
    outputs.push_back(m_neuron[i]->get_output());
  return outputs;
}
}
