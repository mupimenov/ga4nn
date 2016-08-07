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
#include "neural_net.hpp"

namespace ga4nn {
struct neural_net::prv {
  std::vector<layer::ptr> layers;

  prv() {}
  ~prv() {}
};

neural_net::neural_net() : d(new prv) {}
neural_net::~neural_net() {}

void neural_net::add_layer(layer::ptr l) { d->layers.push_back(l); }

size_t neural_net::layer_count() const { return d->layers.size(); }

layer::ptr neural_net::get_layer(size_t index) const {
  if (index < 0 || index >= d->layers.size())
    return layer::ptr();
  return d->layers[index];
}

void neural_net::set_weights(const std::vector<double> &weights) {
  std::vector<double>::const_iterator first = weights.begin();
  std::vector<double>::const_iterator last = weights.end();
  for (size_t i = 0; i < d->layers.size() && first != last; ++i)
    d->layers[i]->set_weights(first, last);
}

std::vector<double> neural_net::get_weights() const {
  std::vector<double> weights;
  if (d->layers.size() < 2)
    return weights;
  for (size_t i = 0; i < d->layers.size(); ++i) {
    std::vector<double> layer_weights = d->layers[i]->get_weights();
    std::copy(layer_weights.begin(), layer_weights.end(),
              std::back_inserter(weights));
  }
  return weights;
}

std::vector<double> neural_net::compute(const std::vector<double> &input) {
  if (d->layers.size() < 2)
    return std::vector<double>();
  std::vector<double>::const_iterator first = input.begin();
  std::vector<double>::const_iterator last = input.end();
  d->layers[0]->set_outputs(first, last);
  size_t i = 0;
  for (; i < d->layers.size(); ++i)
    d->layers[i]->compute();
  return d->layers[i - 1]->get_outputs();
}
}
