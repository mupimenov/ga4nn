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
#include "neuron.hpp"

#include <cmath>

namespace ga4nn {
neuron::~neuron() {}

neuron::link::ptr neuron::create_link(neuron::ptr neuron_back,
                                      double weight,
                                      bool constant) {
  link::ptr l(new link(neuron_back, weight, constant));
  d_link.push_back(l);
  return l;
}

size_t neuron::link_count() const { return d_link.size(); }

neuron::link::ptr neuron::get_link(size_t index) const {
  if (index < 0 || index >= d_link.size())
    return link::ptr();
  return d_link[index];
}

void neuron::set_weights(std::vector<double>::const_iterator &first,
                         const std::vector<double>::const_iterator &last) {
  for (size_t i = 0; (i < d_link.size()) && (first != last); ++i) {
    if (!d_link[i]->constant)
      d_link[i]->weight = *first;
    ++first;
  }
}

std::vector<double> neuron::get_weights() const {
  std::vector<double> weights;
  for (size_t i = 0; i < d_link.size(); ++i) {
    if (!d_link[i]->constant)
      weights.push_back(d_link[i]->weight);
  }
  return weights;
}

double neuron::forward() {
  double sum = 0.0;
  for (size_t i = 0; i < d_link.size(); ++i) {
    const link::ptr &link = d_link[i];
    sum += link->neuron_back->get_output() * link->weight;
  }
  return sum;
}

struct input_neuron::prv {
  double output;

  prv() : output(0.0) {}
};

input_neuron::input_neuron() : d(new prv) {}
input_neuron::~input_neuron() {}

bool input_neuron::activated() const { return true; }

void input_neuron::compute() {
  //
}

double input_neuron::get_output() const { return d->output; }

void input_neuron::set_output(double value) { d->output = value; }

struct sigmoid_neuron::prv {
  bool activated;
  double output;

  prv() : activated(false), output(0.0) {}
};

sigmoid_neuron::sigmoid_neuron() : d(new prv) {}
sigmoid_neuron::~sigmoid_neuron() {}

bool sigmoid_neuron::activated() const { return d->activated; }

void sigmoid_neuron::compute() {
  double sum = forward();
  d->output = sum / (1 + std::abs(sum));
#if 0
  d->activated = d->output > 0.5;
#else
  d->activated = true;
#endif
}

double sigmoid_neuron::get_output() const {
  if (!d->activated)
    return 0.0;
  return d->output;
}

void sigmoid_neuron::set_output(double value) { (void)(value); }

struct linear_neuron::prv {
  double output;

  prv() : output(0.0) {}
};

linear_neuron::linear_neuron() : d(new prv) {}
linear_neuron::~linear_neuron() {}

bool linear_neuron::activated() const { return true; }

void linear_neuron::compute() {
  d->output = forward();
}

double linear_neuron::get_output() const {
  return d->output;
}

void linear_neuron::set_output(double value) { (void)(value); }

struct feedback_neuron::prv {
  std::vector<double> history;
  double output;

  explicit prv(size_t hystory_len) : history(hystory_len + 1), output(0.0) {}
};

feedback_neuron::feedback_neuron(size_t history_len) :
  d(new prv(history_len)) {}
feedback_neuron::~feedback_neuron() {}

bool feedback_neuron::activated() const { return true; }

void feedback_neuron::compute() {
  double now = forward();
  if (d->history.size() < 2) {
    d->output = now;
  } else {
    size_t len = d->history.size() - 1;
    for (size_t i = 0; i < len; i++)
      d->history[i] = d->history[i + 1];
    d->history[len] = now;
    d->output = d->history[0];
  }
}

double feedback_neuron::get_output() const {
  return d->output;
}

void feedback_neuron::set_output(double value) { (void)(value); }
}
