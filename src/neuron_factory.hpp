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
#ifndef __NEURON_FACTORY_HPP
#define __NEURON_FACTORY_HPP
#include <memory>

#include "neuron.hpp"

namespace ga4nn {
class input_neuron_factory {
public:
  input_neuron_factory();
  neuron::ptr create_neuron();
};

class sigmoid_neuron_factory {
public:
  sigmoid_neuron_factory();
  neuron::ptr create_neuron();
};

class linear_neuron_factory {
public:
  linear_neuron_factory();
  neuron::ptr create_neuron();
};

typedef linear_neuron_factory output_neuron_factory;

class feedback_neuron_factory {
public:
  feedback_neuron_factory();
  neuron::ptr create_neuron();
private:
  size_t m_hystory_len;
};
}

#endif
