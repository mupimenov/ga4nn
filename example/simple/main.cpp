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
#include <iostream>

#include "connector.hpp"
#include "neural_net.hpp"
#include "neuron_factory.hpp"

int main(int argc, char const *argv[]) {
  ga4nn::neural_net::ptr net(new ga4nn::neural_net);

  ga4nn::layer::ptr input_layer(new ga4nn::layer);
  input_layer->add_neurons(ga4nn::input_neuron_factory(), 2);

  ga4nn::layer::ptr hidden_layer(new ga4nn::layer);
  hidden_layer->add_neurons(ga4nn::sigmoid_neuron_factory(), 5);

  ga4nn::layer::ptr output_layer(new ga4nn::layer);
  output_layer->add_neurons(ga4nn::sigmoid_neuron_factory(), 1);

  hidden_layer->connect_back(input_layer, ga4nn::mm_connector());
  output_layer->connect_back(hidden_layer, ga4nn::mm_connector());

  net->add_layer(input_layer);
  net->add_layer(hidden_layer);
  net->add_layer(output_layer);

  return 0;
}
