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
#ifndef __LAYER_HPP
#define __LAYER_HPP
#include <memory>
#include <vector>

#include "neuron.hpp"

namespace ga4nn {
class layer {
public:
  typedef typename std::shared_ptr<layer> ptr;
  struct connection {
    typedef typename std::shared_ptr<connection> ptr;
    neuron::link::ptr link_back;
    neuron::ptr neuron_front;

    connection(const neuron::link::ptr &l_back, const neuron::ptr &n_front)
        : link_back(l_back), neuron_front(n_front) {}
  };

  ~layer();

  template <class Factory> void add_neurons(Factory factory, size_t count) {
    for (size_t i = 0; i < count; ++i)
      m_neuron.push_back(factory.create_neuron());
  }

  size_t neuron_count() const;
  neuron::ptr get_neuron(size_t index) const;

  template <class Connector>
  void connect_back(const layer::ptr &l_back, Connector connector) {
    for (size_t i = 0; i < m_neuron.size(); ++i) {
      const neuron::ptr &n_front = m_neuron[i];
      for (size_t j = 0; j < l_back->neuron_count(); ++j) {
        neuron::ptr n_back = l_back->get_neuron(j);
        if (connector.valid_connection(j, i))
          m_connection.push_back(connection::ptr(
              new connection(n_front->create_link(n_back,
                              connector.weight(),
                              connector.constant()), n_front)));
      }
    }
  }

  size_t connection_count() const;
  connection::ptr get_connection(size_t index) const;

  void set_weights(std::vector<double>::const_iterator &first,
                   const std::vector<double>::const_iterator &last);
  std::vector<double> get_weights() const;

  void compute();

  void set_outputs(std::vector<double>::const_iterator &first,
                   const std::vector<double>::const_iterator &last);
  std::vector<double> get_outputs() const;

protected:
  std::vector<neuron::ptr> m_neuron;
  std::vector<connection::ptr> m_connection;
};
}

#endif
