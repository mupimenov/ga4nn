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
#ifndef __NEURAL_NET_HPP
#define __NEURAL_NET_HPP
#include <memory>
#include <vector>

#include "layer.hpp"

namespace ga4nn {
class neural_net {
public:
  typedef typename std::shared_ptr<neural_net> ptr;
  neural_net();
  virtual ~neural_net();

  void add_layer(layer::ptr l);
  size_t layer_count() const;
  layer::ptr get_layer(size_t index) const;

  void set_weights(const std::vector<double> &weights);
  std::vector<double> get_weights() const;

  std::vector<double> compute(const std::vector<double> &input);

private:
  struct prv;
  std::shared_ptr<prv> d;
};
}

#endif
