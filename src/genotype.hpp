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
#ifndef __GENOTYPE_HPP
#define __GENOTYPE_HPP
#include <memory>

namespace ga4nn {
template<class Data>
class genotype {
public:
  typedef Data data_type;
  typedef typename std::shared_ptr<genotype<data_type> > ptr;
  explicit genotype(const data_type &data) : m_data(data) {}
  virtual ~genotype() {}

  virtual double fitness() = 0;

  virtual bool operator<(genotype<data_type> &g) {
    return (fitness() < g.fitness());
  }

  data_type &get_data() { return m_data; }
  const data_type &get_data() const { return m_data; }

  void set_data(const data_type &data) { m_data = data; }

protected:
  data_type m_data;
};
}

#endif
