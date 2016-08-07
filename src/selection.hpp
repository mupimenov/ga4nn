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
#ifndef __SELECTION_HPP
#define __SELECTION_HPP
#include <memory>
#include <vector>

namespace ga4nn {
template<class Population>
class selection {
public:
  typedef Population population;
  typedef typename population::genotype genotype;
  typedef typename std::shared_ptr<selection<population> > ptr;

  virtual ~selection() {}

  virtual std::vector<typename genotype::ptr> get_parents(
    typename population::ptr p) = 0;
};

template<class Population>
class bm_selection : selection<Population> {
public:
  typedef Population population;
  typedef typename population::genotype genotype;
  typedef typename std::shared_ptr<bm_selection> ptr;

  virtual ~bm_selection() {}

  virtual std::vector<typename genotype::ptr> get_parents(
    typename population::ptr p) {
    std::vector<typename genotype::ptr> vec(2);
    vec[0] = p->take_beauty();
    vec[1] = p->take_monster();
    return vec;
  }
};

template<class Population>
class b_selection : selection<Population> {
public:
  typedef Population population;
  typedef typename population::genotype genotype;
  typedef typename std::shared_ptr<b_selection> ptr;

  virtual ~b_selection() {}

  virtual std::vector<typename genotype::ptr> get_parents(
    typename population::ptr p) {
    std::vector<typename genotype::ptr> vec(1);
    vec[0] = p->take_beauty();
    return vec;
  }
};

template<class Population>
class bb_selection : selection<Population> {
public:
  typedef Population population;
  typedef typename population::genotype genotype;
  typedef typename std::shared_ptr<bb_selection> ptr;

  virtual ~bb_selection() {}

  virtual std::vector<typename genotype::ptr> get_parents(
    typename population::ptr p) {
      std::vector<typename genotype::ptr> vec(2);
      vec[0] = p->take_beauty();
      vec[1] = p->take_beauty();
      return vec;
  }
};
}

#endif
