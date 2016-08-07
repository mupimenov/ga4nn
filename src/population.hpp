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
#ifndef __POPULATION_HPP
#define __POPULATION_HPP
#include <cstdlib>

#include <memory>
#include <map>

namespace ga4nn {
template<class Genotype>
class population {
public:
  typedef Genotype genotype;
  typedef typename std::shared_ptr<population<genotype> > ptr;
  virtual ~population() {}

  virtual void insert(typename genotype::ptr g) = 0;
  virtual typename genotype::ptr take_beauty() = 0;
  virtual typename genotype::ptr take_monster() = 0;
  virtual size_t count() const = 0;
  virtual void clear() = 0;
};

template<class Genotype>
class rb_population : public population<Genotype> {
public:
  typedef Genotype genotype;
  typedef typename std::shared_ptr<rb_population> ptr;
  virtual ~rb_population() {}

  virtual void insert(typename genotype::ptr g) {
    d_genotype.insert(std::pair<double, typename genotype::ptr>(g->fitness(), g));
  }
  virtual typename genotype::ptr take_beauty() {
    if (d_genotype.begin() == d_genotype.end())
      return typename genotype::ptr();
    typename std::map<double, typename genotype::ptr>::iterator it =
      d_genotype.begin();
    typename genotype::ptr g = it->second;
    d_genotype.erase(d_genotype.begin());
    return g;
  }
  virtual typename Genotype::ptr take_monster() {
    if (d_genotype.begin() == d_genotype.end())
      return typename genotype::ptr();
    typename std::map<double, typename genotype::ptr>::iterator it =
      d_genotype.end();
    --it;
    typename genotype::ptr g = it->second;
    d_genotype.erase(it);
    return g;
  }
  virtual size_t count() const {
    return d_genotype.size();
  }
  virtual void clear() {
    d_genotype.clear();
  }
  virtual typename Genotype::ptr take_middle() {
    if (d_genotype.begin() == d_genotype.end())
      return typename genotype::ptr();
    typename std::map<double, typename genotype::ptr>::iterator it =
      d_genotype.begin();
    for (size_t i = 0; i < d_genotype.size() / 2; ++i)
      ++it;
    typename genotype::ptr g = it->second;
    d_genotype.erase(it);
    return g;
  }
private:
  std::multimap<double, typename genotype::ptr> d_genotype;
};
}

#endif
