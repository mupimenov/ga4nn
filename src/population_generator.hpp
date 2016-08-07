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
#ifndef __POPULATION_GENERATOR_HPP
#define __POPULATION_GENERATOR_HPP
#include <cstdlib>

#include <memory>

namespace ga4nn {
template<class Genotype>
class genotype_creator {
public:
  typedef Genotype genotype;
  typedef std::shared_ptr<genotype_creator<genotype> > ptr;
  virtual ~genotype_creator() {}
  virtual typename genotype::ptr make() = 0;
};

template<class Population, class GenotypeCreator>
void fill_population( typename Population::ptr p,
                      typename GenotypeCreator::ptr creator,
                      size_t count) {
  for (size_t i = 0; i < count; ++i)
    p->insert(creator->make());
}
}

#endif
