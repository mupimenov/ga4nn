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
#ifndef __GENETIC_HPP
#define __GENETIC_HPP
#include <cstdlib>

#include <vector>

#include "genotype.hpp"
#include "population.hpp"
#include "population_generator.hpp"
#include "selection.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include "stop_function.hpp"

namespace ga4nn {
  template< class Population,
            class Selection,
            class Crossover,
            class Mutation,
            class StopFunction>
  typename Population::ptr evolve(typename Population::ptr initial_population,
                    typename Selection::ptr selection,
                    typename Crossover::ptr crossover,
                    typename Mutation::ptr mutation,
                    typename StopFunction::ptr stop) {
    typename Population::ptr child_population(new Population(*initial_population));
    while (!stop->done(child_population)) {
      typename Population::ptr parent_population(new Population(*child_population));
      child_population->clear();
      while (parent_population->count() > 0) {
        std::vector<typename Population::genotype::ptr> parents =
          selection->get_parents(parent_population);
        std::vector<typename Population::genotype::ptr> children =
          crossover->cross(parents);
        for (size_t i = 0; i < children.size(); i++) {
          child_population->insert(mutation->mutate(children[i]));
        }
      }
    }
    return child_population;
  }
}

#endif
