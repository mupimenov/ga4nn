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
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <iostream>
#include <vector>

#include "connector.hpp"
#include "neural_net.hpp"
#include "neuron_factory.hpp"

#include "genetic.hpp"

#define USE_MUTATION 1

class my_data {
public:
  typedef std::shared_ptr<my_data> ptr;
  struct prime {
    double x;
    double y;
    prime() : x(0.0), y(0.0) {}
    prime(double x_, double y_) : x(x_), y(y_) {}
  };

  my_data(double T, double dt, size_t points) : m_prime(points) {
    double gain = dt / T;
    double x = 0.0;
    double y1 = 0.0;
    for (size_t i = 0; i < points; i++) {
      double dx = 1.0 / (points / 2);
      x += i < (points / 2)? dx: -dx;
      double y = y1 + gain * (x - y1);
      m_prime[i] = prime(x, y);
      y1 = y;
    }
  }

  size_t points() const { return m_prime.size(); }
  const prime &get_prime(size_t index) const { return m_prime[index]; }

private:
  std::vector<prime> m_prime;
};

class my_genotype : public ga4nn::genotype<std::vector<double> > {
public:
  typedef std::shared_ptr<my_genotype> ptr;
  explicit my_genotype( ga4nn::neural_net::ptr net_,
                        my_data::ptr data_,
                        const std::vector<double> &weights_) :
    ga4nn::genotype<std::vector<double> >(weights_),
    net(net_),
    data(data_),
    m_computed(false),
    m_fitval(0.0) {}

  virtual double fitness() {
    if (m_computed)
      return m_fitval;

    std::vector<double> input(2);
    net->set_weights(get_data());

    m_fitval = 0.0;
    double y1 = 0.0;
    for (size_t i = 0; i < data->points(); i++) {
      const my_data::prime &prime = data->get_prime(i);
      input[0] = prime.x;
      input[1] = y1;
      std::vector<double> output = net->compute(input);
      y1 = output[0];
      double error = prime.y - output[0];
      m_fitval += (error * error);
    }
    m_computed = true;
    return m_fitval;
  }

  void reset() {
    m_computed = false;
  }

  ga4nn::neural_net::ptr net;
  my_data::ptr data;

private:
  bool m_computed;
  double m_fitval;
};

class my_genotype_creator : public ga4nn::genotype_creator<my_genotype> {
public:
  typedef std::shared_ptr<my_genotype_creator> ptr;
  my_genotype_creator(ga4nn::neural_net::ptr net,
                      my_data::ptr data,
                      double lower_bound,
                      double upper_bound,
                      size_t size) :
    m_net(net),
    m_data(data),
    m_lower_bound(lower_bound),
    m_upper_bound(upper_bound),
    m_size(size) { std::srand(std::time(0)); }

  my_genotype::ptr make() {
    std::vector<double> weights(m_size);
    const double precision = 1000.;
    for (int i = 0; i < m_size; ++i) {
      weights[i] = (std::rand()
        % static_cast<int>((m_upper_bound - m_lower_bound) * precision))
        / precision + m_lower_bound;
    }
    return my_genotype::ptr(new my_genotype(m_net, m_data, weights));
  }
private:
  ga4nn::neural_net::ptr m_net;
  my_data::ptr m_data;
  double m_lower_bound;
  double m_upper_bound;
  size_t m_size;
};

class my_population : public ga4nn::rb_population<my_genotype> {
public:
  typedef std::shared_ptr<my_population> ptr;
};

class my_selection : public ga4nn::b_selection<my_population> {
public:
  typedef std::shared_ptr<my_selection> ptr;
};

class my_crossover : public ga4nn::crossover<my_genotype> {
public:
  typedef std::shared_ptr<my_crossover> ptr;
  explicit my_crossover(double dx, double lambda0) :
    m_dx(dx),
    m_lambda0(lambda0) {}

  virtual std::vector<my_genotype::ptr> cross(
    const std::vector<my_genotype::ptr> &p) {
    std::vector<my_genotype::ptr> vec(1);
    std::vector<double> dv(p[0]->get_data().size());
    double fitness = p[0]->fitness();

    vec[0] = my_genotype::ptr(
      new my_genotype(p[0]->net, p[0]->data, p[0]->get_data()));

    for (size_t i = 0; i < dv.size(); ++i) {
      vec[0]->get_data()[i] = p[0]->get_data()[i] + m_dx;
      vec[0]->reset();
      dv[i] = (vec[0]->fitness() - fitness) > 0? -1.0: 1.0;
    }

    double lambda = m_lambda0;
    const size_t iter_count = 10;
    for (size_t iter = 0; iter < iter_count; ++iter) {
      for (size_t i = 0; i < p[0]->get_data().size(); ++i) {
        vec[0]->get_data()[i] = p[0]->get_data()[i] + lambda * dv[i];
      }
      vec[0]->reset();
      if (vec[0]->fitness() < p[0]->fitness())
        break;
      lambda /= 2.0;
    }

    if (vec[0]->fitness() > p[0]->fitness()) {
      for (size_t i = 0; i < p[0]->get_data().size(); ++i)
        vec[0]->get_data()[i] = p[0]->get_data()[i];
    }

    return vec;
  }
private:
  double m_dx;
  double m_lambda0;
};

class my_mutation : public ga4nn::mutation<my_genotype> {
public:
  typedef std::shared_ptr<my_mutation> ptr;
  explicit my_mutation(int propability_perc) :
    m_propability(propability_perc) { std::srand(std::time(0)); }

  virtual my_genotype::ptr mutate(my_genotype::ptr g) {
#if !USE_MUTATION
    (void)m_propability;
    return g;
#else
    const int gain = 1000;
    int r = std::rand() % (100 * gain);
    if (r < (m_propability * gain)) {
      int pos1 = std::rand() % g->get_data().size();
      int pos2 = std::rand() % g->get_data().size();
      double tmp = g->get_data()[pos1];
      g->get_data()[pos1] = g->get_data()[pos2];
      g->get_data()[pos2] = tmp;
    }
    return g;
#endif
  }

private:
  int m_propability;
};

class my_stop_function : public ga4nn::stop_function<my_population> {
public:
  typedef std::shared_ptr<my_stop_function> ptr;
  explicit my_stop_function(size_t epoch_number) :
    m_counter(0),
    m_epoch_number(epoch_number) {}

  virtual bool done(my_population::ptr p) {
    if (m_counter < m_epoch_number) {
      my_genotype::ptr b = p->take_beauty();
      my_genotype::ptr m = p->take_monster();

      p->insert(b);
      p->insert(m);

      if (!m_best
        || (m_best && m_best->fitness() > b->fitness())) {
        std::cout << "Epoch number= " << (m_counter + 1)
          << "\tGenotype= " << b->fitness()
          << "(" << m->fitness() << ")" << std::endl;

        m_best = b;
      }

      ++m_counter;
      return false;
    }

    if (m_best) {
      m_best->net->set_weights(m_best->get_data());
      std::vector<double> input(2);
      double y1 = 0.0;
      std::cout << "=== Result ===" << std::endl;
      std::cout << "x\ty" << std::endl;
      for (size_t i = 0; i < m_best->data->points(); i++) {
        const my_data::prime &prime = m_best->data->get_prime(i);
        input[0] = prime.x;
        input[1] = y1;
        std::vector<double> output = m_best->net->compute(input);
        y1 = output[0];
        std::cout << prime.x << "\t" << output[0] << std::endl;
      }

      for (size_t i = 0; i < m_best->get_data().size(); i++) {
        std::cout << m_best->get_data()[i] << std::endl;
      }
    }

    return true;
  }
private:
  size_t m_counter;
  size_t m_epoch_number;
  my_genotype::ptr m_best;
};

int main(int argc, char const *argv[]) {
  ga4nn::neural_net::ptr net(new ga4nn::neural_net);

  ga4nn::layer::ptr input_layer(new ga4nn::layer);
  input_layer->add_neurons(ga4nn::input_neuron_factory(), 2);

  ga4nn::layer::ptr hidden_layer(new ga4nn::layer);
  hidden_layer->add_neurons(ga4nn::linear_neuron_factory(), 4);

  ga4nn::layer::ptr output_layer(new ga4nn::layer);
  output_layer->add_neurons(ga4nn::output_neuron_factory(), 1);

  hidden_layer->connect_back(input_layer, ga4nn::internal_connector());
  output_layer->connect_back(hidden_layer, ga4nn::internal_connector());

  net->add_layer(input_layer);
  net->add_layer(hidden_layer);
  net->add_layer(output_layer);

  my_population::ptr population(new my_population);
  my_data::ptr data(new my_data(0.01, 0.001, 100));
  std::cout << "=== Data ===" << std::endl;
  std::cout << "x\ty" << std::endl;
  for (size_t i = 0; i < data->points(); i++) {
    const my_data::prime &p = data->get_prime(i);
    std::cout << p.x << "\t" << p.y << std::endl;
  }
  my_genotype_creator::ptr genotype_creator(
    new my_genotype_creator(net, data,
                            -1.0, 1.0,
                            net->get_weights().size()));

  ga4nn::fill_population<my_population,my_genotype_creator>(
    population,
    genotype_creator,
    600);

  my_selection::ptr selection(new my_selection);
  my_crossover::ptr crossover(new my_crossover(0.001, 0.001));
  my_mutation::ptr mutation(new my_mutation(5));
  my_stop_function::ptr stop_function(new my_stop_function(100));

  std::cout << "=== Evolve ===" << std::endl;

  my_population::ptr new_population = ga4nn::evolve<
    my_population,
    my_selection,
    my_crossover,
    my_mutation,
    my_stop_function
    >(population, selection, crossover, mutation, stop_function);

  return 0;
}
