/*
COPYRIGHT (c) 2017 Mikhail Pimenov

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
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>

#include <iostream>
#include <vector>

#include "connector.hpp"
#include "neural_net.hpp"
#include "neuron_factory.hpp"

#include "genetic.hpp"

#define USE_MUTATION 0

class balance {
public:
  typedef std::shared_ptr<balance> ptr;
  balance(double length, double theta0, double dt) :
    m_length(length),
    m_theta0(theta0),
    m_dt(dt) {
    reset();
  }

  void reset() {
    m_theta = m_theta0;
    m_deriv_theta = 0.0;
  }

  void accelerate(double acceleration) {
    const double g = 9.81;
    double accel_theta = (g * std::sin(m_theta)
      - acceleration * std::cos(m_theta)) / m_length;
    m_deriv_theta += accel_theta * m_dt;
    m_theta += m_deriv_theta * m_dt;
  }

  double get_length() const {
    return m_length;
  }

  double get_theta0() const {
    return m_theta0;
  }

  double get_dt() const {
    return m_dt;
  }

  double get_theta() const {
    return m_theta;
  }

  double get_deriv_theta() const {
    return m_deriv_theta;
  }

private:
  double m_length;
  double m_theta0;
  double m_dt;

  double m_theta;
  double m_deriv_theta;
};

class my_genotype : public ga4nn::genotype<std::vector<double> > {
public:
  typedef std::shared_ptr<my_genotype> ptr;
  explicit my_genotype( ga4nn::neural_net::ptr net_,
                        balance::ptr balance_,
                        double simulation_time_,
                        const std::vector<double> &weights_) :
    ga4nn::genotype<std::vector<double> >(weights_),
    m_net(net_),
    m_balance(balance_),
    m_simulation_time(simulation_time_),
    m_computed(false),
    m_fitval(0.0) {}

  virtual double fitness() {
    if (m_computed)
      return m_fitval;

    std::vector<double> input(2);
    m_net->set_weights(get_data());

    m_balance->reset();

    m_fitval = 0.0;
    for (double time = 0.0;
        time < m_simulation_time;
        time += m_balance->get_dt()) {
      input[0] = m_balance->get_theta();
      input[1] = m_balance->get_deriv_theta();
      std::vector<double> output = m_net->compute(input);
      m_balance->accelerate(output[0]);
      double error = 0.0 - m_balance->get_theta();
      m_fitval += (error * error);
    }
    m_computed = true;
    return m_fitval;
  }

  void reset() {
    m_computed = false;
  }

  ga4nn::neural_net::ptr m_net;
  balance::ptr m_balance;
  double m_simulation_time;

private:
  bool m_computed;
  double m_fitval;
};

class my_genotype_creator : public ga4nn::genotype_creator<my_genotype> {
public:
  typedef std::shared_ptr<my_genotype_creator> ptr;
  my_genotype_creator(ga4nn::neural_net::ptr net_,
                      balance::ptr balance_,
                      double simulation_time_,
                      double lower_bound_,
                      double upper_bound_,
                      size_t size_) :
    m_net(net_),
    m_balance(balance_),
    m_simulation_time(simulation_time_),
    m_lower_bound(lower_bound_),
    m_upper_bound(upper_bound_),
    m_size(size_) { std::srand(std::time(0)); }

  my_genotype::ptr make() {
    std::vector<double> weights(m_size);
    const double precision = 1000.;
    for (int i = 0; i < m_size; ++i) {
      weights[i] = (std::rand()
        % static_cast<int>((m_upper_bound - m_lower_bound) * precision))
        / precision + m_lower_bound;
    }
    return my_genotype::ptr(new my_genotype(m_net,
      m_balance,
      m_simulation_time,
      weights));
  }
private:
  ga4nn::neural_net::ptr m_net;
  balance::ptr m_balance;
  double m_simulation_time;
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
      new my_genotype(p[0]->m_net,
        p[0]->m_balance,
        p[0]->m_simulation_time,
        p[0]->get_data()));

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
      m_best->m_net->set_weights(m_best->get_data());
      std::vector<double> input(2);
      m_best->m_balance->reset();
      std::cout << "=== Result ===" << std::endl;
      std::cout << "theta\tderiv_theta\taccel\ttheta*" << std::endl;
      for (double time = 0.0;
          time < m_best->m_simulation_time;
          time += m_best->m_balance->get_dt()) {
        input[0] = m_best->m_balance->get_theta();
        input[1] = m_best->m_balance->get_deriv_theta();
        std::vector<double> output = m_best->m_net->compute(input);
        m_best->m_balance->accelerate(output[0]);
        std::cout << input[0] << "\t" << input[0]
          << "\t" << output[0]
          << "\t" << m_best->m_balance->get_theta() << std::endl;
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
#define RADIAN_FROM_DEGREES(degree) (M_PI * degree / 180.0)
  balance::ptr balance0(new balance(0.2, RADIAN_FROM_DEGREES(10.0), 0.005));
#undef RADIAN_FROM_DEGREES

  my_genotype_creator::ptr genotype_creator(
    new my_genotype_creator(net, balance0, 1.0,
                            -10.0, 10.0,
                            net->get_weights().size()));

  ga4nn::fill_population<my_population,my_genotype_creator>(
    population,
    genotype_creator,
    600);

  my_selection::ptr selection(new my_selection);
  my_crossover::ptr crossover(new my_crossover(0.01, 0.01));
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

  /*
  Fitness = 0.18

  Weights:
  -7.99475
  -3.92775
  -5.01075
  0.68825
  9.77625
  -1.81025
  -5.06975
  -5.98875
  -0.90675
  -8.14475
  9.87025
  -4.30075
  */

  return 0;
}
