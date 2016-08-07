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
#ifndef __NEURON_HPP
#define __NEURON_HPP
#include <memory>
#include <vector>

namespace ga4nn {
class neuron {
public:
  typedef typename std::shared_ptr<neuron> ptr;
  struct link {
    typedef typename std::shared_ptr<link> ptr;
    neuron::ptr neuron_back;
    double weight;
    bool constant;

    explicit link(const neuron::ptr &neuron_back_,
                  double weight_,
                  bool constant_) :
                  neuron_back(neuron_back_),
                  weight(weight_),
                  constant(constant_) {}
  };

  virtual ~neuron();

  link::ptr create_link(neuron::ptr neuron_back,
                        double weight,
                        bool constant);
  size_t link_count() const;
  link::ptr get_link(size_t index) const;

  virtual bool activated() const = 0;
  virtual void compute() = 0;
  virtual double get_output() const = 0;
  virtual void set_output(double value) = 0;

  void set_weights(std::vector<double>::const_iterator &first,
                   const std::vector<double>::const_iterator &last);
  std::vector<double> get_weights() const;

protected:
  double forward();

  std::vector<link::ptr> d_link;
};

class input_neuron : public neuron {
public:
  input_neuron();
  virtual ~input_neuron();

  virtual bool activated() const;
  virtual void compute();
  virtual double get_output() const;
  virtual void set_output(double value);

private:
  struct prv;
  std::shared_ptr<prv> d;
};

class sigmoid_neuron : public neuron {
public:
  sigmoid_neuron();
  virtual ~sigmoid_neuron();

  virtual bool activated() const;
  virtual void compute();
  virtual double get_output() const;
  virtual void set_output(double value);

private:
  struct prv;
  std::shared_ptr<prv> d;
};

class linear_neuron : public neuron {
public:
  linear_neuron();
  virtual ~linear_neuron();

  virtual bool activated() const;
  virtual void compute();
  virtual double get_output() const;
  virtual void set_output(double value);

private:
  struct prv;
  std::shared_ptr<prv> d;
};

typedef linear_neuron output_neuron;

class feedback_neuron : public neuron {
public:
  explicit feedback_neuron(size_t history_len);
  virtual ~feedback_neuron();

  virtual bool activated() const;
  virtual void compute();
  virtual double get_output() const;
  virtual void set_output(double value);

private:
  struct prv;
  std::shared_ptr<prv> d;
};
}

#endif
