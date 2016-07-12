cmake_minimum_required(VERSION 3.5)
project(Core)

set(Core_MAJOR_VERSION 0)
set(Core_MINOR_VERSION 1)

add_definitions(-std=c++11)

add_library(core layer.cpp neural_net.cpp neuron_factory.cpp neuron.cpp)
