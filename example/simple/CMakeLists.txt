cmake_minimum_required(VERSION 3.4)
project(Simple)

set(Simple_MAJOR_VERSION 0)
set(Simple_MINOR_VERSION 1)

add_definitions(-std=c++11)

include_directories(${Core_SOURCE_DIR})

add_executable(simple main.cpp)
target_link_libraries(simple core)
