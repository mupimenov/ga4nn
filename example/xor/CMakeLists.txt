cmake_minimum_required(VERSION 3.5)
project(Xor)

set(Xor_MAJOR_VERSION 1)
set(Xor_MINOR_VERSION 0)

add_definitions(-std=c++11)

include_directories(${Core_SOURCE_DIR})

add_executable(xor main.cpp)
target_link_libraries(xor core)
