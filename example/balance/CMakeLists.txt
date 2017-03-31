cmake_minimum_required(VERSION 3.4)
project(Balance)

set(Balance_MAJOR_VERSION 0)
set(Balance_MINOR_VERSION 1)

add_definitions(-std=c++11)

include_directories(${Core_SOURCE_DIR})

add_executable(balance main.cpp)
target_link_libraries(balance core)
