cmake_minimum_required(VERSION 3.4)
project(Balance-extended)

set(BalanceExtended_MAJOR_VERSION 0)
set(BalanceExtended_MINOR_VERSION 1)

add_definitions(-std=c++11)

include_directories(${Core_SOURCE_DIR})

add_executable(balance-extended main.cpp)
target_link_libraries(balance-extended core)
