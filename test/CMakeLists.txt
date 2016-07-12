cmake_minimum_required(VERSION 3.5)
project(Test)

include_directories(${Core_SOURCE_DIR})
add_definitions(-std=c++11)

add_executable(testcore main.cpp core.cpp)

target_link_libraries(testcore
    core
    libgtest
    libgmock
)

install(TARGETS testcore DESTINATION bin)
