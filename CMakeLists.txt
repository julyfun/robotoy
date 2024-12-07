cmake_minimum_required(VERSION 3.4...3.18)
project("fast_math")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package("pybind11" REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

pybind11_add_module(itp_state "itp_state/itp_state.cpp")
pybind11_add_module(ring_buffer "container/ring_buffer.cpp")
