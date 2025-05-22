cmake_minimum_required(VERSION 3.10)
project(kalman-filter-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-Wall)

# [exe]

add_executable(${PROJECT_NAME} "test.cpp")

# [dep.system]
find_package(Eigen3 REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})
