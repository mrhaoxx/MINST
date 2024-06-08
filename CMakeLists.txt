cmake_minimum_required(VERSION 3.15)

project(MNIST)

set(CMAKE_CXX_STANDARD 23)

# O3 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(include)

add_executable(MNIST src/main.cpp
        include/dataloader.hpp
        include/image.hpp)
