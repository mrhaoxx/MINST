cmake_minimum_required(VERSION 3.15)

project(MNIST)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)

add_executable(MNIST src/main.cpp
        include/dataloader.hpp
        include/image.hpp)
