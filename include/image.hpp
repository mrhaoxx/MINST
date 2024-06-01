#pragma once

#include <array>
#include <cstdlib>
#include <print>
#include <iomanip>


#include "matrix.hpp"


template<typename raw_pixel, int ColN, int RowN>
class Image : public Matrix<raw_pixel, ColN, RowN> {
public:
    Image() {}
    explicit Image(std::array<raw_pixel, ColN * RowN> pixels): Matrix<raw_pixel, ColN, RowN> (std::move(pixels)) {}
    explicit Image(const Matrix<raw_pixel, ColN, RowN>& m): Matrix<raw_pixel, ColN, RowN> (m) {}
    ~Image() = default;

};

template<typename raw_pixel, int ColN, int RowN>
std::ostream& operator<<(std::ostream& os, const Image<raw_pixel,ColN,RowN>& p) {
    for (int i = 0; i < ColN; i++) {
        for (int j = 0; j < RowN; j++) 
            std::print(os, "\033[48;2;{0};{0};{0}m ",p.data[i * ColN + j]);
        std::print(os, "\033[0m\n");
    }
    return os;
}