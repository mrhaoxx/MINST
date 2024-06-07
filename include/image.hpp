#pragma once

#include <array>
#include <cstdlib>
#include <print>
#include <iomanip>


#include "tensor.hpp"


template<typename raw_pixel, int ColN, int RowN>
class Image : public Tensor<raw_pixel, ColN, RowN> {
public:
    Image() {}
    explicit Image(const std::array<raw_pixel, ColN * RowN>& pixels): Tensor<raw_pixel, ColN, RowN> (pixels) {}
    explicit Image(const Tensor<raw_pixel, ColN, RowN>& m): Tensor<raw_pixel, ColN, RowN> (m) {}
    ~Image() = default;

};

template<typename raw_pixel, int ColN, int RowN>
std::ostream& operator<<(std::ostream& os, const Image<raw_pixel,ColN,RowN>& p) {
    for (int i = 0; i < ColN; i++) {
        for (int j = 0; j < RowN; j++) {
            auto color = p.data[i * ColN + j];
            os << std::format("\033[48;2;{0};{0};{0}m ", color);
        }
        os << "\033[0m\n";
    }
    return os;
}