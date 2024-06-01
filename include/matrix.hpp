#pragma once

#include <array>
#include <cstdlib>
#include <print>
#include <iomanip>
#include <cmath>
#include <cstring>

#include <iostream>

template<typename T, int RowN, int ColN>
class Matrix {
public:
    Matrix() : data(){}
    explicit Matrix(std::array<T, ColN * RowN> data): data(std::move(data)) {}
    virtual ~Matrix() = default;

    template<int RowN_B, int ColN_B>
    Matrix<T, RowN, ColN_B> operator* (const Matrix<T, RowN_B, ColN_B>& other)  const{
        static_assert(ColN == RowN_B);
        
        Matrix<T, RowN, ColN_B> result;

        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN_B; j++) {
                T sum = 0;
                for (int k = 0; k < ColN; k++) {
                    sum += data[i * ColN + k] * other.data[k * ColN_B + j];
                }
                result.data[i * ColN_B + j] = sum;
            }
        }

        return result;
    }

    Matrix& operator*= (const T& scalar) {
        for (int i = 0; i < RowN * ColN; i++) {
            data[i] *= scalar;
        }
        return *this;
    }

    Matrix operator* (const T& scalar) const {
        Matrix result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }


    Matrix operator+ (const Matrix& other)  const {
        Matrix result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = data[i * ColN + j] + other.data[i * ColN + j];
            }
        }
        return result;
    }
    
    Matrix<T,RowN,ColN>& operator+= (const Matrix& other) {
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                data[i * ColN + j] += other.data[i * ColN + j];
            }
        }
        return *this;
    }

    Matrix<T,RowN,ColN>& operator-= (const Matrix& other) {
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                data[i * ColN + j] -= other.data[i * ColN + j];
            }
        }
        return *this;
    }

    Matrix<T,ColN,RowN> transpose()  const {
        Matrix<T,ColN,RowN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[j * RowN + i] = data[i * ColN + j];
            }
        }
        return result;
    }

    template<typename U = T>
    U reduce()  const {
        static_assert(RowN == 1 || ColN == 1);
        U sum = 0;
        for (int i = 0; i < RowN * ColN; i++) {
            sum += data[i];
        }
        return sum;
    }

    template<typename U = T>
    Matrix<U, RowN, ColN> scale(U scalar) {
        Matrix<U, RowN, ColN> result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
   


    template<int Row_B, int ColN_B>
    Matrix<T, Row_B,ColN_B> reshape() {
        static_assert(RowN * ColN == Row_B * ColN_B);
        Matrix<T, Row_B, ColN_B> result;
        std::memcpy(result.data.data(), data.data(), sizeof(T) * Row_B * ColN_B);
        return result;
    }

    // friend std::ostream& operator<<<>(std::ostream& os, const Matrix<T,RowN,ColN>& p);

// private:
    std::array<T, ColN * RowN> data;
};

template<typename T, int RowN, int ColN>
std::ostream& operator<<(std::ostream& os, const Matrix<T,RowN,ColN>& p) {
    for (int i = 0; i < RowN; i++) {
        for (int j = 0; j < ColN; j++) 
            std::print(os, "{0} ",p.data[i * ColN + j]);
        std::print(os, "\n");
    }
    return os;
}

template<typename T, int RowN, int ColN>
Matrix<T, RowN, ColN> random() {
    Matrix<T, RowN, ColN> result;
    for (int i = 0; i < RowN * ColN; i++) {
        result.data[i] = static_cast<T>(std::rand()) / RAND_MAX;
    }
    return result;
}