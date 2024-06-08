#pragma once

#include <array>
#include <mdspan>
#include <tuple>
#include <cassert>
#include <iostream>

template<typename T, int ...dims>
class Tensor {
public:

    static constexpr std::array<int, sizeof...(dims)> dimensions = {dims...};

    template<int n>
    constexpr int shape() const {
        return std::get<n>(std::make_tuple(dims...));
    }
    
    std::array<T, (sizeof...(dims) > 0 ? (dims * ...) : 1)> data;


    Tensor() : data() {}
    explicit Tensor(std::array<T, (dims * ...)> data): data(std::move(data)) {}
    virtual ~Tensor() = default;

    template<typename U>
    Tensor<U, dims...> scale(U scale) const {
        std::array<U, (dims * ...)> result_data;
        for (int i = 0; i < (dims * ...); i++) {
            result_data[i] = static_cast<U>(data[i] * scale);
        }
        return Tensor<U, dims...>(result_data);
    }

    template<int ...dimsB>
    auto operator* (const Tensor<T, dimsB...>& other)  const{
        static_assert(sizeof...(dims) == 2);
        static_assert(sizeof...(dimsB) == 2);
        static_assert(std::get<0>(std::make_tuple(dimsB...)) == dimensions[sizeof...(dims) - 1]);
        
        constexpr int rows = dimensions[0];
        constexpr int cols = dimensions[1];
        constexpr int colsB = std::get<1>(std::make_tuple(dimsB...));

        std::array<T, rows * colsB> result_data;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < colsB; j++) {
                T sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i * cols + k] * other.data[k * colsB + j];
                }
                result_data[i * colsB + j] = sum;
            }
        }

        return Tensor<T, rows, colsB>(result_data);
    }

    Tensor operator* (const T& scalar) const {
        std::array<T, (dims * ...)> result_data;
        for (int i = 0; i < (dims * ...); i++) {
            result_data[i] = data[i] * scalar;
        }
        return Tensor<T, dims...>(result_data);
    }

   Tensor& operator*= (const T& scalar) {
        for (int i = 0; i < (dims * ...); i++) {
            data[i] *= scalar;
        }
        return *this;
    }
    
    Tensor operator+ (const Tensor& other)  const {
        std::array<T, (dims * ...)> result_data;
        for (int i = 0; i < (dims * ...); i++) {
            result_data[i] = data[i] + other.data[i];
        }
        return Tensor<T, dims...>(result_data);
    }

    Tensor& operator+= (const Tensor& other) {
        for (int i = 0; i < (dims * ...); i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    Tensor operator- (const Tensor& other)  const {
        std::array<T, (dims * ...)> result_data;
        for (int i = 0; i < (dims * ...); i++) {
            result_data[i] = data[i] - other.data[i];
        }
        return Tensor<T, dims...>(result_data);
    }


    Tensor& operator-= (const Tensor& other) {
        for (int i = 0; i < (dims * ...); i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    template<int ...new_dims>
    Tensor<T, new_dims...> reshape() const {
        static_assert((dims * ...) == (new_dims * ...));
        return Tensor<T, new_dims...>(data);
    }

    auto span() const {
        return std::mdspan(data.data(), dimensions);
    }
    

    template<size_t... Is>
    auto extractSubdimensionImpl(int index, std::index_sequence<Is...>) const {
        constexpr int first_dim = dimensions[0];
        assert(index < first_dim && "Index out of range.");

        constexpr int sub_size = (dimensions[Is + 1] * ...);
        std::array<T, sub_size> sub_data;

        int start = index * sub_size;
        std::copy_n(data.begin() + start, sub_size, sub_data.begin());

        return Tensor<T, dimensions[Is + 1]...>(sub_data);
    }

    auto extractSubdimension(int index) const {
        static_assert(sizeof...(dims) > 1);
        return extractSubdimensionImpl(index, std::make_index_sequence<sizeof...(dims) - 1>());
    }


    auto transpose() const {
        static_assert(sizeof...(dims) == 2 || sizeof...(dims) == 1);

        if constexpr (sizeof...(dims) == 2) {
            constexpr int rows = dimensions[0];
            constexpr int cols = dimensions[1];
            std::array<T, (dims * ...)> transposed_data;

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    transposed_data[j * rows + i] = data[i * cols + j];
                }
            }

            return Tensor<T, cols, rows>(transposed_data);
        }
    }

/*
   def _rotpi(self,x):
        if x.ndim==3:
            return x[:,::-1,::-1]
        return x[:,:,::-1,::-1]
*/
    template<int degree>
    Tensor rotate() const {
        static_assert(degree  == 180);
        static_assert(sizeof...(dims) == 3 || sizeof...(dims) == 2);

        if constexpr (sizeof...(dims) == 3) {
            constexpr int channels = dimensions[0];
            constexpr int rows = dimensions[1];
            constexpr int cols = dimensions[2];
            std::array<T, (dims * ...)> rotated_data;

            for (int i = 0; i < channels; ++i) {
                for (int j = 0; j < rows; ++j) {
                    for (int k = 0; k < cols; ++k) {
                        rotated_data[i * rows * cols + j * cols + k] = data[i * rows * cols + (rows - j - 1) * cols + (cols - k - 1)];
                    }
                }
            }

            return Tensor<T, channels, rows, cols>(rotated_data);
        } else if constexpr (sizeof...(dims) == 2) {
            constexpr int rows = dimensions[0];
            constexpr int cols = dimensions[1];
            std::array<T, (dims * ...)> rotated_data;

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    rotated_data[(rows - i - 1) * cols + (cols - j - 1)] = data[i * cols + j];
                }
            }

            return Tensor<T, rows, cols>(rotated_data);
        }
    }

    template<int pr, int pc>
    auto pad() const {
        static_assert(sizeof...(dims) == 2 || sizeof...(dims) == 3);

        if constexpr (sizeof...(dims) == 2) {
            constexpr int rows = dimensions[0];
            constexpr int cols = dimensions[1];
            std::array<T,  (rows + 2 * pr) * (cols + 2 *pc)> padded_data;

            for (int i = 0; i < rows + 2 * pr; ++i) {
                for (int j = 0; j < cols + 2 * pc; ++j) {
                    if (i < pr || i >= rows + pr || j < pc || j >= cols + pc) {
                        padded_data[i * (cols + 2 * pc) + j] = 0;
                    } else {
                        padded_data[i * (cols + 2 * pc) + j] = data[(i - pr) * cols + (j - pc)];
                    }
                }
            }

            return Tensor<T, (rows + 2 * pr),(cols + 2 * pc)>(padded_data);
        } else if constexpr (sizeof...(dims) == 3) {
            constexpr int channels = dimensions[0];
            constexpr int rows = dimensions[1];
            constexpr int cols = dimensions[2];
            std::array<T, channels * (rows + 2 * pr) * (cols + 2 * pc)> padded_data;

            for (int i = 0; i < channels; ++i) {
                for (int j = 0; j < rows + 2 * pr; ++j) {
                    for (int k = 0; k < cols + 2 * pc; ++k) {
                        if (j < pr || j >= rows + pr || k < pc || k >= cols + pc) {
                            padded_data[i * (rows + 2 * pr) * (cols + 2 * pc) + j * (cols + 2 * pc) + k] = 0;
                        } else {
                            padded_data[i * (rows + 2 * pr) * (cols + 2 * pc) + j * (cols + 2 * pc) + k] = data[i * rows * cols + (j - pr) * cols + (k - pc)];
                        }
                    }
                }
            }

            return Tensor<T, channels, (rows + 2 * pr), (cols + 2 * pc)>(padded_data);
        }

    }


    std::ostream&  print(std::ostream& os, const std::string& indent = "", int depth = 0, int w = 4) const {
        os << indent << "[";
        if constexpr (sizeof...(dims) > 1) {
            os << "\n";
            for (int i = 0; i < dimensions[0]; ++i) {
                auto subTensor = extractSubdimension(i);
                subTensor.print(os, indent + "  ", depth + 1);
                if (i < dimensions[0] - 1) std::cout << ",";
                os << "\n";
            }
            std::cout << indent;
        } else {
            for (int i = 0; i < dimensions[0]; ++i) {
                std::cout << " " << std::setw(w) << data[i];
            }
        }
        os << "]";
        
        return os;
    }

};


template<typename T, int ...dims>
std::ostream& operator<<(std::ostream& os, const Tensor<T, dims...>& p) {
    return p.print(os);
}


template<typename T, int ...dims>
Tensor<T, dims...> random() {
    Tensor<T, dims...> t;
    for (int i = 0; i < (dims * ...); i++) {
        t.data[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    return t;
}
