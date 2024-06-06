#pragma once

#include <array>
#include <cstdlib>
#include <print>
#include <type_traits>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <utility>
#include <map>
#include "matrix.hpp"
#include <any>


namespace nn :: function{
    template<typename T, int RowN, int ColN>
    Matrix<T, RowN, ColN> softmax(const Matrix<T, RowN, ColN>& m) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            T max = m.data[i * ColN];
            for (int j = 1; j < ColN; j++) {
                if (m.data[i * ColN + j] > max) {
                    max = m.data[i * ColN + j];
                }
            }
            T sum = 0;
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = std::exp(m.data[i * ColN + j] - max);
                sum += result.data[i * ColN + j];
            }
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] /= sum;
            }
        }
        return result;
    }
    template<typename T, int RowN, int ColN>
    Matrix<T, RowN, ColN> softmax_prime(const Matrix<T, RowN, ColN>& m) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = m.data[i * ColN + j] * (1 - m.data[i * ColN + j]);
            }
        }
        return result;
    }
    
    template<typename T, int RowN, int ColN>
    Matrix<T, RowN, ColN> onehot(const Matrix<T, RowN, 1>& m) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = (j == m.data[i]);
            }
        }
        return result;
    }

    template<typename T, int RowN, int ColN>    
    Matrix<T, RowN, ColN> relu(const Matrix<T, RowN, ColN>& m) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = m.data[i * ColN + j] > 0 ? m.data[i * ColN + j] : 0;
            }
        }
        return result;
    }
            
}

template<typename T, int RowN, int ColN, int ColN_B>
class Linear {
public:
    Linear() {
        weights = Matrix<T, RowN, ColN>(std::array<T, RowN * ColN>());
        bias = Matrix<T, ColN_B, ColN>(std::array<T, ColN * ColN_B>());
    }
    Linear(const Matrix<T, RowN, ColN>& w, const Matrix<T, ColN_B, ColN>& b) : weights(w), bias(b){
    }
    ~Linear() = default;

    Matrix<T, ColN_B, ColN> forward(const Matrix<T, ColN_B, RowN>& input) {
        return input * weights + bias;
    }

    //backward
    Matrix<T, ColN_B, RowN> backward(const Matrix<T, ColN_B, RowN>& input, const Matrix<T,ColN_B, ColN>& grad) {
        weights_grad = input.transpose() * grad;
        bias_grad = grad;
        return grad * weights.transpose();
    }

    void step(T learning_rate) {
        weights -= weights_grad * learning_rate;
        bias -= bias_grad * learning_rate;
    }

    Linear& zero_grad() {
        std::memset(weights_grad.data.data(), 0, sizeof(T) * RowN * ColN);
        std::memset(bias_grad.data.data(), 0, sizeof(T) * ColN_B * ColN);
        return *this;
    }

    


// private:
    Matrix<T, RowN, ColN> weights;
    Matrix<T, ColN_B, ColN> bias;
    Matrix<T, RowN, ColN> weights_grad;
    Matrix<T, ColN_B, ColN> bias_grad;
};

template<typename T, int RowN, int ColN>
class Dropout {
public:
    Dropout(int p):p(p) {
        this->reset();
    }
    ~Dropout() = default;

    Matrix<T, RowN, ColN> forward(const Matrix<T, RowN, ColN>& input) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = input.data[i] * dropout_mask.data[i];
        }

        return result;
    }

    Matrix<T, RowN, ColN> backward(const Matrix<T, RowN, ColN>& input, const Matrix<T, RowN, ColN>& grad) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = grad.data[i] * dropout_mask.data[i];
        }
        return result;
    }

    void reset() {
        for (int i = 0; i < RowN * ColN; i++) {
            dropout_mask.data[i] = (rand() % 100) < p ? 0 : 1;
        }
    }

    int p;
    Matrix<T, RowN, ColN> dropout_mask;
    
};

template<typename T, int RowN, int ColN>
class Softmax {
public:
    Softmax() {}
    ~Softmax() = default;

    Matrix<T, RowN, ColN> forward(const Matrix<T, RowN, ColN>& input) {
        return nn::function::softmax(input);
    }

    Matrix<T, RowN, ColN> backward(const Matrix<T, RowN, ColN>& input, const Matrix<T, RowN, ColN>& grad) {
        return grad;
    }

    void step(T learning_rate) {}
    
};


template<typename T, int RowN, int ColN>
class CrossEntropy {
public:
    CrossEntropy() {}
    ~CrossEntropy() = default;

    T forward(const Matrix<T, RowN, ColN>& input, const Matrix<T, RowN, ColN>& target) {
        T loss = 0;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                loss += target.data[i * ColN + j] * std::log(input.data[i * ColN + j]);
            }
        }
        return -loss;
    }

    //backward
    Matrix<T, RowN, ColN> backward(const Matrix<T, RowN, ColN>& input, const Matrix<T, RowN, ColN>& target) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = -target.data[i * ColN + j] + input.data[i * ColN + j];
            }
        }
        return result;
    }

};



template<typename T, int RowN, int ColN>
class ReLU {
public:
    ReLU() {}
    ~ReLU() = default;

    Matrix<T, RowN, ColN> forward(const Matrix<T, RowN, ColN>& input) {
        return nn::function::relu(input);
    }

    Matrix<T, RowN, ColN> backward(const Matrix<T, RowN, ColN>& input, const Matrix<T, RowN, ColN>& grad) {
        Matrix<T, RowN, ColN> result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = input.data[i] > 0 ? grad.data[i] : 0;
        }
        return result;
    }

    void step(T learning_rate) {}
    
};



template<typename T>
struct input_type_trait;

template<typename ClassType, typename ReturnType, typename ArgType>
struct input_type_trait<ReturnType(ClassType::*)(ArgType)> {
    using type = ArgType;
};

template<typename ClassType, typename ReturnType, typename ArgType>
struct input_type_trait<ReturnType(ClassType::*)(ArgType) const> {
    using type = ArgType;
};

template<typename Layer>
using forward_input_type = typename input_type_trait<decltype(&Layer::forward)>::type;

template<typename ...Layers>
class Sequential {
public:

    Sequential(Layers&... layers): layers(std::tie(layers...)) {}

    ~Sequential() = default;

    template<int RowN, int ColN>
    auto forward(const Matrix<double, RowN, ColN>& input) {
        layer_inputs.clear();
        layer_inputs.push_back(input);
        return forward_helper(input, layers);
    }

    template<std::size_t I = 0, typename Input, typename... T>
    auto forward_helper(Input&& input, std::tuple<T&...>& t) {
        if constexpr (I < sizeof...(T)) {
            auto& layer = std::get<I>(t);
            if constexpr (I > 0) {
                layer_inputs.push_back(input);
            }
            auto output = layer.forward(std::forward<Input>(input));
            return forward_helper<I + 1>(std::move(output), t);
        } else {
            return std::forward<Input>(input);
        }
    }

    template<typename Loss>
    [[maybe_unused]] auto backward(Loss&& loss) {
        auto gradient = std::forward<Loss>(loss);
        backward_helper<sizeof...(Layers)-1>(gradient);
        return gradient;
    }

    template<std::size_t I, typename Gradient>
    auto backward_helper(Gradient& gradient) {
        if constexpr (I < sizeof...(Layers)) {
            auto layer_input = std::any_cast<forward_input_type<typename std::tuple_element<I, std::tuple<Layers...>>::type >>(layer_inputs[I]);
            auto gradient_out = std::get<I>(layers).backward(layer_input, gradient);

            if constexpr (I > 0) {
                return backward_helper<I-1>(gradient_out);
            } else {
                return gradient_out;
            }
        } else {
            std::unreachable();
        }
    }

private:
    std::tuple<Layers&...> layers;
    std::vector<std::any> layer_inputs;

};