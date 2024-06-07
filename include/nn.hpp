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
#include <any>

#include "Tensor.hpp"
#include "tensor.hpp"

namespace nn :: function{
    template<typename T, int RowN, int ColN>
    Tensor<T, RowN, ColN> softmax(const Tensor<T, RowN, ColN>& m) {
        Tensor<T, RowN, ColN> result;
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
    Tensor<T, RowN, ColN> softmax_prime(const Tensor<T, RowN, ColN>& m) {
        Tensor<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = m.data[i * ColN + j] * (1 - m.data[i * ColN + j]);
            }
        }
        return result;
    }
    
    template<typename T, int RowN, int ColN>
    Tensor<T, RowN, ColN> onehot(const Tensor<T, RowN, 1>& m) {
        Tensor<T, RowN, ColN> result;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                result.data[i * ColN + j] = (j == m.data[i]);
            }
        }
        return result;
    }

    template<typename T, int RowN, int ColN>    
    Tensor<T, RowN, ColN> relu(const Tensor<T, RowN, ColN>& m) {
        Tensor<T, RowN, ColN> result;
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
        weights = Tensor<T, RowN, ColN>(std::array<T, RowN * ColN>());
        bias = Tensor<T, ColN_B, ColN>(std::array<T, ColN * ColN_B>());
    }
    Linear(const Tensor<T, RowN, ColN>& w, const Tensor<T, ColN_B, ColN>& b) : weights(w), bias(b){
    }
    ~Linear() = default;

    Tensor<T, ColN_B, ColN> forward(const Tensor<T, ColN_B, RowN>& input) {
        return input * weights + bias;
    }

    //backward
    Tensor<T, ColN_B, RowN> backward(const Tensor<T, ColN_B, RowN>& input, const Tensor<T,ColN_B, ColN>& grad) {
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
    Tensor<T, RowN, ColN> weights;
    Tensor<T, ColN_B, ColN> bias;
    Tensor<T, RowN, ColN> weights_grad;
    Tensor<T, ColN_B, ColN> bias_grad;
};

template<typename T, int RowN, int ColN>
class Dropout {
public:
    Dropout(int p):p(p) {
        this->reset();
    }
    ~Dropout() = default;

    Tensor<T, RowN, ColN> forward(const Tensor<T, RowN, ColN>& input) {
        Tensor<T, RowN, ColN> result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = input.data[i] * dropout_mask.data[i];
        }

        return result;
    }

    Tensor<T, RowN, ColN> backward(const Tensor<T, RowN, ColN>& input, const Tensor<T, RowN, ColN>& grad) {
        Tensor<T, RowN, ColN> result;
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
    Tensor<T, RowN, ColN> dropout_mask;
    
};

template<typename T, int RowN, int ColN>
class Softmax {
public:
    Softmax() {}
    ~Softmax() = default;

    Tensor<T, RowN, ColN> forward(const Tensor<T, RowN, ColN>& input) {
        return nn::function::softmax(input);
    }

    Tensor<T, RowN, ColN> backward(const Tensor<T, RowN, ColN>& input, const Tensor<T, RowN, ColN>& grad) {
        return grad;
    }

    void step(T learning_rate) {}
    
};


template<typename T, int RowN, int ColN>
class CrossEntropy {
public:
    CrossEntropy() {}
    ~CrossEntropy() = default;

    T forward(const Tensor<T, RowN, ColN>& input, const Tensor<T, RowN, ColN>& target) {
        T loss = 0;
        for (int i = 0; i < RowN; i++) {
            for (int j = 0; j < ColN; j++) {
                loss += target.data[i * ColN + j] * std::log(input.data[i * ColN + j]);
            }
        }
        return -loss;
    }

    //backward
    Tensor<T, RowN, ColN> backward(const Tensor<T, RowN, ColN>& input, const Tensor<T, RowN, ColN>& target) {
        Tensor<T, RowN, ColN> result;
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

    Tensor<T, RowN, ColN> forward(const Tensor<T, RowN, ColN>& input) {
        return nn::function::relu(input);
    }

    Tensor<T, RowN, ColN> backward(const Tensor<T, RowN, ColN>& input, const Tensor<T, RowN, ColN>& grad) {
        Tensor<T, RowN, ColN> result;
        for (int i = 0; i < RowN * ColN; i++) {
            result.data[i] = input.data[i] > 0 ? grad.data[i] : 0;
        }
        return result;
    }

    void step(T learning_rate) {}
    
};


template<typename T,int c /* in_channels */, int oc /* out_channels */, int kh, int kw, int sh, int sw, int ph = 0, int pw = 0, int dh = 1,int dw = 1>
class Conv2d {
public:
    Conv2d() {}
    Conv2d(const Tensor<T, oc, c * kh * kw>& kernel) : kernel(kernel) {}
    ~Conv2d() = default;

    template<int h, int w>
    auto _get_blocks(const Tensor<T, c, h, w>& input) {
        constexpr int oh = (h + 2 * ph - ((kh - 1)*dh + 1)) / sh + 1;
        constexpr int ow = (w + 2 * pw - ((kw - 1)*dw + 1)) / sw + 1;
        Tensor<T, oh, ow, c, kh * kw> _tmp;
        for (int i = 0; i < oh; i++) {
            for (int j = 0; j < ow; j++) { // feature map
                for (int k = 0; k < c; k++){
                    std::memcpy(
                        &_tmp.data[i * ow * c * kh * kw + j * c * kh * kw + k * kh * kw], 
                        &input.data[k * h * w + i * sh * w + j * sw], 
                        kh * kw * sizeof(T)
                        );
                }

            }
        }
        return _tmp;
    }
    
    template<int h, int w>
    auto forward(const Tensor<T, c, h, w>& input) {
        constexpr int oh = (h + 2 * ph - ((kh - 1)*dh + 1)) / sh + 1;
        constexpr int ow = (w + 2 * pw - ((kw - 1)*dw + 1)) / sw + 1;

        const auto FA = this->_get_blocks(input).template reshape<oh, ow, c * kh * kw>();;

        const auto MFAT = FA.template reshape<oh * ow, c * kh * kw>().transpose();

        auto FAT = MFAT.template reshape<c * kh * kw ,oh * ow>();

        auto result = kernel * FAT;

        return result.template reshape<oc, oh, ow>();
    }

    template<int h, int w>
    

// private:
    Tensor<T, oc, c * kh * kw> kernel;

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
    auto forward(const Tensor<double, RowN, ColN>& input) {
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
        return backward_helper<sizeof...(Layers)-1>(loss);
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