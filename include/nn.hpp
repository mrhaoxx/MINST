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
#include <chrono>
#include <any>
#include <ctime>
#include "tensor.hpp"




const auto start_time = std::chrono::high_resolution_clock::now();

std::ostream& pt(std::ostream& os = std::cout) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    os << "+" << duration << "ms ";
    return os;
}

void progressbar(int current, int total)
{
    float progress = (float)current / total;
    int barWidth = 70;

    pt() << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " % " << current << "/" << total << " \r";
    std::cout.flush();
}



namespace nn :: function{
    template<typename T, int ColN>
    Tensor<T, ColN> softmax(const Tensor<T, ColN>& m) {
        Tensor<T, ColN> result;
        T max = m[0];
        for (int j = 1; j < ColN; j++) {
            if (m[j] > max) {
                max = m[j];
            }
        }
        T sum = 0;
        for (int j = 0; j < ColN; j++) {
            result[j] = std::exp(m[j] - max);
            sum += result[j];
        }
        for (int j = 0; j < ColN; j++) {
            result[j] /= sum;
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
    
    template<typename T, int ColN>
    Tensor<T, ColN> onehot(const Tensor<T, 1>& m) {
        Tensor<T, ColN> result;
        for (int j = 0; j < ColN; j++) {
            result[j] = (j == m[0]);
        }
        return result;
    }

    template<typename T, int ...dims>    
    Tensor<T, dims...> relu(const Tensor<T, dims...>& m) {
        Tensor<T, dims...> result;
        for (int i = 0; i < (dims * ...); i++) {
            result[i] = m[i] > 0 ? m[i] : 0;
        }
        return result;
    }
            
}

template<typename T, int RowN, int ColN>
class Linear {
public:
    Linear() {}
    Linear(const Tensor<T, RowN, ColN>& w, const Tensor<T, ColN>& b) : weights(w), bias(b){
    }
    ~Linear() = default;

    Tensor<T, ColN> forward(const Tensor<T, RowN>& input) {
        return (input.template reshape<1, RowN>() * weights).template reshape<ColN>() + bias;
    }

    //backward
    Tensor<T, RowN> backward(const Tensor<T, RowN>& input, const Tensor<T, ColN>& grad) {
        // std::cout << "input: " << input << std::endl;
              bias_grad = grad;
                // std::cout << "bgrad_saved";
        weights_grad = input.template reshape<1, RowN>().transpose() * grad.template reshape<1, ColN>();
                // std::cout << "grad_saved";
  

        return (grad.template reshape<1, ColN>() * weights.transpose()).template reshape<RowN>();
    }

    void step(T learning_rate) {
        weights -= weights_grad * learning_rate;
        bias -= bias_grad * learning_rate;
    }

    Linear& zero_grad() {
        std::memset(weights_grad.data.data(), 0, sizeof(T) * RowN * ColN);
        std::memset(bias_grad.data.data(), 0, sizeof(T) * ColN);
        return *this;
    }

    


// private:
    Tensor<T, RowN, ColN> weights;
    Tensor<T, ColN> bias;
    Tensor<T, RowN, ColN> weights_grad;
    Tensor<T, ColN> bias_grad;
};

template<typename T, int ...dims>
class Dropout {
public:
    Dropout(int p):p(p) {
        this->reset();
    }
    ~Dropout() = default;

    Tensor<T, dims...>  forward(const Tensor<T, dims...>& input) {
        Tensor<T, dims...> result;
        for (int i = 0; i < (dims * ...); i++) {
            result[i] = input[i] * dropout_mask[i];
        }

        return result;
    }

    Tensor<T, dims...> backward(const Tensor<T, dims...>& input, const Tensor<T, dims...>& grad) {
        Tensor<T, dims...> result;
        for (int i = 0; i < (dims * ...); i++) {
            result[i] = grad[i] * dropout_mask[i];
        }
        return result;
    }

    void reset() {
        for (int i = 0; i <  (dims * ...); i++) {
            dropout_mask[i] = (rand() % 100) < p ? 0 : 1;
        }
    }

    int p;
    Tensor<T,dims...> dropout_mask;
    
};

template<typename T, int ColN>
class Softmax {
public:
    Softmax() {}
    ~Softmax() = default;

    Tensor<T, ColN> forward(const Tensor<T, ColN>& input) {
        return nn::function::softmax(input);
    }

    Tensor<T, ColN> backward(const Tensor<T, ColN>& input, const Tensor<T, ColN>& grad) {
        return grad;
    }

    void step(T learning_rate) {}
    
};


template<typename T, int ColN>
class CrossEntropy {
public:
    CrossEntropy() {}
    ~CrossEntropy() = default;

    T forward(const Tensor<T, ColN>& input, const Tensor<T, ColN>& target) {
        T loss = 0;
        for (int j = 0; j < ColN; j++) {
            loss += target[j] * std::log(input[j]);
        }
        return -loss;
    }

    //backward
    Tensor<T, ColN> backward(const Tensor<T, ColN>& input, const Tensor<T, ColN>& target) {
        Tensor<T, ColN> result;
        for (int j = 0; j < ColN; j++) {
            result[j] = -target[j] + input[j];
        }
        return result;
    }

};



template<typename T, int ...dims>
class ReLU {
public:
    ReLU() {}
    ~ReLU() = default;

    Tensor<T, dims...> forward(const Tensor<T, dims...>& input) {
        return nn::function::relu(input);
    }

    Tensor<T, dims...> backward(const Tensor<T, dims...>& input, const Tensor<T, dims...>& grad) {
        Tensor<T, dims...> result;
        for (int i = 0; i < (dims * ... ); i++) {
            result[i] = input[i] > 0 ? grad[i] : 0;
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

    template<int h, int w, int _kh = kh, int _kw = kw>
    auto _get_blocks(const Tensor<T, c, h, w>& input) {
        constexpr int oh = (h + 2 * ph - ((_kh - 1)*dh + 1)) / sh + 1;
        constexpr int ow = (w + 2 * pw - ((_kw - 1)*dw + 1)) / sw + 1;
        Tensor<T, oh, ow, c, _kh * _kw> _tmp;
        for (int i = 0; i < oh; i++) {
            for (int j = 0; j < ow; j++) { // feature map
                for (int k = 0; k < c; k++){
                    std::memcpy(
                        &_tmp[i * ow * c * _kh * _kw + j * c * _kh * _kw + k * _kh * _kw], 
                        &input[k * h * w + i * sh * w + j * sw], 
                        _kh * _kw * sizeof(T)
                        );
                }

            }
        }
        return _tmp;
    }
    
    template<int h, int w>
    auto _prk_i(int i){
        return kernel.template reshape<oc, c , kh, kw>().extractSubdimension(i).template rotate<180>().template pad<h - kh,w - kw>();
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

//(c,h,w,kh,kw,sh,sw,oh,ow) 
//(c,2*pah-dkh,2*paw-dkw,pah,paw,sh,sw,oh,ow)

    template<int h, int w,int oh, int ow>
    auto backward(const Tensor<T, c, h, w>& input, const Tensor<T, oc, oh, ow>& grad) {        
        static_assert( oh == (h + 2 * ph - ((kh - 1)*dh + 1)) / sh + 1);
        static_assert( ow == (w + 2 * pw - ((kw - 1)*dw + 1)) / sw + 1);
        constexpr int pah = h + 2*ph;
        constexpr int paw = w + 2*pw;
        constexpr int dkh = (kh - 1)* dh + 1;
        constexpr int dkw = (kw - 1)* dw + 1;


        kernel_grad = grad.template reshape<oc, oh * ow>() * this->_get_blocks(input).template reshape<oh * ow, c * kh * kw>();

        Tensor<T, c, h, w> next_grad;

        pt() << "kernel_grad complete\n";

        for (int i = 0; i < oc;  i++){
            
            auto bPRKi = _get_blocks<2*pah - dkh, 2*paw - dkw ,pah, paw>(_prk_i<h,w>(i)).template reshape<oh * ow, c * h * w>();
            // pt() << "next_Grad get complete\n";

            auto xGrad = grad.extractSubdimension(i).template reshape<1, oh * ow>();
            // pt() << "next_Grad prep complete\n";

            next_grad += (xGrad * bPRKi).template reshape<c, h ,w>();
            // pt() << "next_Grad mult complete\n";
        }
        // pt() << "next_Grad complete\n";

        return next_grad;

    }
    
    void step(T learning_rate) {
        kernel -= kernel_grad * learning_rate;
    }


// private:
    Tensor<T, oc, c * kh * kw> kernel;
    Tensor<T, oc, c * kh * kw> kernel_grad;

};

template<typename T>
struct input_type_trait;

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

    template<int ...dims>
    auto forward(const Tensor<double, dims...>& input) {
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


template<typename T, int kh, int kw, int sh, int sw>
class MaxPool2d {
public:
    MaxPool2d() {}
    ~MaxPool2d() = default;

    template<int c, int h, int w>
    auto forward(const Tensor<T, c, h, w>& input) {

        constexpr int oh = (h - kh) / sh + 1;
        constexpr int ow = (w - kw) / sw + 1;

        Tensor<T,c, oh, ow> result;

        for (int i = 0; i < oh; i++) {
            for (int j = 0; j < ow; j++) {
                for (int k = 0; k < c; k++) {
                    T max = input[k * h * w + i * sh * w + j * sw];
                    for (int l = 0; l < kh; l++) {
                        for (int m = 0; m < kw; m++) {
                            max = std::max(max, input[k * h * w + (i * sh + l) * w + j * sw + m]);
                        }
                    }
                    result[k * oh * ow + i * ow + j] = max;
                }
            }
        }

        return result;
    }

    template<int c, int h, int w, int oh, int ow>
    auto backward(const Tensor<T, c, h, w>& input, const Tensor<T, c, oh, ow>& grad) {
        static_assert(oh == (h - kh) / sh + 1);
        static_assert(ow == (w - kw) / sw + 1);

        Tensor<T, c, h, w> result;

        for (int i = 0; i < oh; i++) {
            for (int j = 0; j < ow; j++) {
                for (int k = 0; k < c; k++) {
                    T max = input[k * h * w + i * sh * w + j * sw];
                    int max_idx = 0;
                    for (int l = 0; l < kh; l++) {
                        for (int m = 0; m < kw; m++) {
                            if (input[k * h * w + (i * sh + l) * w + j * sw + m] > max) {
                                max = input[k * h * w + (i * sh + l) * w + j * sw + m];
                                max_idx = l * kw + m;
                            }
                        }
                    }
                    result[k * h * w + (i * sh + max_idx / kw) * w + j * sw + max_idx % kw] = grad[k * oh * ow + i * ow + j];
                }
            }
        }

        return result;
    }

};

template<typename T>
class Flatten {
public:
    Flatten() {}
    ~Flatten() = default;

    template<int ...dims>
    auto forward(const Tensor<T, dims...>& input) {
        return input.template reshape<(dims * ... )>();
    }

    template<int ...dims>
    auto backward(const Tensor<T, dims...>& input, const Tensor<T, (dims * ... )>& grad) {
        return grad.template reshape<dims...>();
    }

};