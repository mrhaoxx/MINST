#pragma once

#include <array>
#include <tuple>

template<typename T, int ...dims>
class Tensor {
public:
    template<int n>
    constexpr int shape() {
        return std::get<n>(std::make_tuple(dims...));
    }
    std::array<T, (sizeof...(dims) > 0 ? (dims * ...) : 1)> data;

    Tensor() : data() {}
    explicit Tensor(std::array<T, (dims * ...)> data): data(std::move(data)) {}
    virtual ~Tensor() = default;

    template<int ...new_dims>
    Tensor<T, new_dims...> reshape() const {
        static_assert((dims * ...) == (new_dims * ...));
        return Tensor<T, new_dims...>(data);
    }

    auto& get(int ...indices) {
        static_assert(sizeof...(indices) == sizeof...(dims), "Number of indices must match the number of dimensions");
        int idx[] = {indices...};
        const int sz[] = {dims...};

        size_t offset = 0;
        size_t acc = 1;
        for (int i = sizeof...(dims) - 1; i >= 0; --i) {
            assert(idx[i] < sz[i] && "Index out of bounds");
            offset += idx[i] * acc;
            acc *= sz[i];
        }
        return data[offset];
    }
};