#include "kernels.hpp"

void saxpy_auto(const float a, const float* x, float* y, const std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
