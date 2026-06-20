#pragma once

#include <cstddef>

using SaxpyKernel = void (*)(float a, const float* x, float* y, std::size_t n);

// y[i] = a * x[i] + y[i]
void saxpy_scalar(float a, const float* x, float* y, std::size_t n);
void saxpy_auto(float a, const float* x, float* y, std::size_t n);
void saxpy_native_simd(float a, const float* x, float* y, std::size_t n);

[[nodiscard]] const char* native_simd_name() noexcept;
