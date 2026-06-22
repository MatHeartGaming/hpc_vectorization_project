#pragma once

#include <cstddef>
#include <cstdint>

// Function-pointer aliases retained from the original project.
using BinaryKernel = void (*)(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t count
);

using SaxpyKernel = void (*)(
    float scale,
    const float* x,
    float* y,
    std::size_t count
);

using DotKernel = float (*)(
    const float* lhs,
    const float* rhs,
    std::size_t count
);

using PolynomialKernel = void (*)(
    const float* input,
    float* output,
    std::size_t count,
    const float* coefficients
);

using MatrixMultiplyKernel = void (*)(
    const float* lhs,
    const float* rhs_transposed,
    float* output,
    std::size_t dimension
);

using ConvolutionKernel = void (*)(
    const float* input,
    const float* filter,
    float* output,
    std::size_t input_count,
    std::size_t filter_count
);

// Streaming and element-wise kernels.
void vector_copy_scalar(const float* input, float* output, std::size_t count);
void vector_copy_auto(const float* input, float* output, std::size_t count);
void vector_copy_native_simd(const float* input, float* output, std::size_t count);

void vector_add_scalar(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_add_auto(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_add_native_simd(const float* lhs, const float* rhs, float* output, std::size_t count);

void vector_subtract_scalar(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_subtract_auto(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_subtract_native_simd(const float* lhs, const float* rhs, float* output, std::size_t count);

void vector_multiply_scalar(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_multiply_auto(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_multiply_native_simd(const float* lhs, const float* rhs, float* output, std::size_t count);

void vector_divide_scalar(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_divide_auto(const float* lhs, const float* rhs, float* output, std::size_t count);
void vector_divide_native_simd(const float* lhs, const float* rhs, float* output, std::size_t count);

// STREAM triad: output[i] = lhs[i] + scale * rhs[i].
void vector_triad_scalar(const float* lhs, const float* rhs, float* output, float scale, std::size_t count);
void vector_triad_auto(const float* lhs, const float* rhs, float* output, float scale, std::size_t count);
void vector_triad_native_simd(const float* lhs, const float* rhs, float* output, float scale, std::size_t count);

// SAXPY: y[i] = scale * x[i] + y[i].
void saxpy_scalar(float scale, const float* x, float* y, std::size_t count);
void saxpy_auto(float scale, const float* x, float* y, std::size_t count);
void saxpy_native_simd(float scale, const float* x, float* y, std::size_t count);

float dot_product_scalar(const float* lhs, const float* rhs, std::size_t count);
float dot_product_auto(const float* lhs, const float* rhs, std::size_t count);
float dot_product_native_simd(const float* lhs, const float* rhs, std::size_t count);

// Fourth-degree Horner polynomial. coefficients = {c0, c1, c2, c3, c4}.
void polynomial_scalar(const float* input, float* output, std::size_t count, const float* coefficients);
void polynomial_auto(const float* input, float* output, std::size_t count, const float* coefficients);
void polynomial_native_simd(const float* input, float* output, std::size_t count, const float* coefficients);

// Small GEMM. rhs_transposed must contain transpose(rhs), enabling contiguous dot products.
void matrix_multiply_scalar(const float* lhs, const float* rhs_transposed, float* output, std::size_t dimension);
void matrix_multiply_auto(const float* lhs, const float* rhs_transposed, float* output, std::size_t dimension);
void matrix_multiply_native_simd(const float* lhs, const float* rhs_transposed, float* output, std::size_t dimension);

// Cache-blocked large GEMM. rhs is in ordinary row-major layout.
void matrix_multiply_blocked_scalar(const float* lhs, const float* rhs, float* output, std::size_t dimension, std::size_t block_size);
void matrix_multiply_blocked_auto(const float* lhs, const float* rhs, float* output, std::size_t dimension, std::size_t block_size);
void matrix_multiply_blocked_native_simd(const float* lhs, const float* rhs, float* output, std::size_t dimension, std::size_t block_size);

// Valid one-dimensional convolution. output has input_count - filter_count + 1 elements.
void convolution_1d_scalar(const float* input, const float* filter, float* output, std::size_t input_count, std::size_t filter_count);
void convolution_1d_auto(const float* input, const float* filter, float* output, std::size_t input_count, std::size_t filter_count);
void convolution_1d_native_simd(const float* input, const float* filter, float* output, std::size_t input_count, std::size_t filter_count);

// In-place radix-2 FFT. count must be a power of two. Twiddle arrays contain
// cos(-2*pi*k/count) and sin(-2*pi*k/count), k in [0, count/2).
void fft_radix2_scalar(float* real, float* imaginary, std::size_t count, const float* twiddle_real, const float* twiddle_imaginary);
void fft_radix2_auto(float* real, float* imaginary, std::size_t count, const float* twiddle_real, const float* twiddle_imaginary);
void fft_radix2_native_simd(float* real, float* imaginary, std::size_t count, const float* twiddle_real, const float* twiddle_imaginary);

// Irregular memory-access kernels.
float gather_sum_scalar(const float* values, const std::uint32_t* indices, std::size_t count);
float gather_sum_auto(const float* values, const std::uint32_t* indices, std::size_t count);
float gather_sum_native_simd(const float* values, const std::uint32_t* indices, std::size_t count);

void scatter_add_scalar(const float* values, const std::uint32_t* indices, float* output, std::size_t count);
void scatter_add_auto(const float* values, const std::uint32_t* indices, float* output, std::size_t count);
void scatter_add_native_simd(const float* values, const std::uint32_t* indices, float* output, std::size_t count);

void histogram_scalar(const std::uint32_t* values, std::uint32_t* bins, std::size_t count, std::size_t bin_count);
void histogram_auto(const std::uint32_t* values, std::uint32_t* bins, std::size_t count, std::size_t bin_count);
void histogram_native_simd(const std::uint32_t* values, std::uint32_t* bins, std::size_t count, std::size_t bin_count);

[[nodiscard]] const char* native_simd_name() noexcept;
