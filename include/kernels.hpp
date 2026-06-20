#pragma once

#include <cstddef>

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

// Element-wise streaming arithmetic.
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

// y[i] = scale * x[i] + y[i]
void saxpy_scalar(float scale, const float* x, float* y, std::size_t count);
void saxpy_auto(float scale, const float* x, float* y, std::size_t count);
void saxpy_native_simd(float scale, const float* x, float* y, std::size_t count);

// sum(lhs[i] * rhs[i])
float dot_product_scalar(const float* lhs, const float* rhs, std::size_t count);
float dot_product_auto(const float* lhs, const float* rhs, std::size_t count);
float dot_product_native_simd(const float* lhs, const float* rhs, std::size_t count);

// Horner evaluation with five coefficients:
// output[i] = ((((c4 * x + c3) * x + c2) * x + c1) * x + c0)
void polynomial_scalar(
    const float* input,
    float* output,
    std::size_t count,
    const float* coefficients
);
void polynomial_auto(
    const float* input,
    float* output,
    std::size_t count,
    const float* coefficients
);
void polynomial_native_simd(
    const float* input,
    float* output,
    std::size_t count,
    const float* coefficients
);

// Square matrix multiplication. rhs_transposed must contain transpose(rhs),
// so every output element is a dot product between two contiguous rows.
void matrix_multiply_scalar(
    const float* lhs,
    const float* rhs_transposed,
    float* output,
    std::size_t dimension
);
void matrix_multiply_auto(
    const float* lhs,
    const float* rhs_transposed,
    float* output,
    std::size_t dimension
);
void matrix_multiply_native_simd(
    const float* lhs,
    const float* rhs_transposed,
    float* output,
    std::size_t dimension
);

// Valid 1D convolution. The output contains input_count - filter_count + 1 values.
void convolution_1d_scalar(
    const float* input,
    const float* filter,
    float* output,
    std::size_t input_count,
    std::size_t filter_count
);
void convolution_1d_auto(
    const float* input,
    const float* filter,
    float* output,
    std::size_t input_count,
    std::size_t filter_count
);
void convolution_1d_native_simd(
    const float* input,
    const float* filter,
    float* output,
    std::size_t input_count,
    std::size_t filter_count
);

[[nodiscard]] const char* native_simd_name() noexcept;
