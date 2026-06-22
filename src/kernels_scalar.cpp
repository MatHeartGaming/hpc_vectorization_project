#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER)
#define SIMD_NO_VECTORIZE __pragma(loop(no_vector))
#else
#define SIMD_NO_VECTORIZE
#endif

namespace {

void validate_fft_count(const std::size_t count) {
    if (count < 2 || (count & (count - 1)) != 0) {
        throw std::invalid_argument("FFT count must be a power of two and at least 2");
    }
}

void bit_reverse_permute(float* real, float* imaginary, const std::size_t count) {
    std::size_t j = 0;
    for (std::size_t i = 1; i < count; ++i) {
        std::size_t bit = count >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imaginary[i], imaginary[j]);
        }
    }
}

} // namespace

void vector_copy_scalar(const float* input, float* output, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[i] = input[i];
}

void vector_add_scalar(const float* lhs, const float* rhs, float* output, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[i] = lhs[i] + rhs[i];
}

void vector_subtract_scalar(const float* lhs, const float* rhs, float* output, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[i] = lhs[i] - rhs[i];
}

void vector_multiply_scalar(const float* lhs, const float* rhs, float* output, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[i] = lhs[i] * rhs[i];
}

void vector_divide_scalar(const float* lhs, const float* rhs, float* output, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[i] = lhs[i] / rhs[i];
}

void vector_triad_scalar(const float* lhs, const float* rhs, float* output, const float scale, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[i] = lhs[i] + scale * rhs[i];
}

void saxpy_scalar(const float scale, const float* x, float* y, const std::size_t count) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) y[i] = scale * x[i] + y[i];
}

float dot_product_scalar(const float* lhs, const float* rhs, const std::size_t count) {
    float sum = 0.0F;
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) sum += lhs[i] * rhs[i];
    return sum;
}

void polynomial_scalar(const float* input, float* output, const std::size_t count, const float* c) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) {
        const float x = input[i];
        output[i] = (((c[4] * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0];
    }
}

void matrix_multiply_scalar(
    const float* lhs,
    const float* rhs_transposed,
    float* output,
    const std::size_t dimension
) {
    for (std::size_t row = 0; row < dimension; ++row) {
        const float* lhs_row = lhs + row * dimension;
        for (std::size_t column = 0; column < dimension; ++column) {
            const float* rhs_row = rhs_transposed + column * dimension;
            float sum = 0.0F;
            SIMD_NO_VECTORIZE
            for (std::size_t k = 0; k < dimension; ++k) sum += lhs_row[k] * rhs_row[k];
            output[row * dimension + column] = sum;
        }
    }
}

void matrix_multiply_blocked_scalar(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t dimension,
    const std::size_t block_size
) {
    std::fill(output, output + dimension * dimension, 0.0F);
    for (std::size_t row_block = 0; row_block < dimension; row_block += block_size) {
        for (std::size_t k_block = 0; k_block < dimension; k_block += block_size) {
            for (std::size_t column_block = 0; column_block < dimension; column_block += block_size) {
                const std::size_t row_end = std::min(row_block + block_size, dimension);
                const std::size_t k_end = std::min(k_block + block_size, dimension);
                const std::size_t column_end = std::min(column_block + block_size, dimension);
                for (std::size_t row = row_block; row < row_end; ++row) {
                    for (std::size_t k = k_block; k < k_end; ++k) {
                        const float lhs_value = lhs[row * dimension + k];
                        float* output_row = output + row * dimension;
                        const float* rhs_row = rhs + k * dimension;
                        SIMD_NO_VECTORIZE
                        for (std::size_t column = column_block; column < column_end; ++column) {
                            output_row[column] += lhs_value * rhs_row[column];
                        }
                    }
                }
            }
        }
    }
}

void convolution_1d_scalar(
    const float* input,
    const float* filter,
    float* output,
    const std::size_t input_count,
    const std::size_t filter_count
) {
    const std::size_t output_count = input_count - filter_count + 1;
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < output_count; ++i) output[i] = 0.0F;
    for (std::size_t tap = 0; tap < filter_count; ++tap) {
        const float coefficient = filter[tap];
        SIMD_NO_VECTORIZE
        for (std::size_t i = 0; i < output_count; ++i) output[i] += input[i + tap] * coefficient;
    }
}

void fft_radix2_scalar(
    float* real,
    float* imaginary,
    const std::size_t count,
    const float* twiddle_real,
    const float* twiddle_imaginary
) {
    validate_fft_count(count);
    bit_reverse_permute(real, imaginary, count);
    for (std::size_t length = 2; length <= count; length <<= 1) {
        const std::size_t half = length >> 1;
        const std::size_t twiddle_stride = count / length;
        for (std::size_t base = 0; base < count; base += length) {
            SIMD_NO_VECTORIZE
            for (std::size_t j = 0; j < half; ++j) {
                const std::size_t even = base + j;
                const std::size_t odd = even + half;
                const std::size_t twiddle = j * twiddle_stride;
                const float wr = twiddle_real[twiddle];
                const float wi = twiddle_imaginary[twiddle];
                const float odd_real = real[odd];
                const float odd_imaginary = imaginary[odd];
                const float temporary_real = wr * odd_real - wi * odd_imaginary;
                const float temporary_imaginary = wr * odd_imaginary + wi * odd_real;
                const float even_real = real[even];
                const float even_imaginary = imaginary[even];
                real[even] = even_real + temporary_real;
                imaginary[even] = even_imaginary + temporary_imaginary;
                real[odd] = even_real - temporary_real;
                imaginary[odd] = even_imaginary - temporary_imaginary;
            }
        }
    }
}

float gather_sum_scalar(const float* values, const std::uint32_t* indices, const std::size_t count) {
    float sum = 0.0F;
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) sum += values[indices[i]];
    return sum;
}

void scatter_add_scalar(
    const float* values,
    const std::uint32_t* indices,
    float* output,
    const std::size_t count
) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) output[indices[i]] += values[i];
}

void histogram_scalar(
    const std::uint32_t* values,
    std::uint32_t* bins,
    const std::size_t count,
    const std::size_t bin_count
) {
    SIMD_NO_VECTORIZE
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bin = values[i];
        if (bin < bin_count) ++bins[bin];
    }
}

#undef SIMD_NO_VECTORIZE
