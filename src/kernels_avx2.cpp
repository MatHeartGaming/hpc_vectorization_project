#include "kernels.hpp"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

namespace {

float horizontal_sum(const __m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void validate_fft_count(const std::size_t count) {
    if (count < 2 || (count & (count - 1)) != 0) {
        throw std::invalid_argument("FFT count must be a power of two and at least 2");
    }
}

void bit_reverse_permute(float* real, float* imaginary, const std::size_t count) {
    std::size_t j = 0;
    for (std::size_t i = 1; i < count; ++i) {
        std::size_t bit = count >> 1;
        while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) { std::swap(real[i], real[j]); std::swap(imaginary[i], imaginary[j]); }
    }
}

} // namespace

void vector_copy_native_simd(const float* input, float* output, const std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) _mm256_storeu_ps(output + i, _mm256_loadu_ps(input + i));
    for (; i < count; ++i) output[i] = input[i];
}

#define DEFINE_BINARY_KERNEL(name, intrinsic, scalar_op) \
void name##_native_simd(const float* lhs, const float* rhs, float* output, const std::size_t count) { \
    std::size_t i = 0; \
    for (; i + 8 <= count; i += 8) { \
        const __m256 a = _mm256_loadu_ps(lhs + i); \
        const __m256 b = _mm256_loadu_ps(rhs + i); \
        _mm256_storeu_ps(output + i, intrinsic(a, b)); \
    } \
    for (; i < count; ++i) output[i] = lhs[i] scalar_op rhs[i]; \
}

DEFINE_BINARY_KERNEL(vector_add, _mm256_add_ps, +)
DEFINE_BINARY_KERNEL(vector_subtract, _mm256_sub_ps, -)
DEFINE_BINARY_KERNEL(vector_multiply, _mm256_mul_ps, *)
DEFINE_BINARY_KERNEL(vector_divide, _mm256_div_ps, /)
#undef DEFINE_BINARY_KERNEL

void vector_triad_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const float scale,
    const std::size_t count
) {
    std::size_t i = 0;
    const __m256 scale_vector = _mm256_set1_ps(scale);
    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + i);
        const __m256 b = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(output + i, _mm256_fmadd_ps(scale_vector, b, a));
    }
    for (; i < count; ++i) output[i] = lhs[i] + scale * rhs[i];
}

void saxpy_native_simd(const float scale, const float* x, float* y, const std::size_t count) {
    std::size_t i = 0;
    const __m256 scale_vector = _mm256_set1_ps(scale);
    for (; i + 8 <= count; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 yv = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(scale_vector, xv, yv));
    }
    for (; i < count; ++i) y[i] = scale * x[i] + y[i];
}

float dot_product_native_simd(const float* lhs, const float* rhs, const std::size_t count) {
    std::size_t i = 0;
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    for (; i + 16 <= count; i += 16) {
        accumulator0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i), _mm256_loadu_ps(rhs + i), accumulator0);
        accumulator1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i + 8), _mm256_loadu_ps(rhs + i + 8), accumulator1);
    }
    __m256 accumulator = _mm256_add_ps(accumulator0, accumulator1);
    for (; i + 8 <= count; i += 8) {
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i), _mm256_loadu_ps(rhs + i), accumulator);
    }
    float sum = horizontal_sum(accumulator);
    for (; i < count; ++i) sum += lhs[i] * rhs[i];
    return sum;
}

void polynomial_native_simd(
    const float* input,
    float* output,
    const std::size_t count,
    const float* c
) {
    std::size_t i = 0;
    const __m256 c0 = _mm256_set1_ps(c[0]);
    const __m256 c1 = _mm256_set1_ps(c[1]);
    const __m256 c2 = _mm256_set1_ps(c[2]);
    const __m256 c3 = _mm256_set1_ps(c[3]);
    const __m256 c4 = _mm256_set1_ps(c[4]);
    for (; i + 8 <= count; i += 8) {
        const __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = _mm256_fmadd_ps(c4, x, c3);
        result = _mm256_fmadd_ps(result, x, c2);
        result = _mm256_fmadd_ps(result, x, c1);
        result = _mm256_fmadd_ps(result, x, c0);
        _mm256_storeu_ps(output + i, result);
    }
    for (; i < count; ++i) {
        const float x = input[i];
        output[i] = (((c[4] * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0];
    }
}

void matrix_multiply_native_simd(
    const float* lhs,
    const float* rhs_transposed,
    float* output,
    const std::size_t dimension
) {
    for (std::size_t row = 0; row < dimension; ++row) {
        const float* lhs_row = lhs + row * dimension;
        for (std::size_t column = 0; column < dimension; ++column) {
            const float* rhs_row = rhs_transposed + column * dimension;
            output[row * dimension + column] = dot_product_native_simd(lhs_row, rhs_row, dimension);
        }
    }
}

void matrix_multiply_blocked_native_simd(
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
                    float* output_row = output + row * dimension;
                    for (std::size_t k = k_block; k < k_end; ++k) {
                        const __m256 lhs_vector = _mm256_set1_ps(lhs[row * dimension + k]);
                        const float* rhs_row = rhs + k * dimension;
                        std::size_t column = column_block;
                        for (; column + 8 <= column_end; column += 8) {
                            const __m256 current = _mm256_loadu_ps(output_row + column);
                            const __m256 right = _mm256_loadu_ps(rhs_row + column);
                            _mm256_storeu_ps(output_row + column, _mm256_fmadd_ps(lhs_vector, right, current));
                        }
                        for (; column < column_end; ++column) {
                            output_row[column] += lhs[row * dimension + k] * rhs_row[column];
                        }
                    }
                }
            }
        }
    }
}

void convolution_1d_native_simd(
    const float* input,
    const float* filter,
    float* output,
    const std::size_t input_count,
    const std::size_t filter_count
) {
    const std::size_t output_count = input_count - filter_count + 1;
    std::size_t i = 0;
    for (; i + 8 <= output_count; i += 8) _mm256_storeu_ps(output + i, _mm256_setzero_ps());
    for (; i < output_count; ++i) output[i] = 0.0F;
    for (std::size_t tap = 0; tap < filter_count; ++tap) {
        const __m256 coefficient = _mm256_set1_ps(filter[tap]);
        i = 0;
        for (; i + 8 <= output_count; i += 8) {
            const __m256 current = _mm256_loadu_ps(output + i);
            const __m256 values = _mm256_loadu_ps(input + i + tap);
            _mm256_storeu_ps(output + i, _mm256_fmadd_ps(values, coefficient, current));
        }
        for (; i < output_count; ++i) output[i] += input[i + tap] * filter[tap];
    }
}

void fft_radix2_native_simd(
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
        const std::size_t stride = count / length;
        for (std::size_t base = 0; base < count; base += length) {
            std::size_t j = 0;
            for (; j + 8 <= half; j += 8) {
                const __m256i twiddle_indices = _mm256_setr_epi32(
                    static_cast<int>((j + 0) * stride), static_cast<int>((j + 1) * stride),
                    static_cast<int>((j + 2) * stride), static_cast<int>((j + 3) * stride),
                    static_cast<int>((j + 4) * stride), static_cast<int>((j + 5) * stride),
                    static_cast<int>((j + 6) * stride), static_cast<int>((j + 7) * stride));
                const __m256 wr = _mm256_i32gather_ps(twiddle_real, twiddle_indices, 4);
                const __m256 wi = _mm256_i32gather_ps(twiddle_imaginary, twiddle_indices, 4);
                const std::size_t even = base + j;
                const std::size_t odd = even + half;
                const __m256 even_real = _mm256_loadu_ps(real + even);
                const __m256 even_imaginary = _mm256_loadu_ps(imaginary + even);
                const __m256 odd_real = _mm256_loadu_ps(real + odd);
                const __m256 odd_imaginary = _mm256_loadu_ps(imaginary + odd);
                const __m256 temporary_real = _mm256_fmsub_ps(wr, odd_real, _mm256_mul_ps(wi, odd_imaginary));
                const __m256 temporary_imaginary = _mm256_fmadd_ps(wr, odd_imaginary, _mm256_mul_ps(wi, odd_real));
                _mm256_storeu_ps(real + even, _mm256_add_ps(even_real, temporary_real));
                _mm256_storeu_ps(imaginary + even, _mm256_add_ps(even_imaginary, temporary_imaginary));
                _mm256_storeu_ps(real + odd, _mm256_sub_ps(even_real, temporary_real));
                _mm256_storeu_ps(imaginary + odd, _mm256_sub_ps(even_imaginary, temporary_imaginary));
            }
            for (; j < half; ++j) {
                const std::size_t even = base + j;
                const std::size_t odd = even + half;
                const std::size_t twiddle = j * stride;
                const float wr = twiddle_real[twiddle];
                const float wi = twiddle_imaginary[twiddle];
                const float temporary_real = wr * real[odd] - wi * imaginary[odd];
                const float temporary_imaginary = wr * imaginary[odd] + wi * real[odd];
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

float gather_sum_native_simd(const float* values, const std::uint32_t* indices, const std::size_t count) {
    std::size_t i = 0;
    __m256 accumulator = _mm256_setzero_ps();
    for (; i + 8 <= count; i += 8) {
        const __m256i index_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
        accumulator = _mm256_add_ps(accumulator, _mm256_i32gather_ps(values, index_vector, 4));
    }
    float sum = horizontal_sum(accumulator);
    for (; i < count; ++i) sum += values[indices[i]];
    return sum;
}

void scatter_add_native_simd(
    const float* values,
    const std::uint32_t* indices,
    float* output,
    const std::size_t count
) {
    std::size_t i = 0;
    alignas(32) float value_lanes[8];
    alignas(32) std::uint32_t index_lanes[8];
    for (; i + 8 <= count; i += 8) {
        _mm256_store_ps(value_lanes, _mm256_loadu_ps(values + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(index_lanes),
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i)));
        // AVX2 has gather but no general scatter; preserve collisions with scalar lane updates.
        for (std::size_t lane = 0; lane < 8; ++lane) output[index_lanes[lane]] += value_lanes[lane];
    }
    for (; i < count; ++i) output[indices[i]] += values[i];
}

void histogram_native_simd(
    const std::uint32_t* values,
    std::uint32_t* bins,
    const std::size_t count,
    const std::size_t bin_count
) {
    std::size_t i = 0;
    alignas(32) std::uint32_t lanes[8];
    for (; i + 8 <= count; i += 8) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i)));
        // Histogram conflicts require ordered scalar updates on AVX2.
        for (const std::uint32_t bin : lanes) if (bin < bin_count) ++bins[bin];
    }
    for (; i < count; ++i) if (values[i] < bin_count) ++bins[values[i]];
}

const char* native_simd_name() noexcept {
    return "x86 AVX2 + FMA (256-bit, 8 x float32)";
}
