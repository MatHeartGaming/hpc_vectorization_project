#include "kernels.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <stdexcept>

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
        while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) { std::swap(real[i], real[j]); std::swap(imaginary[i], imaginary[j]); }
    }
}

} // namespace

void vector_copy_native_simd(const float* input, float* output, const std::size_t count) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) vst1q_f32(output + i, vld1q_f32(input + i));
    for (; i < count; ++i) output[i] = input[i];
}

#define DEFINE_BINARY_KERNEL(name, intrinsic, scalar_op) \
void name##_native_simd(const float* lhs, const float* rhs, float* output, const std::size_t count) { \
    std::size_t i = 0; \
    for (; i + 4 <= count; i += 4) { \
        vst1q_f32(output + i, intrinsic(vld1q_f32(lhs + i), vld1q_f32(rhs + i))); \
    } \
    for (; i < count; ++i) output[i] = lhs[i] scalar_op rhs[i]; \
}

DEFINE_BINARY_KERNEL(vector_add, vaddq_f32, +)
DEFINE_BINARY_KERNEL(vector_subtract, vsubq_f32, -)
DEFINE_BINARY_KERNEL(vector_multiply, vmulq_f32, *)
DEFINE_BINARY_KERNEL(vector_divide, vdivq_f32, /)
#undef DEFINE_BINARY_KERNEL

void vector_triad_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const float scale,
    const std::size_t count
) {
    std::size_t i = 0;
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    for (; i + 4 <= count; i += 4) {
        vst1q_f32(output + i, vfmaq_f32(vld1q_f32(lhs + i), scale_vector, vld1q_f32(rhs + i)));
    }
    for (; i < count; ++i) output[i] = lhs[i] + scale * rhs[i];
}

void saxpy_native_simd(const float scale, const float* x, float* y, const std::size_t count) {
    std::size_t i = 0;
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    for (; i + 4 <= count; i += 4) {
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), scale_vector, vld1q_f32(x + i)));
    }
    for (; i < count; ++i) y[i] = scale * x[i] + y[i];
}

float dot_product_native_simd(const float* lhs, const float* rhs, const std::size_t count) {
    std::size_t i = 0;
    float32x4_t accumulator0 = vdupq_n_f32(0.0F);
    float32x4_t accumulator1 = vdupq_n_f32(0.0F);
    for (; i + 8 <= count; i += 8) {
        accumulator0 = vfmaq_f32(accumulator0, vld1q_f32(lhs + i), vld1q_f32(rhs + i));
        accumulator1 = vfmaq_f32(accumulator1, vld1q_f32(lhs + i + 4), vld1q_f32(rhs + i + 4));
    }
    float32x4_t accumulator = vaddq_f32(accumulator0, accumulator1);
    for (; i + 4 <= count; i += 4) {
        accumulator = vfmaq_f32(accumulator, vld1q_f32(lhs + i), vld1q_f32(rhs + i));
    }
    float sum = vaddvq_f32(accumulator);
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
    const float32x4_t c0 = vdupq_n_f32(c[0]);
    const float32x4_t c1 = vdupq_n_f32(c[1]);
    const float32x4_t c2 = vdupq_n_f32(c[2]);
    const float32x4_t c3 = vdupq_n_f32(c[3]);
    const float32x4_t c4 = vdupq_n_f32(c[4]);
    for (; i + 4 <= count; i += 4) {
        const float32x4_t x = vld1q_f32(input + i);
        float32x4_t result = vfmaq_f32(c3, c4, x);
        result = vfmaq_f32(c2, result, x);
        result = vfmaq_f32(c1, result, x);
        result = vfmaq_f32(c0, result, x);
        vst1q_f32(output + i, result);
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
            output[row * dimension + column] =
                dot_product_native_simd(lhs_row, rhs_transposed + column * dimension, dimension);
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
                        const float lhs_scalar = lhs[row * dimension + k];
                        const float32x4_t lhs_vector = vdupq_n_f32(lhs_scalar);
                        const float* rhs_row = rhs + k * dimension;
                        std::size_t column = column_block;
                        for (; column + 4 <= column_end; column += 4) {
                            vst1q_f32(output_row + column,
                                vfmaq_f32(vld1q_f32(output_row + column), lhs_vector, vld1q_f32(rhs_row + column)));
                        }
                        for (; column < column_end; ++column) output_row[column] += lhs_scalar * rhs_row[column];
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
    for (; i + 4 <= output_count; i += 4) vst1q_f32(output + i, vdupq_n_f32(0.0F));
    for (; i < output_count; ++i) output[i] = 0.0F;
    for (std::size_t tap = 0; tap < filter_count; ++tap) {
        const float32x4_t coefficient = vdupq_n_f32(filter[tap]);
        i = 0;
        for (; i + 4 <= output_count; i += 4) {
            vst1q_f32(output + i,
                vfmaq_f32(vld1q_f32(output + i), vld1q_f32(input + i + tap), coefficient));
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
            for (; j + 4 <= half; j += 4) {
                alignas(16) float wr_lanes[4];
                alignas(16) float wi_lanes[4];
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    wr_lanes[lane] = twiddle_real[(j + lane) * stride];
                    wi_lanes[lane] = twiddle_imaginary[(j + lane) * stride];
                }
                const float32x4_t wr = vld1q_f32(wr_lanes);
                const float32x4_t wi = vld1q_f32(wi_lanes);
                const std::size_t even = base + j;
                const std::size_t odd = even + half;
                const float32x4_t even_real = vld1q_f32(real + even);
                const float32x4_t even_imaginary = vld1q_f32(imaginary + even);
                const float32x4_t odd_real = vld1q_f32(real + odd);
                const float32x4_t odd_imaginary = vld1q_f32(imaginary + odd);
                const float32x4_t temporary_real = vfmsq_f32(vmulq_f32(wr, odd_real), wi, odd_imaginary);
                const float32x4_t temporary_imaginary = vfmaq_f32(vmulq_f32(wr, odd_imaginary), wi, odd_real);
                vst1q_f32(real + even, vaddq_f32(even_real, temporary_real));
                vst1q_f32(imaginary + even, vaddq_f32(even_imaginary, temporary_imaginary));
                vst1q_f32(real + odd, vsubq_f32(even_real, temporary_real));
                vst1q_f32(imaginary + odd, vsubq_f32(even_imaginary, temporary_imaginary));
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
    float32x4_t accumulator = vdupq_n_f32(0.0F);
    alignas(16) float gathered[4];
    for (; i + 4 <= count; i += 4) {
        const uint32x4_t index_vector = vld1q_u32(indices + i);
        gathered[0] = values[vgetq_lane_u32(index_vector, 0)];
        gathered[1] = values[vgetq_lane_u32(index_vector, 1)];
        gathered[2] = values[vgetq_lane_u32(index_vector, 2)];
        gathered[3] = values[vgetq_lane_u32(index_vector, 3)];
        accumulator = vaddq_f32(accumulator, vld1q_f32(gathered));
    }
    float sum = vaddvq_f32(accumulator);
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
    alignas(16) float value_lanes[4];
    alignas(16) std::uint32_t index_lanes[4];
    for (; i + 4 <= count; i += 4) {
        vst1q_f32(value_lanes, vld1q_f32(values + i));
        vst1q_u32(index_lanes, vld1q_u32(indices + i));
        // Classic NEON has no general scatter; scalar lanes preserve duplicate-index semantics.
        for (std::size_t lane = 0; lane < 4; ++lane) output[index_lanes[lane]] += value_lanes[lane];
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
    alignas(16) std::uint32_t lanes[4];
    for (; i + 4 <= count; i += 4) {
        vst1q_u32(lanes, vld1q_u32(values + i));
        for (const std::uint32_t bin : lanes) if (bin < bin_count) ++bins[bin];
    }
    for (; i < count; ++i) if (values[i] < bin_count) ++bins[values[i]];
}

const char* native_simd_name() noexcept {
    return "ARM NEON (128-bit, 4 x float32)";
}
