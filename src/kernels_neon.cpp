#include "kernels.hpp"

#include <arm_neon.h>

namespace {

float horizontal_sum(const float32x4_t value) {
    return vaddvq_f32(value);
}

} // namespace

void vector_add_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4_t a = vld1q_f32(lhs + i);
        const float32x4_t b = vld1q_f32(rhs + i);
        vst1q_f32(output + i, vaddq_f32(a, b));
    }
    for (; i < count; ++i) {
        output[i] = lhs[i] + rhs[i];
    }
}

void vector_subtract_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4_t a = vld1q_f32(lhs + i);
        const float32x4_t b = vld1q_f32(rhs + i);
        vst1q_f32(output + i, vsubq_f32(a, b));
    }
    for (; i < count; ++i) {
        output[i] = lhs[i] - rhs[i];
    }
}

void vector_multiply_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4_t a = vld1q_f32(lhs + i);
        const float32x4_t b = vld1q_f32(rhs + i);
        vst1q_f32(output + i, vmulq_f32(a, b));
    }
    for (; i < count; ++i) {
        output[i] = lhs[i] * rhs[i];
    }
}

void vector_divide_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4_t a = vld1q_f32(lhs + i);
        const float32x4_t b = vld1q_f32(rhs + i);
        vst1q_f32(output + i, vdivq_f32(a, b));
    }
    for (; i < count; ++i) {
        output[i] = lhs[i] / rhs[i];
    }
}

void saxpy_native_simd(
    const float scale,
    const float* x,
    float* y,
    const std::size_t count
) {
    std::size_t i = 0;
    const float32x4_t scale_vector = vdupq_n_f32(scale);

    for (; i + 4 <= count; i += 4) {
        const float32x4_t x_vector = vld1q_f32(x + i);
        const float32x4_t y_vector = vld1q_f32(y + i);
        vst1q_f32(y + i, vfmaq_f32(y_vector, x_vector, scale_vector));
    }
    for (; i < count; ++i) {
        y[i] = scale * x[i] + y[i];
    }
}

float dot_product_native_simd(
    const float* lhs,
    const float* rhs,
    const std::size_t count
) {
    std::size_t i = 0;
    float32x4_t accumulator = vdupq_n_f32(0.0F);

    for (; i + 4 <= count; i += 4) {
        const float32x4_t a = vld1q_f32(lhs + i);
        const float32x4_t b = vld1q_f32(rhs + i);
        accumulator = vfmaq_f32(accumulator, a, b);
    }

    float sum = horizontal_sum(accumulator);
    for (; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

void polynomial_native_simd(
    const float* input,
    float* output,
    const std::size_t count,
    const float* coefficients
) {
    std::size_t i = 0;
    const float32x4_t c0 = vdupq_n_f32(coefficients[0]);
    const float32x4_t c1 = vdupq_n_f32(coefficients[1]);
    const float32x4_t c2 = vdupq_n_f32(coefficients[2]);
    const float32x4_t c3 = vdupq_n_f32(coefficients[3]);
    const float32x4_t c4 = vdupq_n_f32(coefficients[4]);

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
        output[i] = (((coefficients[4] * x + coefficients[3]) * x + coefficients[2]) * x +
                     coefficients[1]) * x + coefficients[0];
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
            std::size_t k = 0;
            float32x4_t accumulator = vdupq_n_f32(0.0F);

            for (; k + 4 <= dimension; k += 4) {
                const float32x4_t a = vld1q_f32(lhs_row + k);
                const float32x4_t b = vld1q_f32(rhs_row + k);
                accumulator = vfmaq_f32(accumulator, a, b);
            }

            float sum = horizontal_sum(accumulator);
            for (; k < dimension; ++k) {
                sum += lhs_row[k] * rhs_row[k];
            }
            output[row * dimension + column] = sum;
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

    const float32x4_t zero = vdupq_n_f32(0.0F);
    for (; i + 4 <= output_count; i += 4) {
        vst1q_f32(output + i, zero);
    }
    for (; i < output_count; ++i) {
        output[i] = 0.0F;
    }

    for (std::size_t tap = 0; tap < filter_count; ++tap) {
        i = 0;
        const float coefficient = filter[tap];
        for (; i + 4 <= output_count; i += 4) {
            const float32x4_t samples = vld1q_f32(input + i + tap);
            const float32x4_t current = vld1q_f32(output + i);
            vst1q_f32(output + i, vfmaq_n_f32(current, samples, coefficient));
        }
        for (; i < output_count; ++i) {
            output[i] += input[i + tap] * coefficient;
        }
    }
}

const char* native_simd_name() noexcept {
    return "ARM NEON (128-bit, 4 x float32)";
}
