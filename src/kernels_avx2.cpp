#include "kernels.hpp"

#include <immintrin.h>

namespace {

float horizontal_sum(const __m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

} // namespace

void vector_add_native_simd(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + i);
        const __m256 b = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(output + i, _mm256_add_ps(a, b));
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
    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + i);
        const __m256 b = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(output + i, _mm256_sub_ps(a, b));
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
    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + i);
        const __m256 b = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(a, b));
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
    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + i);
        const __m256 b = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(output + i, _mm256_div_ps(a, b));
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
    const __m256 scale_vector = _mm256_set1_ps(scale);

    for (; i + 8 <= count; i += 8) {
        const __m256 x_vector = _mm256_loadu_ps(x + i);
        const __m256 y_vector = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(scale_vector, x_vector, y_vector));
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
    __m256 accumulator = _mm256_setzero_ps();

    for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + i);
        const __m256 b = _mm256_loadu_ps(rhs + i);
        accumulator = _mm256_fmadd_ps(a, b, accumulator);
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
    const __m256 c0 = _mm256_set1_ps(coefficients[0]);
    const __m256 c1 = _mm256_set1_ps(coefficients[1]);
    const __m256 c2 = _mm256_set1_ps(coefficients[2]);
    const __m256 c3 = _mm256_set1_ps(coefficients[3]);
    const __m256 c4 = _mm256_set1_ps(coefficients[4]);

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
            __m256 accumulator = _mm256_setzero_ps();

            for (; k + 8 <= dimension; k += 8) {
                const __m256 a = _mm256_loadu_ps(lhs_row + k);
                const __m256 b = _mm256_loadu_ps(rhs_row + k);
                accumulator = _mm256_fmadd_ps(a, b, accumulator);
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

    const __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= output_count; i += 8) {
        _mm256_storeu_ps(output + i, zero);
    }
    for (; i < output_count; ++i) {
        output[i] = 0.0F;
    }

    for (std::size_t tap = 0; tap < filter_count; ++tap) {
        i = 0;
        const __m256 coefficient = _mm256_set1_ps(filter[tap]);
        for (; i + 8 <= output_count; i += 8) {
            const __m256 samples = _mm256_loadu_ps(input + i + tap);
            const __m256 current = _mm256_loadu_ps(output + i);
            _mm256_storeu_ps(
                output + i,
                _mm256_fmadd_ps(samples, coefficient, current)
            );
        }
        for (; i < output_count; ++i) {
            output[i] += input[i + tap] * filter[tap];
        }
    }
}

const char* native_simd_name() noexcept {
    return "x86 AVX2 + FMA (256-bit, 8 x float32)";
}
