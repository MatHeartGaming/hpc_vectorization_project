#include "kernels.hpp"

void vector_add_auto(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = lhs[i] + rhs[i];
    }
}

void vector_subtract_auto(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = lhs[i] - rhs[i];
    }
}

void vector_multiply_auto(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = lhs[i] * rhs[i];
    }
}

void vector_divide_auto(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t count
) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = lhs[i] / rhs[i];
    }
}

void saxpy_auto(
    const float scale,
    const float* x,
    float* y,
    const std::size_t count
) {
    for (std::size_t i = 0; i < count; ++i) {
        y[i] = scale * x[i] + y[i];
    }
}

float dot_product_auto(
    const float* lhs,
    const float* rhs,
    const std::size_t count
) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

void polynomial_auto(
    const float* input,
    float* output,
    const std::size_t count,
    const float* coefficients
) {
    const float c0 = coefficients[0];
    const float c1 = coefficients[1];
    const float c2 = coefficients[2];
    const float c3 = coefficients[3];
    const float c4 = coefficients[4];

    for (std::size_t i = 0; i < count; ++i) {
        const float x = input[i];
        output[i] = (((c4 * x + c3) * x + c2) * x + c1) * x + c0;
    }
}

void matrix_multiply_auto(
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

            for (std::size_t k = 0; k < dimension; ++k) {
                sum += lhs_row[k] * rhs_row[k];
            }

            output[row * dimension + column] = sum;
        }
    }
}

void convolution_1d_auto(
    const float* input,
    const float* filter,
    float* output,
    const std::size_t input_count,
    const std::size_t filter_count
) {
    const std::size_t output_count = input_count - filter_count + 1;

    for (std::size_t i = 0; i < output_count; ++i) {
        output[i] = 0.0F;
    }

    for (std::size_t tap = 0; tap < filter_count; ++tap) {
        const float coefficient = filter[tap];
        for (std::size_t i = 0; i < output_count; ++i) {
            output[i] += input[i + tap] * coefficient;
        }
    }
}
