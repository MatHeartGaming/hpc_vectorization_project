#include "kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using UnaryKernel = void (*)(const float*, float*, std::size_t);
using BinaryKernel = void (*)(const float*, const float*, float*, std::size_t);
using TriadKernel = void (*)(const float*, const float*, float*, float, std::size_t);
using SaxpyKernel = void (*)(float, const float*, float*, std::size_t);
using ReductionKernel = float (*)(const float*, const float*, std::size_t);
using PolynomialKernel = void (*)(const float*, float*, std::size_t, const float*);
using SmallGemmKernel = void (*)(const float*, const float*, float*, std::size_t);
using LargeGemmKernel = void (*)(const float*, const float*, float*, std::size_t, std::size_t);
using ConvolutionKernel = void (*)(const float*, const float*, float*, std::size_t, std::size_t);
using FftKernel = void (*)(float*, float*, std::size_t, const float*, const float*);
using GatherKernel = float (*)(const float*, const std::uint32_t*, std::size_t);
using ScatterKernel = void (*)(const float*, const std::uint32_t*, float*, std::size_t);
using HistogramKernel = void (*)(const std::uint32_t*, std::uint32_t*, std::size_t, std::size_t);

constexpr double not_applicable = -1.0;

struct BenchmarkResult {
    std::string implementation;
    double seconds{};
    double gb_per_second{};
    double gflops{};
    double checksum{};
};

std::size_t parse_positive_size(const char* value, const char* name) {
    try {
        const auto parsed = std::stoull(value);
        if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<std::size_t>(parsed);
    } catch (...) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
}

bool is_power_of_two(const std::size_t value) {
    return value >= 2 && (value & (value - 1)) == 0;
}

double checksum(const std::vector<float>& values) {
    double sum = 0.0;
    for (const float value : values) sum += static_cast<double>(value);
    return sum;
}

double checksum(const std::vector<std::uint32_t>& values) {
    double sum = 0.0;
    for (const std::uint32_t value : values) sum += static_cast<double>(value);
    return sum;
}

void require_close(
    const std::string& label,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const float absolute_tolerance = 1.0e-5F,
    const float relative_tolerance = 1.0e-4F
) {
    if (expected.size() != actual.size()) throw std::runtime_error(label + ": size mismatch");
    float maximum_error = 0.0F;
    std::size_t maximum_index = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(expected[i] - actual[i]);
        const float allowed = absolute_tolerance + relative_tolerance * std::max(std::abs(expected[i]), std::abs(actual[i]));
        if (error > maximum_error) {
            maximum_error = error;
            maximum_index = i;
        }
        if (error > allowed) {
            throw std::runtime_error(label + " failed validation at index " + std::to_string(i) +
                                     "; expected=" + std::to_string(expected[i]) +
                                     ", actual=" + std::to_string(actual[i]) +
                                     ", error=" + std::to_string(error));
        }
    }
    (void)maximum_error;
    (void)maximum_index;
}

void require_close_scalar(
    const std::string& label,
    const float expected,
    const float actual,
    const float absolute_tolerance = 1.0e-4F,
    const float relative_tolerance = 2.0e-3F
) {
    const float error = std::abs(expected - actual);
    const float allowed = absolute_tolerance + relative_tolerance * std::max(std::abs(expected), std::abs(actual));
    if (error > allowed) {
        throw std::runtime_error(label + " failed validation; expected=" + std::to_string(expected) +
                                 ", actual=" + std::to_string(actual) +
                                 ", error=" + std::to_string(error));
    }
}

template <typename Setup, typename Operation, typename Digest>
BenchmarkResult measure(
    std::string implementation,
    Setup&& setup,
    Operation&& operation,
    Digest&& digest,
    const std::size_t iterations,
    const std::size_t samples,
    const double bytes_per_iteration,
    const double flops_per_iteration
) {
    using Clock = std::chrono::steady_clock;

    setup();
    operation();

    std::vector<double> timings;
    timings.reserve(samples);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        setup();
        const auto start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) operation();
        const auto end = Clock::now();
        timings.push_back(std::chrono::duration<double>(end - start).count());
    }

    std::sort(timings.begin(), timings.end());
    const double seconds = timings[timings.size() / 2];
    const double total_bytes = bytes_per_iteration * static_cast<double>(iterations);
    const double total_flops = flops_per_iteration * static_cast<double>(iterations);

    return {
        .implementation = std::move(implementation),
        .seconds = seconds,
        .gb_per_second = bytes_per_iteration < 0.0 ? not_applicable : total_bytes / seconds / 1.0e9,
        .gflops = flops_per_iteration < 0.0 ? not_applicable : total_flops / seconds / 1.0e9,
        .checksum = digest(),
    };
}

void print_metric(
    const double value,
    const int width,
    const int precision,
    const bool negative_means_not_applicable = false
) {
    if (negative_means_not_applicable && value < 0.0) {
        std::cout << std::setw(width) << "-";
    } else {
        std::cout << std::setw(width) << std::fixed << std::setprecision(precision) << value;
    }
}

void print_triplet(
    const std::string& title,
    const BenchmarkResult& scalar,
    const BenchmarkResult& automatic,
    const BenchmarkResult& native
) {
    std::cout << "\n" << title << '\n'
              << std::left << std::setw(18) << "implementation"
              << std::right << std::setw(12) << "seconds"
              << std::setw(12) << "GB/s"
              << std::setw(12) << "GFLOP/s"
              << std::setw(12) << "speedup"
              << std::setw(18) << "checksum" << '\n';

    for (const BenchmarkResult* result : {&scalar, &automatic, &native}) {
        std::cout << std::left << std::setw(18) << result->implementation << std::right;
        print_metric(result->seconds, 12, 6);
        print_metric(result->gb_per_second, 12, 2, true);
        print_metric(result->gflops, 12, 2, true);
        print_metric(scalar.seconds / result->seconds, 12, 2);
        print_metric(result->checksum, 18, 4);
        std::cout << '\n';
    }
}

void run_unary_vector_kernel(
    const std::string& title,
    const UnaryKernel scalar_kernel,
    const UnaryKernel auto_kernel,
    const UnaryKernel native_kernel,
    const std::vector<float>& input,
    const std::size_t iterations,
    const std::size_t samples,
    const double bytes_per_element,
    const double flops_per_element
) {
    std::vector<float> expected(input.size());
    std::vector<float> actual(input.size());
    scalar_kernel(input.data(), expected.data(), input.size());
    auto_kernel(input.data(), actual.data(), input.size());
    require_close(title + " auto", expected, actual);
    native_kernel(input.data(), actual.data(), input.size());
    require_close(title + " SIMD", expected, actual);

    std::vector<float> output(input.size());
    const auto benchmark = [&](const char* name, const UnaryKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(input.data(), output.data(), input.size()); },
            [&] { return checksum(output); },
            iterations, samples,
            bytes_per_element * static_cast<double>(input.size()),
            flops_per_element * static_cast<double>(input.size()));
    };
    const auto scalar = benchmark("scalar", scalar_kernel);
    const auto automatic = benchmark("auto-vectorized", auto_kernel);
    const auto native = benchmark("manual SIMD", native_kernel);
    print_triplet(title, scalar, automatic, native);
}

void run_binary_vector_kernel(
    const std::string& title,
    const BinaryKernel scalar_kernel,
    const BinaryKernel auto_kernel,
    const BinaryKernel native_kernel,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t iterations,
    const std::size_t samples,
    const double flops_per_element
) {
    std::vector<float> expected(lhs.size());
    std::vector<float> actual(lhs.size());
    scalar_kernel(lhs.data(), rhs.data(), expected.data(), lhs.size());
    auto_kernel(lhs.data(), rhs.data(), actual.data(), lhs.size());
    require_close(title + " auto", expected, actual, 2.0e-5F, 2.0e-4F);
    native_kernel(lhs.data(), rhs.data(), actual.data(), lhs.size());
    require_close(title + " SIMD", expected, actual, 2.0e-5F, 2.0e-4F);

    std::vector<float> output(lhs.size());
    const auto benchmark = [&](const char* name, const BinaryKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(lhs.data(), rhs.data(), output.data(), lhs.size()); },
            [&] { return checksum(output); },
            iterations, samples,
            3.0 * sizeof(float) * static_cast<double>(lhs.size()),
            flops_per_element * static_cast<double>(lhs.size()));
    };
    const auto scalar = benchmark("scalar", scalar_kernel);
    const auto automatic = benchmark("auto-vectorized", auto_kernel);
    const auto native = benchmark("manual SIMD", native_kernel);
    print_triplet(title, scalar, automatic, native);
}

void run_triad(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t iterations,
    const std::size_t samples
) {
    constexpr float scale = 0.375F;
    std::vector<float> expected(lhs.size());
    std::vector<float> actual(lhs.size());
    vector_triad_scalar(lhs.data(), rhs.data(), expected.data(), scale, lhs.size());
    vector_triad_auto(lhs.data(), rhs.data(), actual.data(), scale, lhs.size());
    require_close("STREAM triad auto", expected, actual);
    vector_triad_native_simd(lhs.data(), rhs.data(), actual.data(), scale, lhs.size());
    require_close("STREAM triad SIMD", expected, actual);

    std::vector<float> output(lhs.size());
    const auto benchmark = [&](const char* name, const TriadKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(lhs.data(), rhs.data(), output.data(), scale, lhs.size()); },
            [&] { return checksum(output); },
            iterations, samples,
            3.0 * sizeof(float) * static_cast<double>(lhs.size()),
            2.0 * static_cast<double>(lhs.size()));
    };
    print_triplet("STREAM triad", benchmark("scalar", vector_triad_scalar),
                  benchmark("auto-vectorized", vector_triad_auto),
                  benchmark("manual SIMD", vector_triad_native_simd));
}

void run_saxpy(
    const std::vector<float>& x,
    const std::vector<float>& initial_y,
    const std::size_t iterations,
    const std::size_t samples
) {
    constexpr float scale = 0.25F;
    std::vector<float> expected = initial_y;
    std::vector<float> actual = initial_y;
    saxpy_scalar(scale, x.data(), expected.data(), x.size());
    saxpy_auto(scale, x.data(), actual.data(), x.size());
    require_close("SAXPY auto", expected, actual);
    actual = initial_y;
    saxpy_native_simd(scale, x.data(), actual.data(), x.size());
    require_close("SAXPY SIMD", expected, actual);

    std::vector<float> y = initial_y;
    const auto benchmark = [&](const char* name, const SaxpyKernel kernel) {
        return measure(name,
            [&] { y = initial_y; },
            [&] { kernel(scale, x.data(), y.data(), x.size()); },
            [&] { return checksum(y); },
            iterations, samples,
            3.0 * sizeof(float) * static_cast<double>(x.size()),
            2.0 * static_cast<double>(x.size()));
    };
    print_triplet("SAXPY update", benchmark("scalar", saxpy_scalar),
                  benchmark("auto-vectorized", saxpy_auto),
                  benchmark("manual SIMD", saxpy_native_simd));
}

void run_dot_product(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t iterations,
    const std::size_t samples
) {
    const float expected = dot_product_scalar(lhs.data(), rhs.data(), lhs.size());
    require_close_scalar("dot product auto", expected, dot_product_auto(lhs.data(), rhs.data(), lhs.size()));
    require_close_scalar("dot product SIMD", expected, dot_product_native_simd(lhs.data(), rhs.data(), lhs.size()));

    float result = 0.0F;
    const auto benchmark = [&](const char* name, const ReductionKernel kernel) {
        return measure(name, [] {},
            [&] { result = kernel(lhs.data(), rhs.data(), lhs.size()); },
            [&] { return static_cast<double>(result); },
            iterations, samples,
            2.0 * sizeof(float) * static_cast<double>(lhs.size()),
            2.0 * static_cast<double>(lhs.size()));
    };
    print_triplet("Dot product / reduction", benchmark("scalar", dot_product_scalar),
                  benchmark("auto-vectorized", dot_product_auto),
                  benchmark("manual SIMD", dot_product_native_simd));
}

void run_polynomial(
    const std::vector<float>& input,
    const std::size_t iterations,
    const std::size_t samples
) {
    const float coefficients[5] = {0.75F, -0.25F, 0.125F, 0.0625F, -0.03125F};
    std::vector<float> expected(input.size());
    std::vector<float> actual(input.size());
    polynomial_scalar(input.data(), expected.data(), input.size(), coefficients);
    polynomial_auto(input.data(), actual.data(), input.size(), coefficients);
    require_close("polynomial auto", expected, actual, 2.0e-5F, 3.0e-4F);
    polynomial_native_simd(input.data(), actual.data(), input.size(), coefficients);
    require_close("polynomial SIMD", expected, actual, 2.0e-5F, 3.0e-4F);

    std::vector<float> output(input.size());
    const auto benchmark = [&](const char* name, const PolynomialKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(input.data(), output.data(), input.size(), coefficients); },
            [&] { return checksum(output); },
            iterations, samples,
            2.0 * sizeof(float) * static_cast<double>(input.size()),
            8.0 * static_cast<double>(input.size()));
    };
    print_triplet("Horner polynomial (FMA-heavy)", benchmark("scalar", polynomial_scalar),
                  benchmark("auto-vectorized", polynomial_auto),
                  benchmark("manual SIMD", polynomial_native_simd));
}

std::vector<float> make_matrix(const std::size_t dimension, const std::size_t seed) {
    std::vector<float> matrix(dimension * dimension);
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        const int value = static_cast<int>((i * (seed * 17 + 3) + seed * 11) % 97) - 48;
        matrix[i] = static_cast<float>(value) * 0.01F;
    }
    return matrix;
}

std::vector<float> transpose(const std::vector<float>& matrix, const std::size_t dimension) {
    std::vector<float> result(matrix.size());
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            result[column * dimension + row] = matrix[row * dimension + column];
        }
    }
    return result;
}

void run_small_gemm(const std::size_t dimension, const std::size_t samples) {
    const auto lhs = make_matrix(dimension, 1);
    const auto rhs = make_matrix(dimension, 2);
    const auto rhs_transposed = transpose(rhs, dimension);
    std::vector<float> expected(dimension * dimension);
    std::vector<float> actual(dimension * dimension);
    matrix_multiply_scalar(lhs.data(), rhs_transposed.data(), expected.data(), dimension);
    matrix_multiply_auto(lhs.data(), rhs_transposed.data(), actual.data(), dimension);
    require_close("small GEMM auto", expected, actual, 2.0e-4F, 2.0e-3F);
    matrix_multiply_native_simd(lhs.data(), rhs_transposed.data(), actual.data(), dimension);
    require_close("small GEMM SIMD", expected, actual, 2.0e-4F, 2.0e-3F);

    std::vector<float> output(dimension * dimension);
    const std::size_t iterations = dimension <= 64 ? 3 : 1;
    const double flops = 2.0 * static_cast<double>(dimension) * dimension * dimension;
    const auto benchmark = [&](const char* name, const SmallGemmKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(lhs.data(), rhs_transposed.data(), output.data(), dimension); },
            [&] { return checksum(output); },
            iterations, samples, not_applicable, flops);
    };
    print_triplet("Small GEMM " + std::to_string(dimension) + "x" + std::to_string(dimension),
                  benchmark("scalar", matrix_multiply_scalar),
                  benchmark("auto-vectorized", matrix_multiply_auto),
                  benchmark("manual SIMD", matrix_multiply_native_simd));
}

void run_large_gemm(const std::size_t dimension, const std::size_t samples) {
    constexpr std::size_t block_size = 64;
    const auto lhs = make_matrix(dimension, 3);
    const auto rhs = make_matrix(dimension, 4);
    std::vector<float> expected(dimension * dimension);
    std::vector<float> actual(dimension * dimension);
    matrix_multiply_blocked_scalar(lhs.data(), rhs.data(), expected.data(), dimension, block_size);
    matrix_multiply_blocked_auto(lhs.data(), rhs.data(), actual.data(), dimension, block_size);
    require_close("large GEMM auto", expected, actual, 4.0e-4F, 3.0e-3F);
    matrix_multiply_blocked_native_simd(lhs.data(), rhs.data(), actual.data(), dimension, block_size);
    require_close("large GEMM SIMD", expected, actual, 4.0e-4F, 3.0e-3F);

    std::vector<float> output(dimension * dimension);
    const double flops = 2.0 * static_cast<double>(dimension) * dimension * dimension;
    const auto benchmark = [&](const char* name, const LargeGemmKernel kernel) {
        return measure(name, [] {},
            [&] { kernel(lhs.data(), rhs.data(), output.data(), dimension, block_size); },
            [&] { return checksum(output); },
            1, samples, not_applicable, flops);
    };
    print_triplet("Large blocked GEMM " + std::to_string(dimension) + "x" + std::to_string(dimension),
                  benchmark("scalar", matrix_multiply_blocked_scalar),
                  benchmark("auto-vectorized", matrix_multiply_blocked_auto),
                  benchmark("manual SIMD", matrix_multiply_blocked_native_simd));
}

void run_convolution(
    const std::vector<float>& input,
    const std::size_t iterations,
    const std::size_t samples
) {
    const std::vector<float> filter = {0.05F, 0.1F, 0.2F, 0.3F, 0.2F, 0.1F, 0.05F};
    const std::size_t output_count = input.size() - filter.size() + 1;
    std::vector<float> expected(output_count);
    std::vector<float> actual(output_count);
    convolution_1d_scalar(input.data(), filter.data(), expected.data(), input.size(), filter.size());
    convolution_1d_auto(input.data(), filter.data(), actual.data(), input.size(), filter.size());
    require_close("convolution auto", expected, actual, 2.0e-5F, 3.0e-4F);
    convolution_1d_native_simd(input.data(), filter.data(), actual.data(), input.size(), filter.size());
    require_close("convolution SIMD", expected, actual, 2.0e-5F, 3.0e-4F);

    std::vector<float> output(output_count);
    const double flops = 2.0 * static_cast<double>(output_count) * filter.size();
    const auto benchmark = [&](const char* name, const ConvolutionKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(input.data(), filter.data(), output.data(), input.size(), filter.size()); },
            [&] { return checksum(output); },
            iterations, samples, not_applicable, flops);
    };
    print_triplet("1D convolution (7 taps)", benchmark("scalar", convolution_1d_scalar),
                  benchmark("auto-vectorized", convolution_1d_auto),
                  benchmark("manual SIMD", convolution_1d_native_simd));
}

void run_fft(const std::size_t count, const std::size_t samples) {
    std::vector<float> initial_real(count);
    std::vector<float> initial_imaginary(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(count);
        initial_real[i] = static_cast<float>(0.7 * std::sin(3.0 * angle) + 0.2 * std::cos(17.0 * angle));
        initial_imaginary[i] = static_cast<float>(0.1 * std::sin(5.0 * angle));
    }
    std::vector<float> twiddle_real(count / 2);
    std::vector<float> twiddle_imaginary(count / 2);
    for (std::size_t k = 0; k < count / 2; ++k) {
        const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(count);
        twiddle_real[k] = static_cast<float>(std::cos(angle));
        twiddle_imaginary[k] = static_cast<float>(std::sin(angle));
    }

    std::vector<float> expected_real = initial_real;
    std::vector<float> expected_imaginary = initial_imaginary;
    fft_radix2_scalar(expected_real.data(), expected_imaginary.data(), count, twiddle_real.data(), twiddle_imaginary.data());
    std::vector<float> actual_real = initial_real;
    std::vector<float> actual_imaginary = initial_imaginary;
    fft_radix2_auto(actual_real.data(), actual_imaginary.data(), count, twiddle_real.data(), twiddle_imaginary.data());
    require_close("FFT auto real", expected_real, actual_real, 2.0e-3F, 4.0e-3F);
    require_close("FFT auto imaginary", expected_imaginary, actual_imaginary, 2.0e-3F, 4.0e-3F);
    actual_real = initial_real;
    actual_imaginary = initial_imaginary;
    fft_radix2_native_simd(actual_real.data(), actual_imaginary.data(), count, twiddle_real.data(), twiddle_imaginary.data());
    require_close("FFT SIMD real", expected_real, actual_real, 2.0e-3F, 4.0e-3F);
    require_close("FFT SIMD imaginary", expected_imaginary, actual_imaginary, 2.0e-3F, 4.0e-3F);

    std::vector<float> real = initial_real;
    std::vector<float> imaginary = initial_imaginary;
    const double stages = std::log2(static_cast<double>(count));
    const double approximate_flops = 5.0 * static_cast<double>(count) * stages;
    const auto benchmark = [&](const char* name, const FftKernel kernel) {
        return measure(name,
            [&] { real = initial_real; imaginary = initial_imaginary; },
            [&] { kernel(real.data(), imaginary.data(), count, twiddle_real.data(), twiddle_imaginary.data()); },
            [&] { return checksum(real) + checksum(imaginary); },
            1, samples, not_applicable, approximate_flops);
    };
    print_triplet("Radix-2 FFT (N=" + std::to_string(count) + ")",
                  benchmark("scalar", fft_radix2_scalar),
                  benchmark("auto-vectorized", fft_radix2_auto),
                  benchmark("manual SIMD", fft_radix2_native_simd));
}

void run_gather(
    const std::vector<float>& values,
    const std::vector<std::uint32_t>& indices,
    const std::size_t iterations,
    const std::size_t samples
) {
    const float expected = gather_sum_scalar(values.data(), indices.data(), indices.size());
    require_close_scalar("gather auto", expected, gather_sum_auto(values.data(), indices.data(), indices.size()), 2.0e-3F, 3.0e-3F);
    require_close_scalar("gather SIMD", expected, gather_sum_native_simd(values.data(), indices.data(), indices.size()), 2.0e-3F, 3.0e-3F);

    float result = 0.0F;
    const auto benchmark = [&](const char* name, const GatherKernel kernel) {
        return measure(name, [] {},
            [&] { result = kernel(values.data(), indices.data(), indices.size()); },
            [&] { return static_cast<double>(result); },
            iterations, samples,
            (sizeof(float) + sizeof(std::uint32_t)) * static_cast<double>(indices.size()),
            static_cast<double>(indices.size()));
    };
    print_triplet("Gather / indexed sum", benchmark("scalar", gather_sum_scalar),
                  benchmark("auto-vectorized", gather_sum_auto),
                  benchmark("manual SIMD", gather_sum_native_simd));
}

void run_scatter(
    const std::vector<float>& values,
    const std::vector<std::uint32_t>& indices,
    const std::size_t output_count,
    const std::size_t iterations,
    const std::size_t samples
) {
    std::vector<float> expected(output_count, 0.0F);
    std::vector<float> actual(output_count, 0.0F);
    scatter_add_scalar(values.data(), indices.data(), expected.data(), values.size());
    scatter_add_auto(values.data(), indices.data(), actual.data(), values.size());
    require_close("scatter auto", expected, actual, 1.0e-5F, 1.0e-5F);
    std::fill(actual.begin(), actual.end(), 0.0F);
    scatter_add_native_simd(values.data(), indices.data(), actual.data(), values.size());
    require_close("scatter SIMD", expected, actual, 1.0e-5F, 1.0e-5F);

    std::vector<float> output(output_count);
    const auto benchmark = [&](const char* name, const ScatterKernel kernel) {
        return measure(name,
            [&] { std::fill(output.begin(), output.end(), 0.0F); },
            [&] { kernel(values.data(), indices.data(), output.data(), values.size()); },
            [&] { return checksum(output); },
            iterations, samples,
            3.0 * sizeof(float) * static_cast<double>(values.size()),
            static_cast<double>(values.size()));
    };
    print_triplet("Scatter-add with collisions", benchmark("scalar", scatter_add_scalar),
                  benchmark("auto-vectorized", scatter_add_auto),
                  benchmark("manual/hybrid", scatter_add_native_simd));
}

void run_histogram(
    const std::vector<std::uint32_t>& values,
    const std::size_t bin_count,
    const std::size_t iterations,
    const std::size_t samples
) {
    std::vector<std::uint32_t> expected(bin_count, 0);
    std::vector<std::uint32_t> actual(bin_count, 0);
    histogram_scalar(values.data(), expected.data(), values.size(), bin_count);
    histogram_auto(values.data(), actual.data(), values.size(), bin_count);
    if (expected != actual) throw std::runtime_error("histogram auto failed validation");
    std::fill(actual.begin(), actual.end(), 0);
    histogram_native_simd(values.data(), actual.data(), values.size(), bin_count);
    if (expected != actual) throw std::runtime_error("histogram SIMD failed validation");

    std::vector<std::uint32_t> bins(bin_count);
    const auto benchmark = [&](const char* name, const HistogramKernel kernel) {
        return measure(name,
            [&] { std::fill(bins.begin(), bins.end(), 0); },
            [&] { kernel(values.data(), bins.data(), values.size(), bin_count); },
            [&] { return checksum(bins); },
            iterations, samples,
            3.0 * sizeof(std::uint32_t) * static_cast<double>(values.size()),
            not_applicable);
    };
    print_triplet("Histogram-like update", benchmark("scalar", histogram_scalar),
                  benchmark("auto-vectorized", histogram_auto),
                  benchmark("manual/hybrid", histogram_native_simd));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t vector_count = argc > 1 ? parse_positive_size(argv[1], "vector_count") : (std::size_t{1} << 20);
        const std::size_t vector_iterations = argc > 2 ? parse_positive_size(argv[2], "vector_iterations") : 10;
        const std::size_t samples = argc > 3 ? parse_positive_size(argv[3], "samples") : 5;
        const std::size_t small_gemm_dimension = argc > 4 ? parse_positive_size(argv[4], "small_gemm_dimension") : 64;
        const std::size_t large_gemm_dimension = argc > 5 ? parse_positive_size(argv[5], "large_gemm_dimension") : 256;
        const std::size_t fft_size = argc > 6 ? parse_positive_size(argv[6], "fft_size") : (std::size_t{1} << 15);

        if (samples % 2 == 0) throw std::invalid_argument("samples must be odd");
        if (!is_power_of_two(fft_size)) throw std::invalid_argument("fft_size must be a power of two");
        if (vector_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("vector_count must fit in signed 32-bit indices for AVX2 gather");
        }

        std::vector<float> lhs(vector_count);
        std::vector<float> rhs(vector_count);
        for (std::size_t i = 0; i < vector_count; ++i) {
            lhs[i] = static_cast<float>(static_cast<int>(i % 1024) - 512) * 0.001F;
            rhs[i] = 1.0F + static_cast<float>(static_cast<int>((i * 7) % 509) - 254) * 0.001F;
        }

        std::vector<std::uint32_t> gather_indices(vector_count);
        const std::size_t scatter_bin_count = std::max<std::size_t>(256, vector_count / 16);
        std::vector<std::uint32_t> scatter_indices(vector_count);
        std::vector<std::uint32_t> histogram_values(vector_count);
        for (std::size_t i = 0; i < vector_count; ++i) {
            const std::uint64_t mixed = static_cast<std::uint64_t>(i) * 2654435761ULL + 1013904223ULL;
            gather_indices[i] = static_cast<std::uint32_t>(mixed % vector_count);
            scatter_indices[i] = static_cast<std::uint32_t>(mixed % scatter_bin_count);
            histogram_values[i] = static_cast<std::uint32_t>((mixed >> 8) % 256);
        }

        std::cout << "SIMD benchmark suite\n"
                  << "Native backend: " << native_simd_name() << '\n'
                  << "Vector elements: " << vector_count
                  << ", vector iterations: " << vector_iterations
                  << ", samples: " << samples << " (median)\n"
                  << "Small GEMM: " << small_gemm_dimension << "x" << small_gemm_dimension
                  << ", large GEMM: " << large_gemm_dimension << "x" << large_gemm_dimension
                  << ", FFT size: " << fft_size << "\n";

        run_unary_vector_kernel("STREAM copy", vector_copy_scalar, vector_copy_auto,
                                vector_copy_native_simd, lhs, vector_iterations, samples,
                                2.0 * sizeof(float), 0.0);
        run_binary_vector_kernel("Vector addition", vector_add_scalar, vector_add_auto,
                                 vector_add_native_simd, lhs, rhs, vector_iterations, samples, 1.0);
        run_binary_vector_kernel("Vector subtraction", vector_subtract_scalar, vector_subtract_auto,
                                 vector_subtract_native_simd, lhs, rhs, vector_iterations, samples, 1.0);
        run_binary_vector_kernel("Vector multiplication", vector_multiply_scalar, vector_multiply_auto,
                                 vector_multiply_native_simd, lhs, rhs, vector_iterations, samples, 1.0);
        run_binary_vector_kernel("Vector division", vector_divide_scalar, vector_divide_auto,
                                 vector_divide_native_simd, lhs, rhs, vector_iterations, samples, 1.0);
        run_triad(lhs, rhs, vector_iterations, samples);
        run_saxpy(lhs, rhs, vector_iterations, samples);
        run_dot_product(lhs, rhs, vector_iterations, samples);
        run_polynomial(lhs, vector_iterations, samples);
        run_small_gemm(small_gemm_dimension, samples);
        run_large_gemm(large_gemm_dimension, samples);
        run_convolution(lhs, std::max<std::size_t>(1, vector_iterations / 2), samples);
        run_fft(fft_size, samples);
        run_gather(lhs, gather_indices, vector_iterations, samples);
        run_scatter(lhs, scatter_indices, scatter_bin_count,
                    std::max<std::size_t>(1, vector_iterations / 2), samples);
        run_histogram(histogram_values, 256,
                      std::max<std::size_t>(1, vector_iterations / 2), samples);

        std::cout << "\nAll validation checks passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'
                  << "Usage: simd_benchmark [vector_count] [vector_iterations] [odd_samples] "
                     "[small_gemm_dimension] [large_gemm_dimension] [fft_power_of_two]\n";
        return EXIT_FAILURE;
    }
}
