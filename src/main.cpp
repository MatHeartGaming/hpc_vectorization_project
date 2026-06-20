#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct BenchmarkResult {
    std::string implementation;
    double seconds{};
    double gb_per_second{};
    bool reports_bandwidth{};
    double gflops{};
    double checksum{};
};

struct Configuration {
    std::size_t element_count{std::size_t{1} << 20};
    std::size_t vector_iterations{20};
    std::size_t samples{5};
    std::size_t matrix_dimension{192};
    std::size_t matrix_iterations{2};
};

std::size_t parse_positive_size(const char* value, const char* argument_name) {
    try {
        const auto parsed = std::stoull(value);
        if (parsed == 0) {
            throw std::invalid_argument("zero");
        }
        return static_cast<std::size_t>(parsed);
    } catch (...) {
        throw std::invalid_argument(std::string(argument_name) + " must be a positive integer");
    }
}

double checksum(const float* values, const std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(values[i]);
    }
    return sum;
}

void verify_values(
    const std::string& name,
    const float* actual,
    const float* expected,
    const std::size_t count,
    const float absolute_tolerance,
    const float relative_tolerance
) {
    float maximum_absolute_error = 0.0F;
    float maximum_relative_error = 0.0F;

    for (std::size_t i = 0; i < count; ++i) {
        const float difference = std::abs(actual[i] - expected[i]);
        const float relative_error = difference / std::max(std::abs(expected[i]), 1.0e-12F);
        maximum_absolute_error = std::max(maximum_absolute_error, difference);
        maximum_relative_error = std::max(maximum_relative_error, relative_error);

        if (difference > absolute_tolerance + relative_tolerance * std::abs(expected[i])) {
            throw std::runtime_error(
                name + " failed validation at index " + std::to_string(i) +
                "; expected=" + std::to_string(expected[i]) +
                ", actual=" + std::to_string(actual[i]) +
                ", max abs error=" + std::to_string(maximum_absolute_error) +
                ", max rel error=" + std::to_string(maximum_relative_error)
            );
        }
    }
}

void verify_scalar_value(
    const std::string& name,
    const float actual,
    const float expected,
    const float absolute_tolerance,
    const float relative_tolerance
) {
    const float difference = std::abs(actual - expected);
    if (difference > absolute_tolerance + relative_tolerance * std::abs(expected)) {
        throw std::runtime_error(
            name + " failed validation; expected=" + std::to_string(expected) +
            ", actual=" + std::to_string(actual) +
            ", difference=" + std::to_string(difference)
        );
    }
}

template <typename Prepare, typename Run, typename Consume>
BenchmarkResult measure(
    std::string implementation,
    const std::size_t iterations,
    const std::size_t samples,
    const double floating_point_operations_per_call,
    const double bytes_per_call,
    Prepare&& prepare,
    Run&& run,
    Consume&& consume
) {
    using Clock = std::chrono::steady_clock;

    // Warm-up calls are intentionally outside measured samples.
    for (int warmup = 0; warmup < 2; ++warmup) {
        prepare();
        run();
    }

    std::vector<double> timings;
    timings.reserve(samples);
    double final_checksum = 0.0;

    for (std::size_t sample = 0; sample < samples; ++sample) {
        prepare();

        const auto start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            run();
        }
        const auto end = Clock::now();

        timings.push_back(std::chrono::duration<double>(end - start).count());
        final_checksum = consume();
    }

    std::sort(timings.begin(), timings.end());
    const double median_seconds = timings[timings.size() / 2];
    const double total_calls = static_cast<double>(iterations);

    BenchmarkResult result;
    result.implementation = std::move(implementation);
    result.seconds = median_seconds;
    result.gflops = floating_point_operations_per_call * total_calls / median_seconds / 1.0e9;
    result.checksum = final_checksum;

    if (bytes_per_call > 0.0) {
        result.gb_per_second = bytes_per_call * total_calls / median_seconds / 1.0e9;
        result.reports_bandwidth = true;
    }

    return result;
}

void print_suite(
    const std::string& title,
    const std::string& formula,
    const std::vector<BenchmarkResult>& results
) {
    const double scalar_seconds = results.front().seconds;

    std::cout << "\n=== " << title << " ===\n"
              << formula << '\n'
              << std::left << std::setw(20) << "implementation"
              << std::right << std::setw(13) << "seconds"
              << std::setw(13) << "GB/s*"
              << std::setw(13) << "GFLOP/s"
              << std::setw(12) << "speedup"
              << std::setw(20) << "checksum" << '\n';

    for (const BenchmarkResult& result : results) {
        std::cout << std::left << std::setw(20) << result.implementation
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(13) << result.seconds;

        if (!result.reports_bandwidth) {
            std::cout << std::setw(13) << "-";
        } else {
            std::cout << std::setprecision(2) << std::setw(13) << result.gb_per_second;
        }

        std::cout << std::setprecision(2) << std::setw(13) << result.gflops
                  << std::setw(12) << scalar_seconds / result.seconds
                  << std::setprecision(5) << std::setw(20) << result.checksum
                  << '\n';
    }
}

BenchmarkResult benchmark_binary(
    const std::string& implementation,
    const BinaryKernel kernel,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& output,
    const std::size_t iterations,
    const std::size_t samples
) {
    const double operations = static_cast<double>(lhs.size());
    const double bytes = 3.0 * sizeof(float) * static_cast<double>(lhs.size());

    return measure(
        implementation,
        iterations,
        samples,
        operations,
        bytes,
        [] {},
        [&] { kernel(lhs.data(), rhs.data(), output.data(), lhs.size()); },
        [&] { return checksum(output.data(), output.size()); }
    );
}

void run_binary_suite(
    const std::string& title,
    const std::string& formula,
    const BinaryKernel scalar_kernel,
    const BinaryKernel auto_kernel,
    const BinaryKernel simd_kernel,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t iterations,
    const std::size_t samples
) {
    std::vector<float> reference(lhs.size());
    std::vector<float> actual(lhs.size());

    scalar_kernel(lhs.data(), rhs.data(), reference.data(), lhs.size());
    auto_kernel(lhs.data(), rhs.data(), actual.data(), lhs.size());
    verify_values(title + " auto", actual.data(), reference.data(), actual.size(), 1.0e-6F, 1.0e-6F);
    simd_kernel(lhs.data(), rhs.data(), actual.data(), lhs.size());
    verify_values(title + " SIMD", actual.data(), reference.data(), actual.size(), 1.0e-6F, 1.0e-6F);

    std::vector<BenchmarkResult> results;
    results.reserve(3);
    results.push_back(benchmark_binary("scalar", scalar_kernel, lhs, rhs, actual, iterations, samples));
    results.push_back(benchmark_binary("auto-vectorized", auto_kernel, lhs, rhs, actual, iterations, samples));
    results.push_back(benchmark_binary("manual SIMD", simd_kernel, lhs, rhs, actual, iterations, samples));
    print_suite(title, formula, results);
}

BenchmarkResult benchmark_saxpy(
    const std::string& implementation,
    const SaxpyKernel kernel,
    const float scale,
    const std::vector<float>& x,
    const std::vector<float>& initial_y,
    std::vector<float>& y,
    const std::size_t iterations,
    const std::size_t samples
) {
    const double operations = 2.0 * static_cast<double>(x.size());
    const double bytes = 3.0 * sizeof(float) * static_cast<double>(x.size());

    return measure(
        implementation,
        iterations,
        samples,
        operations,
        bytes,
        [&] { y = initial_y; },
        [&] { kernel(scale, x.data(), y.data(), y.size()); },
        [&] { return checksum(y.data(), y.size()); }
    );
}

void run_saxpy_suite(
    const std::vector<float>& x,
    const std::vector<float>& initial_y,
    const std::size_t iterations,
    const std::size_t samples
) {
    constexpr float scale = 0.25F;
    std::vector<float> reference = initial_y;
    std::vector<float> actual = initial_y;

    saxpy_scalar(scale, x.data(), reference.data(), reference.size());
    saxpy_auto(scale, x.data(), actual.data(), actual.size());
    verify_values("SAXPY auto", actual.data(), reference.data(), actual.size(), 1.0e-6F, 1.0e-6F);
    actual = initial_y;
    saxpy_native_simd(scale, x.data(), actual.data(), actual.size());
    verify_values("SAXPY SIMD", actual.data(), reference.data(), actual.size(), 1.0e-6F, 1.0e-6F);

    std::vector<BenchmarkResult> results;
    results.reserve(3);
    results.push_back(benchmark_saxpy("scalar", saxpy_scalar, scale, x, initial_y, actual, iterations, samples));
    results.push_back(benchmark_saxpy("auto-vectorized", saxpy_auto, scale, x, initial_y, actual, iterations, samples));
    results.push_back(benchmark_saxpy("manual SIMD", saxpy_native_simd, scale, x, initial_y, actual, iterations, samples));
    print_suite("SAXPY / scaled vector addition", "y[i] = scale * x[i] + y[i]", results);
}

BenchmarkResult benchmark_dot(
    const std::string& implementation,
    const DotKernel kernel,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t iterations,
    const std::size_t samples
) {
    float accumulated_result = 0.0F;
    const double operations = 2.0 * static_cast<double>(lhs.size());
    const double bytes = 2.0 * sizeof(float) * static_cast<double>(lhs.size());

    return measure(
        implementation,
        iterations,
        samples,
        operations,
        bytes,
        [&] { accumulated_result = 0.0F; },
        [&] { accumulated_result += kernel(lhs.data(), rhs.data(), lhs.size()); },
        [&] { return static_cast<double>(accumulated_result); }
    );
}

void run_dot_suite(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t iterations,
    const std::size_t samples
) {
    const float reference = dot_product_scalar(lhs.data(), rhs.data(), lhs.size());
    verify_scalar_value(
        "Dot product auto",
        dot_product_auto(lhs.data(), rhs.data(), lhs.size()),
        reference,
        1.0e-2F,
        5.0e-3F
    );
    verify_scalar_value(
        "Dot product SIMD",
        dot_product_native_simd(lhs.data(), rhs.data(), lhs.size()),
        reference,
        1.0e-2F,
        5.0e-3F
    );

    std::vector<BenchmarkResult> results;
    results.reserve(3);
    results.push_back(benchmark_dot("scalar", dot_product_scalar, lhs, rhs, iterations, samples));
    results.push_back(benchmark_dot("auto-vectorized", dot_product_auto, lhs, rhs, iterations, samples));
    results.push_back(benchmark_dot("manual SIMD", dot_product_native_simd, lhs, rhs, iterations, samples));
    print_suite("Dot product / reduction", "sum(lhs[i] * rhs[i])", results);
}

BenchmarkResult benchmark_polynomial(
    const std::string& implementation,
    const PolynomialKernel kernel,
    const std::vector<float>& input,
    std::vector<float>& output,
    const std::array<float, 5>& coefficients,
    const std::size_t iterations,
    const std::size_t samples
) {
    const double operations = 8.0 * static_cast<double>(input.size());
    const double bytes = 2.0 * sizeof(float) * static_cast<double>(input.size());

    return measure(
        implementation,
        iterations,
        samples,
        operations,
        bytes,
        [] {},
        [&] { kernel(input.data(), output.data(), input.size(), coefficients.data()); },
        [&] { return checksum(output.data(), output.size()); }
    );
}

void run_polynomial_suite(
    const std::vector<float>& input,
    const std::size_t iterations,
    const std::size_t samples
) {
    constexpr std::array<float, 5> coefficients{0.75F, -0.5F, 0.25F, -0.125F, 0.0625F};
    std::vector<float> reference(input.size());
    std::vector<float> actual(input.size());

    polynomial_scalar(input.data(), reference.data(), input.size(), coefficients.data());
    polynomial_auto(input.data(), actual.data(), input.size(), coefficients.data());
    verify_values("Polynomial auto", actual.data(), reference.data(), actual.size(), 1.0e-5F, 1.0e-5F);
    polynomial_native_simd(input.data(), actual.data(), input.size(), coefficients.data());
    verify_values("Polynomial SIMD", actual.data(), reference.data(), actual.size(), 1.0e-5F, 1.0e-5F);

    std::vector<BenchmarkResult> results;
    results.reserve(3);
    results.push_back(benchmark_polynomial("scalar", polynomial_scalar, input, actual, coefficients, iterations, samples));
    results.push_back(benchmark_polynomial("auto-vectorized", polynomial_auto, input, actual, coefficients, iterations, samples));
    results.push_back(benchmark_polynomial("manual SIMD", polynomial_native_simd, input, actual, coefficients, iterations, samples));
    print_suite(
        "FMA-heavy polynomial",
        "((((c4 * x + c3) * x + c2) * x + c1) * x + c0)",
        results
    );
}

BenchmarkResult benchmark_matrix(
    const std::string& implementation,
    const MatrixMultiplyKernel kernel,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs_transposed,
    std::vector<float>& output,
    const std::size_t dimension,
    const std::size_t iterations,
    const std::size_t samples
) {
    const double n = static_cast<double>(dimension);
    const double operations = 2.0 * n * n * n;

    return measure(
        implementation,
        iterations,
        samples,
        operations,
        0.0,
        [] {},
        [&] { kernel(lhs.data(), rhs_transposed.data(), output.data(), dimension); },
        [&] { return checksum(output.data(), output.size()); }
    );
}

void run_matrix_suite(
    const std::size_t dimension,
    const std::size_t iterations,
    const std::size_t samples
) {
    const std::size_t matrix_elements = dimension * dimension;
    std::vector<float> lhs(matrix_elements);
    std::vector<float> rhs(matrix_elements);
    std::vector<float> rhs_transposed(matrix_elements);
    std::vector<float> reference(matrix_elements);
    std::vector<float> actual(matrix_elements);

    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            const std::size_t index = row * dimension + column;
            lhs[index] = static_cast<float>(static_cast<int>((row * 13 + column * 7) % 97) - 48) * 0.002F;
            rhs[index] = static_cast<float>(static_cast<int>((row * 5 + column * 11) % 89) - 44) * 0.0025F;
            rhs_transposed[column * dimension + row] = rhs[index];
        }
    }

    matrix_multiply_scalar(lhs.data(), rhs_transposed.data(), reference.data(), dimension);
    matrix_multiply_auto(lhs.data(), rhs_transposed.data(), actual.data(), dimension);
    verify_values("Matrix multiply auto", actual.data(), reference.data(), actual.size(), 1.0e-4F, 5.0e-4F);
    matrix_multiply_native_simd(lhs.data(), rhs_transposed.data(), actual.data(), dimension);
    verify_values("Matrix multiply SIMD", actual.data(), reference.data(), actual.size(), 1.0e-4F, 5.0e-4F);

    std::vector<BenchmarkResult> results;
    results.reserve(3);
    results.push_back(benchmark_matrix(
        "scalar", matrix_multiply_scalar, lhs, rhs_transposed, actual,
        dimension, iterations, samples));
    results.push_back(benchmark_matrix(
        "auto-vectorized", matrix_multiply_auto, lhs, rhs_transposed, actual,
        dimension, iterations, samples));
    results.push_back(benchmark_matrix(
        "manual SIMD", matrix_multiply_native_simd, lhs, rhs_transposed, actual,
        dimension, iterations, samples));
    print_suite(
        "Square matrix multiplication",
        "C = A * B, with B transposed once before timing; dimension = " + std::to_string(dimension),
        results
    );
}

BenchmarkResult benchmark_convolution(
    const std::string& implementation,
    const ConvolutionKernel kernel,
    const std::vector<float>& input,
    const std::vector<float>& filter,
    std::vector<float>& output,
    const std::size_t iterations,
    const std::size_t samples
) {
    const double output_count = static_cast<double>(output.size());
    const double operations = 2.0 * output_count * static_cast<double>(filter.size());

    return measure(
        implementation,
        iterations,
        samples,
        operations,
        0.0,
        [] {},
        [&] {
            kernel(
                input.data(), filter.data(), output.data(), input.size(), filter.size());
        },
        [&] { return checksum(output.data(), output.size()); }
    );
}

void run_convolution_suite(
    const std::vector<float>& input,
    const std::size_t iterations,
    const std::size_t samples
) {
    constexpr std::array<float, 7> filter_values{
        -0.03125F, 0.125F, 0.28125F, 0.25F, 0.28125F, 0.125F, -0.03125F
    };
    const std::vector<float> filter(filter_values.begin(), filter_values.end());
    const std::size_t output_count = input.size() - filter.size() + 1;
    std::vector<float> reference(output_count);
    std::vector<float> actual(output_count);

    convolution_1d_scalar(
        input.data(), filter.data(), reference.data(), input.size(), filter.size());
    convolution_1d_auto(
        input.data(), filter.data(), actual.data(), input.size(), filter.size());
    verify_values("Convolution auto", actual.data(), reference.data(), actual.size(), 1.0e-5F, 1.0e-4F);
    convolution_1d_native_simd(
        input.data(), filter.data(), actual.data(), input.size(), filter.size());
    verify_values("Convolution SIMD", actual.data(), reference.data(), actual.size(), 1.0e-5F, 1.0e-4F);

    std::vector<BenchmarkResult> results;
    results.reserve(3);
    results.push_back(benchmark_convolution(
        "scalar", convolution_1d_scalar, input, filter, actual, iterations, samples));
    results.push_back(benchmark_convolution(
        "auto-vectorized", convolution_1d_auto, input, filter, actual, iterations, samples));
    results.push_back(benchmark_convolution(
        "manual SIMD", convolution_1d_native_simd, input, filter, actual, iterations, samples));
    print_suite(
        "Valid 1D convolution",
        "7-tap finite impulse response filter; output count = " + std::to_string(output_count),
        results
    );
}

Configuration parse_configuration(const int argc, char** argv) {
    Configuration configuration;

    if (argc > 1) {
        configuration.element_count = parse_positive_size(argv[1], "element_count");
    }
    if (argc > 2) {
        configuration.vector_iterations = parse_positive_size(argv[2], "vector_iterations");
    }
    if (argc > 3) {
        configuration.samples = parse_positive_size(argv[3], "samples");
    }
    if (argc > 4) {
        configuration.matrix_dimension = parse_positive_size(argv[4], "matrix_dimension");
    }
    if (argc > 5) {
        configuration.matrix_iterations = parse_positive_size(argv[5], "matrix_iterations");
    }
    if (argc > 6) {
        throw std::invalid_argument("too many arguments");
    }
    if (configuration.samples % 2 == 0) {
        throw std::invalid_argument("samples must be odd so the median is unambiguous");
    }
    if (configuration.element_count < 7) {
        throw std::invalid_argument("element_count must be at least 7 for the convolution kernel");
    }
    if (configuration.matrix_dimension >
        std::numeric_limits<std::size_t>::max() / configuration.matrix_dimension) {
        throw std::invalid_argument("matrix_dimension is too large");
    }

    return configuration;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        const Configuration configuration = parse_configuration(argc, argv);

        std::vector<float> lhs(configuration.element_count);
        std::vector<float> rhs(configuration.element_count);

        for (std::size_t i = 0; i < configuration.element_count; ++i) {
            lhs[i] = 0.001F + static_cast<float>(i % 1024) * 0.0009F;
            rhs[i] = 0.5F + static_cast<float>((i * 17) % 1024) * 0.0007F;
        }

        std::cout << "SIMD benchmark suite\n"
                  << "Native SIMD backend: " << native_simd_name() << '\n'
                  << "Vector elements: " << configuration.element_count << '\n'
                  << "Vector/convolution iterations per sample: "
                  << configuration.vector_iterations << '\n'
                  << "Matrix dimension: " << configuration.matrix_dimension << '\n'
                  << "Matrix iterations per sample: " << configuration.matrix_iterations << '\n'
                  << "Samples: " << configuration.samples << " (median reported)\n"
                  << "GB/s* is shown only where a meaningful minimum streaming byte count exists.\n";

        run_binary_suite(
            "Vector addition", "output[i] = lhs[i] + rhs[i]",
            vector_add_scalar, vector_add_auto, vector_add_native_simd,
            lhs, rhs, configuration.vector_iterations, configuration.samples);

        run_binary_suite(
            "Vector subtraction", "output[i] = lhs[i] - rhs[i]",
            vector_subtract_scalar, vector_subtract_auto, vector_subtract_native_simd,
            lhs, rhs, configuration.vector_iterations, configuration.samples);

        run_binary_suite(
            "Vector multiplication", "output[i] = lhs[i] * rhs[i]",
            vector_multiply_scalar, vector_multiply_auto, vector_multiply_native_simd,
            lhs, rhs, configuration.vector_iterations, configuration.samples);

        run_binary_suite(
            "Vector division", "output[i] = lhs[i] / rhs[i]",
            vector_divide_scalar, vector_divide_auto, vector_divide_native_simd,
            lhs, rhs, configuration.vector_iterations, configuration.samples);

        run_saxpy_suite(
            lhs, rhs, configuration.vector_iterations, configuration.samples);

        run_dot_suite(
            lhs, rhs, configuration.vector_iterations, configuration.samples);

        run_polynomial_suite(
            lhs, configuration.vector_iterations, configuration.samples);

        run_convolution_suite(
            lhs, configuration.vector_iterations, configuration.samples);

        run_matrix_suite(
            configuration.matrix_dimension,
            configuration.matrix_iterations,
            configuration.samples);

        std::cout << "\nAll implementations passed validation.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'
                  << "Usage: simd_benchmark [element_count] [vector_iterations] "
                     "[odd_samples] [matrix_dimension] [matrix_iterations]\n";
        return EXIT_FAILURE;
    }
}
