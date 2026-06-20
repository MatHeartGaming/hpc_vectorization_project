#include "kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchmarkResult {
    std::string name;
    double seconds{};
    double gb_per_second{};
    double gflops{};
    double checksum{};
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

double checksum(const std::vector<float>& values) {
    double sum = 0.0;
    for (const float value : values) {
        sum += static_cast<double>(value);
    }
    return sum;
}

void verify_kernel(
    const char* name,
    const SaxpyKernel kernel,
    const float a,
    const std::vector<float>& x,
    const std::vector<float>& initial_y,
    const std::vector<float>& reference
) {
    std::vector<float> actual = initial_y;
    kernel(a, x.data(), actual.data(), actual.size());

    float maximum_error = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        maximum_error = std::max(maximum_error, std::abs(actual[i] - reference[i]));
    }

    if (maximum_error > 1.0e-5F) {
        throw std::runtime_error(std::string(name) + " failed validation; max error = " +
                                 std::to_string(maximum_error));
    }
}

BenchmarkResult benchmark(
    std::string name,
    const SaxpyKernel kernel,
    const float a,
    const std::vector<float>& x,
    const std::vector<float>& initial_y,
    const std::size_t iterations,
    const std::size_t samples
) {
    using Clock = std::chrono::steady_clock;

    std::vector<float> y = initial_y;

    // Warm the code and data path. This is intentionally outside measured samples.
    for (int warmup = 0; warmup < 3; ++warmup) {
        kernel(a, x.data(), y.data(), y.size());
    }

    std::vector<double> timings;
    timings.reserve(samples);

    for (std::size_t sample = 0; sample < samples; ++sample) {
        y = initial_y;

        const auto start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            kernel(a, x.data(), y.data(), y.size());
        }
        const auto end = Clock::now();

        timings.push_back(std::chrono::duration<double>(end - start).count());
    }

    std::sort(timings.begin(), timings.end());
    const double median_seconds = timings[timings.size() / 2];

    // SAXPY performs one multiply and one add per element.
    const double operations = 2.0 * static_cast<double>(x.size()) * static_cast<double>(iterations);

    // Approximate minimum memory traffic: read x, read y, write y = 12 bytes/element.
    const double bytes = 3.0 * sizeof(float) * static_cast<double>(x.size()) *
                         static_cast<double>(iterations);

    return {
        .name = std::move(name),
        .seconds = median_seconds,
        .gb_per_second = bytes / median_seconds / 1.0e9,
        .gflops = operations / median_seconds / 1.0e9,
        .checksum = checksum(y),
    };
}

void print_result(const BenchmarkResult& result, const double scalar_seconds) {
    std::cout << std::left << std::setw(18) << result.name
              << std::right << std::fixed << std::setprecision(6)
              << std::setw(12) << result.seconds
              << std::setprecision(2)
              << std::setw(12) << result.gb_per_second
              << std::setw(12) << result.gflops
              << std::setw(12) << scalar_seconds / result.seconds
              << std::setprecision(4)
              << std::setw(18) << result.checksum << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t element_count = argc > 1
            ? parse_positive_size(argv[1], "element_count")
            : (std::size_t{1} << 24);
        const std::size_t iterations = argc > 2
            ? parse_positive_size(argv[2], "iterations")
            : 20;
        const std::size_t samples = argc > 3
            ? parse_positive_size(argv[3], "samples")
            : 7;

        if (samples % 2 == 0) {
            throw std::invalid_argument("samples must be odd so that the median is unambiguous");
        }

        constexpr float a = 0.25F;
        std::vector<float> x(element_count);
        std::vector<float> initial_y(element_count);

        for (std::size_t i = 0; i < element_count; ++i) {
            x[i] = static_cast<float>(static_cast<int>(i % 1024) - 512) * 0.001F;
            initial_y[i] = static_cast<float>(static_cast<int>(i % 257) - 128) * 0.002F;
        }

        std::vector<float> reference = initial_y;
        saxpy_scalar(a, x.data(), reference.data(), reference.size());
        verify_kernel("auto", saxpy_auto, a, x, initial_y, reference);
        verify_kernel("native SIMD", saxpy_native_simd, a, x, initial_y, reference);

        std::cout << "Native SIMD backend: " << native_simd_name() << '\n'
                  << "Elements: " << element_count
                  << ", iterations/sample: " << iterations
                  << ", samples: " << samples << " (median reported)\n\n";

        const BenchmarkResult scalar = benchmark(
            "scalar", saxpy_scalar, a, x, initial_y, iterations, samples);
        const BenchmarkResult automatic = benchmark(
            "auto-vectorized", saxpy_auto, a, x, initial_y, iterations, samples);
        const BenchmarkResult native = benchmark(
            "manual SIMD", saxpy_native_simd, a, x, initial_y, iterations, samples);

        std::cout << std::left << std::setw(18) << "kernel"
                  << std::right << std::setw(12) << "seconds"
                  << std::setw(12) << "GB/s"
                  << std::setw(12) << "GFLOP/s"
                  << std::setw(12) << "speedup"
                  << std::setw(18) << "checksum" << '\n';

        print_result(scalar, scalar.seconds);
        print_result(automatic, scalar.seconds);
        print_result(native, scalar.seconds);

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'
                  << "Usage: simd_benchmark [element_count] [iterations] [odd_samples]\n";
        return EXIT_FAILURE;
    }
}
