#include "kernels.hpp"

#include <immintrin.h>

void saxpy_native_simd(const float a, const float* x, float* y, const std::size_t n) {
    std::size_t i = 0;
    const __m256 va = _mm256_set1_ps(a);

    // One AVX2 register contains eight Float32 values.
    for (; i + 8 <= n; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        const __m256 vy = _mm256_loadu_ps(y + i);
        const __m256 result = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(y + i, result);
    }

    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

const char* native_simd_name() noexcept {
    return "x86 AVX2 + FMA (256-bit, 8 x float32)";
}
