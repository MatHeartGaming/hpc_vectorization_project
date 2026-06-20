#include "kernels.hpp"

#include <arm_neon.h>

void saxpy_native_simd(const float a, const float* x, float* y, const std::size_t n) {
    std::size_t i = 0;
    const float32x4_t va = vdupq_n_f32(a);

    // Apple Silicon is AArch64: one NEON register contains four Float32 values.
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vx = vld1q_f32(x + i);
        const float32x4_t vy = vld1q_f32(y + i);
        const float32x4_t result = vfmaq_f32(vy, vx, va);
        vst1q_f32(y + i, result);
    }

    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

const char* native_simd_name() noexcept {
    return "ARM NEON (128-bit, 4 x float32)";
}
