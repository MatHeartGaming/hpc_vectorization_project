# Cross-platform SIMD Vectorization Benchmark Suite

A C++20/CMake university project for comparing:

1. **optimized scalar code** with compiler vectorization disabled;
2. **ordinary C++ auto-vectorization**;
3. **manual native SIMD intrinsics**:
   - ARM NEON on Apple Silicon;
   - AVX2 + FMA on x86-64, including the Ryzen 7 5800H.

The executable validates every implementation against its scalar reference before reporting timings.

## Included benchmark kernels

| Category | Kernel | Main functions |
|---|---|---|
| Streaming | STREAM copy | `vector_copy_*` |
| Streaming | STREAM triad | `vector_triad_*` |
| Element-wise | add, subtract, multiply, divide | `vector_add_*`, `vector_subtract_*`, `vector_multiply_*`, `vector_divide_*` |
| Update | SAXPY | `saxpy_*` |
| Reduction | dot product | `dot_product_*` |
| Compute-heavy | fourth-degree Horner polynomial | `polynomial_*` |
| Matrix | small GEMM using transposed RHS | `matrix_multiply_*` |
| Matrix | cache-blocked large GEMM | `matrix_multiply_blocked_*` |
| Signal processing | seven-tap 1D convolution | `convolution_1d_*` |
| Signal processing | iterative radix-2 FFT | `fft_radix2_*` |
| Irregular access | gather/indexed sum | `gather_sum_*` |
| Irregular access | scatter-add | `scatter_add_*` |
| Irregular access | histogram update | `histogram_*` |

Every `*` family contains:

```text
_scalar
_auto
_native_simd
```

For example:

```text
vector_triad_scalar
vector_triad_auto
vector_triad_native_simd
```

## Why both small and large GEMM exist

`matrix_multiply_*` treats each output cell as a contiguous dot product. The right-hand matrix is transposed before timing, so the inner loop reads both operands sequentially. This is useful for relatively small matrices and directly exposes SIMD reduction performance.

`matrix_multiply_blocked_*` uses `row-block / k-block / column-block` cache tiling and vectorizes across adjacent output columns. It is intended for larger matrices where cache reuse matters. It is still an educational implementation rather than a replacement for Accelerate, BLAS, MKL, or another tuned library.

## FFT design

The FFT is an in-place iterative radix-2 Cooley-Tukey transform:

- input size must be a power of two;
- bit-reversal is performed inside each implementation;
- sine/cosine twiddle tables are precomputed by the benchmark harness outside the timed region;
- AVX2 gathers twiddles with `_mm256_i32gather_ps`;
- NEON processes butterfly data in vectors but assembles irregular twiddle values manually because classic NEON has no general gather instruction.

The reported FFT FLOP rate uses the common approximation:

```text
5 * N * log2(N)
```

## Gather, scatter, and histogram caveat

AVX2 provides hardware gather but no general hardware scatter. Classic NEON provides neither a general gather nor a general scatter instruction.

Consequently:

- AVX2 `gather_sum_native_simd` uses `_mm256_i32gather_ps`;
- NEON gather loads indexed values into lanes manually;
- scatter and histogram native implementations use SIMD loads but ordered scalar lane updates to preserve duplicate-index and bin-collision semantics.

These hybrid implementations are intentional. They demonstrate workloads for which manual SIMD may provide little or no speedup.

## Recommended tools

### Apple M2 Pro and M5 Pro

Install Xcode or the command-line tools:

```bash
xcode-select --install
```

Use CLion with an Apple Clang **Release** profile for building. Use Xcode Instruments separately for CPU profiling.

### Ryzen 7 5800H on Windows

Supported options include:

- CLion with the Visual Studio/MSVC toolchain;
- CLion with LLVM/Clang;
- Visual Studio with CMake;
- WSL with GCC or Clang.

The CMake project selects `/arch:AVX2` for MSVC and `-mavx2 -mfma` for Clang/GCC.

The RTX 3070 is not used: this suite measures CPU SIMD. CUDA would be a separate GPU experiment.

## Build

### macOS, Linux, or WSL

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/simd_benchmark
```

### Windows PowerShell with Visual Studio/MSVC

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
.\build\Release\simd_benchmark.exe
```

Do not report Debug-build results.

## Command-line arguments

```text
simd_benchmark \
  [vector_count] \
  [vector_iterations] \
  [odd_samples] \
  [small_gemm_dimension] \
  [large_gemm_dimension] \
  [fft_power_of_two]
```

Defaults:

```text
vector_count          = 1,048,576
vector_iterations     = 10
samples               = 5
small_gemm_dimension  = 64
large_gemm_dimension  = 256
fft_size              = 32,768
```

Quick smoke test:

```bash
./build/simd_benchmark 65536 2 3 32 64 1024
```

More substantial run:

```bash
./build/simd_benchmark 16777216 20 7 64 512 65536
```

Keep the same arguments on all three computers for a direct comparison.

## Floating-point mode

Fast math is enabled by default so reductions and multiply-add expressions may be reassociated and vectorized:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_FAST_MATH=ON
```

A stricter build is available:

```bash
cmake -S . -B build-strict \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_FAST_MATH=OFF
cmake --build build-strict --parallel
```

Because SIMD reductions can add values in a different order, validation uses small absolute and relative tolerances rather than bit-for-bit equality.

## Compiler vectorization reports

```bash
cmake -S . -B build-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_VECTORIZATION_REPORTS=ON
cmake --build build-report --parallel
```

The project requests:

- Clang/Apple Clang: `-Rpass`, `-Rpass-missed`, and `-Rpass-analysis` for loop vectorization;
- GCC: optimized and missed vectorization reports;
- MSVC: `/Qvec-report:2`.

## Inspect generated instructions

macOS:

```bash
otool -tvV build/simd_benchmark | less
```

Linux/WSL:

```bash
objdump -d -C -Mintel build/simd_benchmark | less
```

On Apple Silicon, look for instructions such as `fmla`, `ld1`, and `st1`. On x86-64, look for `ymm` registers, `vfmadd*`, `vmovups`, and gather instructions.

## Benchmarking rules

- Use Release mode.
- Connect laptops to power and use a consistent power profile.
- Close heavy background applications.
- Use the same compiler family and broadly comparable compiler versions when possible.
- Record CPU/SoC, RAM, OS, compiler version, flags, sample count, problem sizes, and thermal conditions.
- Report medians, not isolated runs.
- Run each configuration more than once and examine variability.
- Treat STREAM, SAXPY, and simple element-wise kernels primarily as memory-bandwidth tests.
- Treat polynomial and GEMM primarily as arithmetic/cache tests.
- Treat gather, scatter, and histogram as irregular-memory tests where SIMD may not help.

## Source layout

```text
simd-vectorization-benchmark/
├── CMakeLists.txt
├── README.md
├── include/
│   └── kernels.hpp
└── src/
    ├── main.cpp
    ├── kernels_scalar.cpp
    ├── kernels_auto.cpp
    ├── kernels_neon.cpp
    └── kernels_avx2.cpp
```
