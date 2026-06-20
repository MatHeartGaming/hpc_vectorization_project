# Cross-platform SIMD Vectorization Benchmark Suite

A C++20/CMake project for comparing three implementations of common HPC kernels:

1. **optimized scalar** code with compiler vectorization disabled;
2. ordinary C++ that the compiler may **auto-vectorize**;
3. **manual architecture-specific SIMD intrinsics**:
   - ARM NEON on Apple Silicon/AArch64;
   - AVX2 + FMA on AMD and Intel x86-64.

The same source tree is intended for:

- Apple M2 Pro;
- Apple M5 Pro;
- AMD Ryzen 7 5800H.

## Included benchmark kernels

| Kernel | Formula or operation | Main behavior exposed |
|---|---|---|
| Vector addition | `out[i] = lhs[i] + rhs[i]` | streaming memory bandwidth |
| Vector subtraction | `out[i] = lhs[i] - rhs[i]` | streaming memory bandwidth |
| Vector multiplication | `out[i] = lhs[i] * rhs[i]` | streaming arithmetic |
| Vector division | `out[i] = lhs[i] / rhs[i]` | expensive arithmetic throughput |
| SAXPY | `y[i] = a * x[i] + y[i]` | FMA plus memory bandwidth |
| Dot product | `sum(lhs[i] * rhs[i])` | SIMD reduction |
| Polynomial | fourth-degree Horner evaluation | compute-heavy FMA throughput |
| 1D convolution | seven-tap valid FIR filter | sliding-window arithmetic and cache reuse |
| Matrix multiplication | `C = A * B` | reductions, cache behavior, and high arithmetic intensity |

Every kernel has scalar, auto-vectorized, and native SIMD implementations.

## Project structure

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

## Recommended macOS setup

Install Xcode or at least the command-line tools:

```bash
xcode-select --install
```

Open the project root in CLion and select a **Release** CMake profile. CMake detects Apple Silicon and includes `kernels_neon.cpp` automatically.

Build from Terminal:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/simd_benchmark
```

## Recommended Windows setup

The project supports either:

- Visual Studio/MSVC;
- CLion with Visual Studio toolchain;
- CLion with LLVM/Clang;
- MinGW GCC;
- WSL with GCC or Clang.

With a Visual Studio Developer PowerShell:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
.\build\Release\simd_benchmark.exe
```

For single-configuration generators such as Ninja, the executable is normally under `build` rather than `build/Release`.

On the Ryzen 7 5800H, CMake enables AVX2 and FMA explicitly. It does not use `-march=native`, which avoids accidentally changing the ISA target between x86 systems.

## Command-line arguments

```text
simd_benchmark \
  [element_count] \
  [vector_iterations] \
  [odd_samples] \
  [matrix_dimension] \
  [matrix_iterations]
```

Defaults:

```text
element_count       = 1,048,576
vector_iterations   = 20
samples             = 5
matrix_dimension    = 192
matrix_iterations   = 2
```

Example quick run:

```bash
./build/simd_benchmark 65536 3 3 48 1
```

Example larger run:

```bash
./build/simd_benchmark 16777216 20 7 256 3
```

The sample count must be odd because the program reports the median sample.

## Reported metrics

The executable reports:

- median elapsed time;
- effective GB/s for streaming kernels;
- GFLOP/s;
- speedup relative to the scalar implementation;
- checksum used to make the result observable.

GB/s is intentionally omitted for convolution and matrix multiplication because a single minimum-byte estimate would hide cache reuse and would not be directly comparable with the streaming kernels.

All implementations are validated before timing. Reduction kernels use tolerances because SIMD changes floating-point accumulation order.

## Fast-math and reductions

`ENABLE_FAST_MATH` is enabled by default:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_FAST_MATH=ON
```

This permits floating-point reassociation, which is normally needed for auto-vectorizing reductions such as dot products and matrix row dot products. It is applied only to kernel translation units, not the benchmark harness.

The scalar translation unit receives the same floating-point mode, but its loop and SLP vectorizers are disabled. This helps isolate vectorization from floating-point policy.

A strict floating-point experiment can also be built:

```bash
cmake -S . -B build-strict \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_FAST_MATH=OFF
cmake --build build-strict --parallel
```

Some reduction loops may not auto-vectorize in strict mode. That difference itself can be documented in the report.

## Compiler vectorization reports

```bash
cmake -S . -B build-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_VECTORIZATION_REPORTS=ON
cmake --build build-report --parallel
```

The project enables the corresponding diagnostics for:

- Apple Clang and LLVM Clang;
- GCC;
- MSVC.

Review the diagnostics for `kernels_auto.cpp`. The scalar file is expected to report disabled or missed vectorization.

## Inspect generated instructions

On macOS:

```bash
otool -tvV build/simd_benchmark | less
```

Useful AArch64 instructions include `fadd`, `fmul`, `fdiv`, and `fmla` operating on vector registers.

On Linux/x86-64:

```bash
objdump -d -C -Mintel build/simd_benchmark | less
```

Useful AVX2/FMA evidence includes `ymm` registers and instructions such as `vaddps`, `vmulps`, `vdivps`, and `vfmadd*`.

On Windows, Visual Studio's Disassembly window or `dumpbin /DISASM` can be used.

## Benchmark design notes

### Matrix multiplication

The right-hand matrix is transposed once before timing. Each output element is therefore calculated as a dot product between contiguous rows:

```text
C[row, column] = dot(A[row, :], transpose(B)[column, :])
```

The transpose cost is excluded deliberately. This benchmark measures multiplication and reduction rather than a mixture of multiplication and data-layout conversion.

### Convolution

The convolution uses a fixed seven-tap filter and valid output boundaries:

```text
output_count = input_count - filter_count + 1
```

All three implementations use the same tap-major algorithm. This avoids comparing different algorithms under the label of SIMD acceleration.

### Vector division

Division is included because it behaves differently from addition and multiplication. It is usually more compute-bound and may expose architecture-specific throughput differences even when add and multiply are limited by memory bandwidth.

## Suggested experimental matrix

Run exactly the same configurations on all three systems:

```text
M2 Pro    scalar / auto / NEON
M5 Pro    scalar / auto / NEON
5800H     scalar / auto / AVX2 + FMA
```

Use several working-set sizes, for example:

```text
16,384
262,144
4,194,304
16,777,216
67,108,864 elements
```

For each result, record:

- CPU or SoC;
- OS version;
- compiler and version;
- Release build flags;
- problem size and iteration count;
- power source and power mode;
- median and variability across samples.

## Benchmarking rules

- Never report Debug-build measurements.
- Begin with one thread so the experiment measures SIMD rather than thread scaling.
- Use the same data type, formulas, sizes, and compiler family where practical.
- Run on AC power and close heavy background applications.
- Avoid benchmarking immediately after startup or during OS updates and indexing.
- Repeat the full run more than once and investigate large variance.
- Treat results from different kernel categories separately: memory-bound, reduction-heavy, and compute-heavy.
- Do not compare the RTX 3070 in the same primary SIMD table. CUDA should be a separate GPU experiment.
