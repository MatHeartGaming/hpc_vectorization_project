# SIMD Vectorization Benchmark Starter

A small C++20/CMake project for comparing three implementations of SAXPY:

1. optimized scalar code with compiler vectorization disabled;
2. ordinary C++ that the compiler may auto-vectorize;
3. manual architecture-specific SIMD intrinsics:
   - ARM NEON on Apple Silicon/AArch64;
   - AVX2 + FMA on AMD/x86-64.

SAXPY computes:

```text
y[i] = a * x[i] + y[i]
```

It is a useful first benchmark because the scalar, auto-vectorized, NEON, and AVX2 versions all implement exactly the same algorithm.

## Recommended macOS setup

Install Xcode or at least its command-line tools:

```bash
xcode-select --install
```

Open the project root in CLion. CLion should detect Apple Clang and CMake automatically. Select a **Release** CMake profile before running benchmarks.

## Build from the terminal

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/simd_benchmark
```

Optional arguments:

```bash
./build/simd_benchmark <element_count> <iterations> <odd_number_of_samples>
```

Example:

```bash
./build/simd_benchmark 16777216 20 7
```

## Compiler vectorization reports

```bash
cmake -S . -B build-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_VECTORIZATION_REPORTS=ON
cmake --build build-report --parallel
```

## Inspect generated instructions

On macOS:

```bash
otool -tvV build/simd_benchmark | less
```

On Linux/x86-64:

```bash
objdump -d -C -Mintel build/simd_benchmark | less
```

Look for AArch64/NEON instructions such as `fmla`, and x86 instructions such as `vfmadd*`, `vmovups`, or other `ymm`-register operations.

## Benchmarking rules

- Build in Release mode; never report Debug numbers.
- Use one thread initially.
- Use the same element count, iteration count, data type, and compiler family on both machines.
- Run on AC power and close heavy background applications.
- Record exact CPU/SoC, OS, compiler version, build flags, and thermal/power conditions.
- Report the median of several samples, not a single run.
- Treat SAXPY mainly as a memory-bandwidth benchmark. Add a reduction and a compute-heavy kernel later.

## Suggested next kernels

1. vector addition;
2. dot product/reduction;
3. element-wise polynomial with several FMAs;
4. small matrix multiplication or image convolution.

For a fair dissertation-style analysis, report both:

- **native best width**: Apple NEON 128-bit versus AMD AVX2 256-bit;
- an optional **width-normalized** experiment, such as NEON 128-bit versus x86 SSE/128-bit, to separate ISA width from other microarchitectural effects.
