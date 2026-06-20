cmake_minimum_required(VERSION 3.20)

project(simd_vectorization_benchmark VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_VECTORIZATION_REPORTS "Print compiler vectorization diagnostics" OFF)

add_executable(simd_benchmark
    src/main.cpp
    src/kernels_scalar.cpp
    src/kernels_auto.cpp
)

target_include_directories(simd_benchmark PRIVATE include)

if(MSVC)
    target_compile_options(simd_benchmark PRIVATE /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(simd_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Keep the scalar baseline optimized, but explicitly disable its vectorizers.
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    set_source_files_properties(src/kernels_scalar.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize"
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(src/kernels_scalar.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize"
    )
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SIMD_PROCESSOR)

if(SIMD_PROCESSOR MATCHES "arm64|aarch64")
    target_sources(simd_benchmark PRIVATE src/kernels_neon.cpp)
    message(STATUS "SIMD backend: ARM NEON")
elseif(SIMD_PROCESSOR MATCHES "x86_64|amd64")
    target_sources(simd_benchmark PRIVATE src/kernels_avx2.cpp)

    if(MSVC)
        # MSVC's /arch:AVX2 also enables FMA code generation on supported CPUs.
        set_source_files_properties(src/kernels_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2"
        )
        set_source_files_properties(src/kernels_auto.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        # Apply AVX2/FMA to the explicit SIMD implementation and to the
        # auto-vectorized implementation, but not to the scalar baseline.
        set_source_files_properties(src/kernels_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma"
        )
        set_source_files_properties(src/kernels_auto.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma"
        )
    else()
        message(FATAL_ERROR "Unsupported x86-64 compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    message(STATUS "SIMD backend: x86 AVX2 + FMA")
else()
    message(FATAL_ERROR "Unsupported processor: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(ENABLE_VECTORIZATION_REPORTS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
        target_compile_options(simd_benchmark PRIVATE
            -Rpass=loop-vectorize
            -Rpass-missed=loop-vectorize
            -Rpass-analysis=loop-vectorize
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(simd_benchmark PRIVATE
            -fopt-info-vec-optimized
            -fopt-info-vec-missed
        )
    elseif(MSVC)
        target_compile_options(simd_benchmark PRIVATE /Qvec-report:2)
    endif()
endif()
