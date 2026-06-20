cmake_minimum_required(VERSION 3.20)

project(simd_vectorization_benchmark VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_VECTORIZATION_REPORTS "Print compiler vectorization diagnostics" OFF)
option(ENABLE_FAST_MATH "Allow reassociation so reduction loops can vectorize" ON)

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

# Fast-math is applied only to kernel translation units, never to the benchmark
# harness. Reduction vectorization usually requires reassociation permission.
if(ENABLE_FAST_MATH)
    if(MSVC)
        set_property(SOURCE src/kernels_scalar.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:fast)
        set_property(SOURCE src/kernels_auto.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:fast)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        set_property(SOURCE src/kernels_scalar.cpp APPEND PROPERTY COMPILE_OPTIONS -ffast-math)
        set_property(SOURCE src/kernels_auto.cpp APPEND PROPERTY COMPILE_OPTIONS -ffast-math)
    endif()
endif()

# Keep the scalar baseline optimized while explicitly disabling loop and SLP
# vectorization in this translation unit. The source also includes MSVC loop
# pragmas because MSVC has no direct per-file equivalent of Clang's flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    set_property(SOURCE src/kernels_scalar.cpp APPEND PROPERTY COMPILE_OPTIONS
        -fno-vectorize -fno-slp-vectorize)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(SOURCE src/kernels_scalar.cpp APPEND PROPERTY COMPILE_OPTIONS
        -fno-tree-vectorize -fno-tree-slp-vectorize)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SIMD_PROCESSOR)

if(SIMD_PROCESSOR MATCHES "arm64|aarch64")
    target_sources(simd_benchmark PRIVATE src/kernels_neon.cpp)

    if(ENABLE_FAST_MATH)
        if(MSVC)
            set_property(SOURCE src/kernels_neon.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:fast)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
            set_property(SOURCE src/kernels_neon.cpp APPEND PROPERTY COMPILE_OPTIONS -ffast-math)
        endif()
    endif()

    message(STATUS "SIMD backend: ARM NEON")
elseif(SIMD_PROCESSOR MATCHES "x86_64|amd64")
    target_sources(simd_benchmark PRIVATE src/kernels_avx2.cpp)

    if(MSVC)
        set_property(SOURCE src/kernels_avx2.cpp APPEND PROPERTY COMPILE_OPTIONS /arch:AVX2)
        set_property(SOURCE src/kernels_auto.cpp APPEND PROPERTY COMPILE_OPTIONS /arch:AVX2)
        if(ENABLE_FAST_MATH)
            set_property(SOURCE src/kernels_avx2.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:fast)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        set_property(SOURCE src/kernels_avx2.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma)
        set_property(SOURCE src/kernels_auto.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma)
        if(ENABLE_FAST_MATH)
            set_property(SOURCE src/kernels_avx2.cpp APPEND PROPERTY COMPILE_OPTIONS -ffast-math)
        endif()
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
