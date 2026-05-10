#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/.."

: "${CXX:=c++}"
: "${BUILD_DIR:=build/local-ci}"

mkdir -p "$BUILD_DIR"

common_flags=(-std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude)

echo "==> direct test compile"
"$CXX" "${common_flags[@]}" tests/multi_queue_priority_queue_tests.cpp -o "$BUILD_DIR/cpp_pq_tests"

echo "==> direct test run"
"$BUILD_DIR/cpp_pq_tests"

echo "==> direct benchmark compile"
"$CXX" "${common_flags[@]}" benchmarks/benchmark_priority_queues.cpp -o "$BUILD_DIR/cpp_pq_benchmark"

if command -v cmake >/dev/null 2>&1; then
    cmake_build_dir="$BUILD_DIR/cmake"

    echo "==> cmake configure"
    cmake -S . -B "$cmake_build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCPP_PQ_BUILD_TESTS=ON \
        -DCPP_PQ_BUILD_BENCHMARKS=ON \
        -DCPP_PQ_ENABLE_FOLLY_FBVECTOR=OFF

    echo "==> cmake build"
    cmake --build "$cmake_build_dir" --parallel

    echo "==> ctest"
    ctest --test-dir "$cmake_build_dir" --output-on-failure
else
    echo "cmake not found; skipped CMake configure/build/ctest."
fi
