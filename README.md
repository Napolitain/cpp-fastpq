# cpp-fastpq

`cpp-fastpq` is a header-only family of bucket-based priority queues for C++20
and newer.

Use it when priorities are bounded or sparse enough that bucket lookup beats a
comparison heap. Priority `0` is the highest priority, and values with the same
priority are popped in FIFO order.

## Consume With CMake

Fetch the library and link the exported interface target:

```cmake
include(FetchContent)

FetchContent_Declare(
    cpp_fastpq
    GIT_REPOSITORY https://github.com/Napolitain/cpp-fastpq.git
    GIT_TAG main # Prefer a release tag or commit SHA in production.
)
FetchContent_MakeAvailable(cpp_fastpq)

target_link_libraries(your_target PRIVATE cpp_pq::cpp_pq)
```

Or use an installed package:

```cmake
find_package(cpp_pq CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE cpp_pq::cpp_pq)
```

Then include:

```cpp
#include <cpp_pq/bucket_priority_queue.hpp>
```

## Choose A Queue

| Queue | Use when | Priority range | Push/pop pattern | Goal |
| --- | --- | --- | --- | --- |
| `cpp_pq::bulk_bucket_priority_queue<T, N>` | All values are pushed before draining starts | Fixed at compile time | Fill, then drain | Avoid live occupancy metadata; total drain work is O(items + buckets) |
| `cpp_pq::bucket_priority_queue<T, N>` / `static_bucket_priority_queue<T, N>` | Push and pop can be interleaved | Fixed at compile time | Live/interleaved | Use a compile-time bitmap summary to find the next non-empty bucket |
| `cpp_pq::dynamic_bucket_priority_queue<T>` | The range grows over time | Dynamic dense range | Live/interleaved | Geometric bucket growth to reduce repeated reallocations |
| `cpp_pq::dynamic_bucket_priority_queue_exact_growth<T>` | The range grows and exact logical capacity matters | Dynamic dense range | Live/interleaved | Grow to the requested priority range without geometric spare capacity |
| `cpp_pq::paged_dynamic_bucket_priority_queue<T>` | The range is large and active priorities cluster by page | Dynamic sparse-ish range | Live/interleaved | Allocate 64-priority pages lazily |
| `cpp_pq::registered_bucket_priority_queue<T>` | A small sparse set of priorities is reused | Registered sparse priorities | Live/interleaved | Push through priority handles without materializing the full range |

Use `bucket_priority_queue<T, N>` when the priority count is known at compile
time. Use `dynamic_bucket_priority_queue<T>` when the dense range grows at
runtime. Use `bulk_bucket_priority_queue<T, N>` only for strict fill-then-drain
batches.

## Live Queue Example

```cpp
#include <iostream>
#include <string>

#include <cpp_pq/bucket_priority_queue.hpp>

int main() {
    cpp_pq::bucket_priority_queue<std::string, 4> queue;

    queue.push(2, "background-a");
    queue.push(0, "urgent");
    queue.push(1, "normal");
    queue.push(2, "background-b");

    while (!queue.empty()) {
        std::cout << queue.top() << '\n';
        queue.pop();
    }
}
```

Output:

```text
urgent
normal
background-a
background-b
```

`top()` and `pop()` throw `std::out_of_range` on an empty queue. Fixed queues
also throw `std::out_of_range` when a pushed priority is outside `[0, N)`.

## Bulk Queue Example

```cpp
cpp_pq::bulk_bucket_priority_queue<int, 128> queue;

for (const auto& job : jobs) {
    queue.push(job.priority, job.id);
}

while (!queue.empty()) {
    run_job(queue.top());
    queue.pop();
}
```

`bulk_bucket_priority_queue` throws `std::logic_error` if you push after
`top()` or `pop()` has started a drain phase and the queue still contains
values. Fully drain it or call `clear()` before starting a new batch.

## Dynamic And Sparse Queues

```cpp
cpp_pq::dynamic_bucket_priority_queue<int> dense;
dense.push(64, 1); // grows to include priority 64

cpp_pq::paged_dynamic_bucket_priority_queue<int> paged;
paged.push(1'000'000, 2); // allocates only the active 64-priority page

cpp_pq::registered_bucket_priority_queue<int> registered;
auto low = registered.register_priority(5);
registered.push(low, 3);
```

The queues are not synchronized. Protect shared instances with your own lock.

## Local Quality Gate

Run the same gate used by GitHub Actions:

```bash
./scripts/ci.sh
```

It compiles and runs the tests, compiles the benchmark, and when CMake is
available also configures the project, builds tests and benchmarks, and runs
`ctest`.

## Benchmarks

The C++ benchmark uses the same workload names and dimensions as the Go
benchmark:

- `fill_drain`: push all items, then pop all items
- `steady_state`: prefill, then repeatedly pop one item and push one item
- `sparse_reused`: reuse 16 active priorities across a 1,000,000-priority range

Every workload includes the `stdlib_heap` baseline implemented with
`std::priority_queue`.

Run:

```bash
cmake -S . -B build -DCPP_PQ_BUILD_BENCHMARKS=ON
cmake --build build --target cpp_pq_benchmark
./build/benchmark_priority_queues
```

The benchmark writes `benchmark_priority_queue_results.csv`. Override the
fixed-range item cutoff with `FASTPQ_BENCH_MAX_ITEMS`.
