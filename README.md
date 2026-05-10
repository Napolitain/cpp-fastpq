# cpp-fastpq

`cpp-fastpq` is a header-only family of bounded-priority queues. The queues
replace heap ordering with priority buckets and small occupancy indexes when
the workload can promise a bounded, growing, or sparse priority shape.

Priority `0` is the highest priority. Equal-priority values are popped in FIFO
order.

## Queue Selection

| Queue | Use when | Priority range | Push/pop pattern | Goal |
| --- | --- | --- | --- | --- |
| `cpp_pq::bulk_bucket_priority_queue<T, N>` | All values are pushed before draining starts | Fixed at compile time | Fill, then drain | Avoid live occupancy metadata; total drain work is O(items + buckets) |
| `cpp_pq::bucket_priority_queue<T, N>` / `static_bucket_priority_queue<T, N>` | Push and pop can be interleaved | Fixed at compile time | Live/interleaved | Use a compile-time bitmap summary to find the next non-empty bucket |
| `cpp_pq::dynamic_bucket_priority_queue_exact_growth<T>` | The range grows and exact logical capacity matters | Dynamic dense range | Live/interleaved | Grow to the requested priority range without geometric spare capacity |
| `cpp_pq::dynamic_bucket_priority_queue<T>` | The range grows over time | Dynamic dense range | Live/interleaved | Geometric bucket growth to reduce repeated reallocations |
| `cpp_pq::paged_dynamic_bucket_priority_queue<T>` | The range is large and active priorities cluster by page | Dynamic sparse-ish range | Live/interleaved | Allocate 64-priority pages lazily |
| `cpp_pq::registered_bucket_priority_queue<T>` | A small sparse set of priorities is reused | Registered sparse priorities | Live/interleaved | Push through priority handles without materializing the full range |

## Bulk Versus Live

Use `bulk_bucket_priority_queue` only when no new values are pushed while a
non-empty drain is in progress. It throws `std::logic_error` if that contract is
violated. After the queue is fully drained, pushing starts a new fill phase.

Use a live queue when enqueue and dequeue can interleave. Live queues maintain
occupancy metadata so a newly active high-priority bucket can become visible
without restarting a bulk scan.

## Usage

```cpp
#include "cpp_pq/bucket_priority_queue.hpp"

cpp_pq::bucket_priority_queue<int, 8> live;
live.push(2, 20);
live.push(7, 70);
live.pop();

cpp_pq::bulk_bucket_priority_queue<int, 8> bulk;
bulk.push(1, 10);
bulk.push(6, 60);
while (!bulk.empty()) {
    bulk.pop();
}
```

## Build

The library is header-only and requires C++20 or newer. The CMake project builds
tests and benchmarks when configured as the top-level project.

## Benchmarks

The C++ benchmark uses the same workload names and dimensions as the Go
benchmark:

- `fill_drain`: push all items, then pop all items
- `steady_state`: prefill, then repeatedly pop one item and push one item
- `sparse_reused`: reuse 16 active priorities across a 1,000,000-priority range

The shared fixed-range matrix is `buckets={16,1024,100000}` and
`items_per_bucket={1,100}`. Every workload includes the `stdlib_heap` baseline,
implemented with `std::priority_queue`.

Run:

```bash
cmake -S . -B build -DCPP_PQ_BUILD_BENCHMARKS=ON
cmake --build build --target cpp_pq_benchmark
./build/benchmark_priority_queues
```

The benchmark writes `benchmark_priority_queue_results.csv` with columns shared
with the Go benchmark concepts: `workload`, `buckets`, `items_per_bucket`,
`items`, `active_priorities`, `container`, `unit`, `ns_per_op`, and
`speedup_vs_std_heap`. Override the fixed-range item cutoff with
`FASTPQ_BENCH_MAX_ITEMS`.
