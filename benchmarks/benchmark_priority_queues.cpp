#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "cpp_pq/bucket_priority_queue.hpp"

namespace {

constexpr std::array<std::size_t, 3> benchmark_bucket_counts{16, 1'024, 100'000};
constexpr std::array<std::size_t, 2> benchmark_items_per_bucket{1, 100};
constexpr std::size_t default_max_items = 10'000'000;
constexpr std::size_t sparse_priority_range = 1'000'000;
constexpr std::size_t sparse_active_priorities = 16;
constexpr std::size_t sparse_item_count = 100'000;
constexpr std::size_t benchmark_rounds = 5;

struct item {
    std::size_t priority;
    std::uint64_t value;
};

struct benchmark_case {
    std::size_t bucket_count;
    std::size_t items_per_bucket;
    std::size_t item_count;
};

struct sparse_benchmark_case {
    std::size_t priority_range;
    std::size_t active_priorities;
    std::size_t item_count;
};

struct benchmark_result {
    std::string workload;
    std::size_t bucket_count;
    std::size_t items_per_bucket;
    std::size_t item_count;
    std::size_t active_priorities;
    std::string container;
    std::string unit;
    double ns_per_op;
    std::uint64_t checksum;
};

struct priority_entry {
    std::size_t priority;
    std::uint64_t value;
    std::uint64_t sequence;
};

struct lower_priority {
    [[nodiscard]] bool operator()(const priority_entry& left, const priority_entry& right) const noexcept {
        return std::tie(left.priority, left.sequence) > std::tie(right.priority, right.sequence);
    }
};

[[nodiscard]] std::uint64_t mix_checksum(std::uint64_t checksum, std::uint64_t value) noexcept {
    checksum ^= value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
    return checksum;
}

class stdlib_heap_adapter {
public:
    explicit stdlib_heap_adapter(std::size_t = 0) {
    }

    void push(const item& next_item) {
        queue_.push(priority_entry{
            next_item.priority,
            next_item.value,
            next_sequence_++,
        });
    }

    [[nodiscard]] std::uint64_t top_value() const {
        return queue_.top().value;
    }

    void pop() {
        queue_.pop();
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

private:
    std::priority_queue<priority_entry, std::vector<priority_entry>, lower_priority> queue_{};
    std::uint64_t next_sequence_{0};
};

template <std::size_t BucketCount>
class bulk_queue_adapter {
public:
    explicit bulk_queue_adapter(std::size_t = 0) {
    }

    void push(const item& next_item) {
        queue_.push(next_item.priority, next_item.value);
    }

    [[nodiscard]] std::uint64_t top_value() const {
        return queue_.top();
    }

    void pop() {
        queue_.pop();
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

private:
    cpp_pq::bulk_bucket_priority_queue<std::uint64_t, BucketCount> queue_{};
};

template <std::size_t BucketCount>
class static_queue_adapter {
public:
    explicit static_queue_adapter(std::size_t = 0) {
    }

    void push(const item& next_item) {
        queue_.push(next_item.priority, next_item.value);
    }

    [[nodiscard]] std::uint64_t top_value() const {
        return queue_.top();
    }

    void pop() {
        queue_.pop();
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

private:
    cpp_pq::static_bucket_priority_queue<std::uint64_t, BucketCount> queue_{};
};

class dynamic_queue_adapter {
public:
    explicit dynamic_queue_adapter(std::size_t bucket_count = 0)
        : queue_(bucket_count) {
    }

    void push(const item& next_item) {
        queue_.push(next_item.priority, next_item.value);
    }

    [[nodiscard]] std::uint64_t top_value() const {
        return queue_.top();
    }

    void pop() {
        queue_.pop();
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

private:
    cpp_pq::dynamic_bucket_priority_queue<std::uint64_t> queue_;
};

class paged_queue_adapter {
public:
    explicit paged_queue_adapter(std::size_t bucket_count = 0)
        : queue_(bucket_count) {
    }

    void push(const item& next_item) {
        queue_.push(next_item.priority, next_item.value);
    }

    [[nodiscard]] std::uint64_t top_value() const {
        return queue_.top();
    }

    void pop() {
        queue_.pop();
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

private:
    cpp_pq::paged_dynamic_bucket_priority_queue<std::uint64_t> queue_;
};

class registered_queue_adapter {
public:
    explicit registered_queue_adapter(std::size_t = 0) {
    }

    void push(const item& next_item) {
        queue_.push(handle_for(next_item.priority), next_item.value);
    }

    [[nodiscard]] std::uint64_t top_value() const {
        return queue_.top();
    }

    void pop() {
        queue_.pop();
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

private:
    using handle_type = cpp_pq::registered_bucket_priority_queue<std::uint64_t>::priority_handle;

    handle_type handle_for(std::size_t priority) {
        if (const auto found = handles_.find(priority); found != handles_.end()) {
            return found->second;
        }

        auto handle = queue_.register_priority(priority);
        handles_.emplace(priority, handle);
        return handle;
    }

    cpp_pq::registered_bucket_priority_queue<std::uint64_t> queue_{};
    std::unordered_map<std::size_t, handle_type> handles_{};
};

[[nodiscard]] std::size_t max_items_from_env() {
    if (const char* raw = std::getenv("FASTPQ_BENCH_MAX_ITEMS")) {
        return std::max<std::size_t>(1, std::stoull(raw));
    }

    return default_max_items;
}

[[nodiscard]] std::size_t steady_priority(std::size_t step, std::size_t bucket_count) noexcept {
    return static_cast<std::size_t>((static_cast<std::uint64_t>(step) * 11'400'714'819'323'198'485ULL) % bucket_count);
}

[[nodiscard]] std::vector<item> make_fill_drain_items(const benchmark_case& next_case) {
    std::vector<item> items;
    items.reserve(next_case.item_count);

    std::uint64_t value = 0;
    for (std::size_t bucket_offset = 0; bucket_offset < next_case.items_per_bucket; ++bucket_offset) {
        for (std::size_t priority = 0; priority < next_case.bucket_count; ++priority) {
            items.push_back(item{priority, value++});
        }
    }

    return items;
}

[[nodiscard]] std::vector<item> make_steady_items(std::size_t operation_count, std::size_t bucket_count) {
    std::vector<item> items;
    items.reserve(operation_count);

    for (std::size_t index = 0; index < operation_count; ++index) {
        items.push_back(item{steady_priority(index, bucket_count), index});
    }

    return items;
}

[[nodiscard]] std::vector<std::size_t> sparse_reused_priorities(const sparse_benchmark_case& next_case) {
    if (next_case.active_priorities <= 1) {
        return {0};
    }

    std::vector<std::size_t> priorities;
    priorities.reserve(next_case.active_priorities);

    for (std::size_t index = 0; index < next_case.active_priorities; ++index) {
        priorities.push_back((index * (next_case.priority_range - 1)) / (next_case.active_priorities - 1));
    }

    return priorities;
}

[[nodiscard]] std::vector<item> make_sparse_reused_items(const sparse_benchmark_case& next_case) {
    const auto priorities = sparse_reused_priorities(next_case);
    std::vector<item> items;
    items.reserve(next_case.item_count);

    for (std::size_t value = 0; value < next_case.item_count; ++value) {
        items.push_back(item{priorities[value % priorities.size()], value});
    }

    return items;
}

template <typename Queue>
std::uint64_t run_fill_drain_round(std::span<const item> items, std::size_t bucket_count) {
    auto queue = std::make_unique<Queue>(bucket_count);
    std::uint64_t checksum = 0;

    for (const auto& next_item : items) {
        queue->push(next_item);
    }

    while (!queue->empty()) {
        checksum = mix_checksum(checksum, queue->top_value());
        queue->pop();
    }

    return checksum;
}

template <typename Queue>
std::uint64_t run_steady_round(
    std::span<const item> seed_items,
    std::span<const item> operation_items,
    std::size_t bucket_count
) {
    auto queue = std::make_unique<Queue>(bucket_count);
    std::uint64_t checksum = 0;

    for (const auto& next_item : seed_items) {
        queue->push(next_item);
    }

    for (const auto& next_item : operation_items) {
        checksum = mix_checksum(checksum, queue->top_value());
        queue->pop();
        queue->push(next_item);
    }

    return checksum;
}

template <typename Fn>
benchmark_result measure(
    std::string workload,
    std::size_t bucket_count,
    std::size_t items_per_bucket,
    std::size_t item_count,
    std::size_t active_priorities,
    std::string container,
    std::string unit,
    std::size_t measured_operations,
    Fn&& fn
) {
    std::uint64_t checksum = 0;
    const auto started_at = std::chrono::steady_clock::now();

    for (std::size_t round = 0; round < benchmark_rounds; ++round) {
        checksum ^= fn();
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    return benchmark_result{
        std::move(workload),
        bucket_count,
        items_per_bucket,
        item_count,
        active_priorities,
        std::move(container),
        std::move(unit),
        (elapsed * 1'000'000'000.0) / static_cast<double>(measured_operations * benchmark_rounds),
        checksum,
    };
}

template <std::size_t BucketCount>
void append_fill_drain_results(const benchmark_case& next_case, std::vector<benchmark_result>& results) {
    const auto items = make_fill_drain_items(next_case);
    const auto measured_operations = next_case.item_count * 3;

    results.push_back(measure("fill_drain", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "stdlib_heap", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<stdlib_heap_adapter>(items, next_case.bucket_count);
                              }));
    results.push_back(measure("fill_drain", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "bulk_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<bulk_queue_adapter<BucketCount>>(
                                      items,
                                      next_case.bucket_count
                                  );
                              }));
    results.push_back(measure("fill_drain", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "static_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<static_queue_adapter<BucketCount>>(
                                      items,
                                      next_case.bucket_count
                                  );
                              }));
    results.push_back(measure("fill_drain", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "dynamic_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<dynamic_queue_adapter>(items, next_case.bucket_count);
                              }));
    results.push_back(measure("fill_drain", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "paged_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<paged_queue_adapter>(items, next_case.bucket_count);
                              }));
}

template <std::size_t BucketCount>
void append_steady_results(const benchmark_case& next_case, std::vector<benchmark_result>& results) {
    const auto seed_items = make_fill_drain_items(next_case);
    const auto operation_items = make_steady_items(next_case.item_count, next_case.bucket_count);

    results.push_back(measure("steady_state", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "stdlib_heap", "pop_push", next_case.item_count, [&]() {
                                  return run_steady_round<stdlib_heap_adapter>(
                                      seed_items,
                                      operation_items,
                                      next_case.bucket_count
                                  );
                              }));
    results.push_back(measure("steady_state", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "static_queue", "pop_push", next_case.item_count, [&]() {
                                  return run_steady_round<static_queue_adapter<BucketCount>>(
                                      seed_items,
                                      operation_items,
                                      next_case.bucket_count
                                  );
                              }));
    results.push_back(measure("steady_state", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "dynamic_queue", "pop_push", next_case.item_count, [&]() {
                                  return run_steady_round<dynamic_queue_adapter>(
                                      seed_items,
                                      operation_items,
                                      next_case.bucket_count
                                  );
                              }));
    results.push_back(measure("steady_state", next_case.bucket_count, next_case.items_per_bucket, next_case.item_count, 0,
                              "paged_queue", "pop_push", next_case.item_count, [&]() {
                                  return run_steady_round<paged_queue_adapter>(
                                      seed_items,
                                      operation_items,
                                      next_case.bucket_count
                                  );
                              }));
}

void append_sparse_results(const sparse_benchmark_case& next_case, std::vector<benchmark_result>& results) {
    const auto items = make_sparse_reused_items(next_case);
    const auto measured_operations = next_case.item_count * 3;

    results.push_back(measure("sparse_reused", next_case.priority_range, 0, next_case.item_count, next_case.active_priorities,
                              "stdlib_heap", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<stdlib_heap_adapter>(items, next_case.priority_range);
                              }));
    results.push_back(measure("sparse_reused", next_case.priority_range, 0, next_case.item_count, next_case.active_priorities,
                              "dynamic_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<dynamic_queue_adapter>(items, next_case.priority_range);
                              }));
    results.push_back(measure("sparse_reused", next_case.priority_range, 0, next_case.item_count, next_case.active_priorities,
                              "paged_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<paged_queue_adapter>(items, next_case.priority_range);
                              }));
    results.push_back(measure("sparse_reused", next_case.priority_range, 0, next_case.item_count, next_case.active_priorities,
                              "registered_queue", "push_top_pop", measured_operations, [&]() {
                                  return run_fill_drain_round<registered_queue_adapter>(items, next_case.priority_range);
                              }));
}

template <typename Fn>
void dispatch_bucket_count(std::size_t bucket_count, Fn&& fn) {
    switch (bucket_count) {
    case 16:
        fn.template operator()<16>();
        return;
    case 1'024:
        fn.template operator()<1'024>();
        return;
    case 100'000:
        fn.template operator()<100'000>();
        return;
    default:
        throw std::invalid_argument("unsupported benchmark bucket count");
    }
}

[[nodiscard]] std::vector<benchmark_case> build_benchmark_cases() {
    const auto max_items = max_items_from_env();
    std::vector<benchmark_case> cases;

    for (const auto bucket_count : benchmark_bucket_counts) {
        for (const auto items_per_bucket : benchmark_items_per_bucket) {
            const auto item_count = bucket_count * items_per_bucket;
            if (item_count > max_items) {
                continue;
            }

            cases.push_back(benchmark_case{
                bucket_count,
                items_per_bucket,
                item_count,
            });
        }
    }

    if (cases.empty()) {
        throw std::runtime_error("no benchmark cases remain under FASTPQ_BENCH_MAX_ITEMS");
    }

    return cases;
}

[[nodiscard]] double speedup_vs_std_heap(const benchmark_result& result, std::span<const benchmark_result> results) {
    const auto baseline = std::ranges::find_if(results, [&](const benchmark_result& candidate) {
        return candidate.workload == result.workload && candidate.bucket_count == result.bucket_count &&
               candidate.items_per_bucket == result.items_per_bucket && candidate.item_count == result.item_count &&
               candidate.active_priorities == result.active_priorities && candidate.container == "stdlib_heap";
    });
    if (baseline == results.end()) {
        throw std::runtime_error("missing stdlib_heap baseline");
    }

    return baseline->ns_per_op / result.ns_per_op;
}

void write_results_csv(const std::filesystem::path& output_path, std::span<const benchmark_result> results) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "language,workload,buckets,items_per_bucket,items,active_priorities,container,unit,ns_per_op,"
              "speedup_vs_std_heap,checksum\n";

    for (const auto& result : results) {
        output << "cpp,"
               << result.workload << ','
               << result.bucket_count << ','
               << result.items_per_bucket << ','
               << result.item_count << ','
               << result.active_priorities << ','
               << result.container << ','
               << result.unit << ','
               << std::fixed << std::setprecision(6) << result.ns_per_op << ','
               << std::fixed << std::setprecision(6) << speedup_vs_std_heap(result, results) << ','
               << result.checksum
               << '\n';
    }
}

void print_results(std::span<const benchmark_result> results) {
    std::cout << "Streamlined priority queue benchmark matrix\n";
    std::cout << "buckets={16,1024,100000}, items_per_bucket={1,100}, sparse={range:1000000,active:16,items:100000}\n";
    std::cout << "baseline container: stdlib_heap\n\n";

    std::cout << std::left
              << std::setw(15) << "workload"
              << std::setw(10) << "buckets"
              << std::setw(18) << "items_per_bucket"
              << std::setw(12) << "items"
              << std::setw(18) << "active_priorities"
              << std::setw(32) << "container"
              << std::setw(14) << "unit"
              << std::setw(14) << "ns/op"
              << "speedup"
              << '\n';

    for (const auto& result : results) {
        std::cout << std::left
                  << std::setw(15) << result.workload
                  << std::setw(10) << result.bucket_count
                  << std::setw(18) << result.items_per_bucket
                  << std::setw(12) << result.item_count
                  << std::setw(18) << result.active_priorities
                  << std::setw(32) << result.container
                  << std::setw(14) << result.unit
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.ns_per_op
                  << std::fixed << std::setprecision(3) << speedup_vs_std_heap(result, results)
                  << '\n';
    }
}

} // namespace

int main() {
    const auto cases = build_benchmark_cases();
    std::vector<benchmark_result> results;

    for (const auto& next_case : cases) {
        dispatch_bucket_count(next_case.bucket_count, [&]<std::size_t BucketCount>() {
            append_fill_drain_results<BucketCount>(next_case, results);
            append_steady_results<BucketCount>(next_case, results);
        });
    }

    append_sparse_results(
        sparse_benchmark_case{
            sparse_priority_range,
            sparse_active_priorities,
            sparse_item_count,
        },
        results
    );

    print_results(results);

    const auto output_path = std::filesystem::current_path() / "benchmark_priority_queue_results.csv";
    write_results_csv(output_path, results);
    std::cout << "\nCSV written to " << output_path.string() << '\n';

    return 0;
}
