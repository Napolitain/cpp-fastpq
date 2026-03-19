#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "cpp_pq/bucket_priority_queue.hpp"

namespace {

struct item {
    std::size_t priority;
    std::uint64_t value;
};

struct benchmark_config {
    std::size_t bulk_items = 100'000;
    std::size_t bulk_rounds = 10;
    std::size_t steady_state_size = 4'096;
    std::size_t steady_state_operations = 100'000;
    std::size_t steady_state_rounds = 10;
    std::size_t edge_bucket_count = 100'000;
    std::size_t construct_sparse_items = 2'048;
    std::size_t construct_sparse_rounds = 50;
    std::size_t unique_priority_rounds = 5;
    std::size_t growth_ramp_items = 4'096;
    std::size_t growth_ramp_rounds = 50;
};

struct benchmark_result {
    std::string workload;
    std::string container;
    std::size_t bucket_count;
    std::size_t operations;
    double seconds;
    std::uint64_t checksum;
};

using clock_type = std::chrono::steady_clock;

std::uint64_t mix_checksum(std::uint64_t checksum, std::uint64_t value) {
    checksum ^= value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
    return checksum;
}

std::vector<item> make_items(std::size_t count, std::uint64_t seed, std::size_t bucket_count) {
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<std::size_t> priority_distribution(0, bucket_count - 1);

    std::vector<item> items;
    items.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        items.push_back(item{
            priority_distribution(generator),
            generator(),
        });
    }

    return items;
}

std::vector<item> make_unique_priority_items(std::size_t bucket_count, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::vector<item> items;
    items.reserve(bucket_count);

    for (std::size_t priority = 0; priority < bucket_count; ++priority) {
        items.push_back(item{
            priority,
            generator(),
        });
    }

    std::shuffle(items.begin(), items.end(), generator);
    return items;
}

std::vector<item> make_growth_ramp_items(std::size_t count, std::uint64_t seed, std::size_t bucket_count) {
    std::mt19937_64 generator(seed);

    if (count == 0 || bucket_count == 0) {
        return {};
    }

    const auto item_count = std::min(count, bucket_count);
    std::vector<item> items;
    items.reserve(item_count);

    if (item_count == 1) {
        items.push_back(item{
            bucket_count - 1,
            generator(),
        });
        return items;
    }

    for (std::size_t index = 0; index < item_count; ++index) {
        const auto priority = (index * (bucket_count - 1)) / (item_count - 1);
        items.push_back(item{
            priority,
            generator(),
        });
    }

    return items;
}

template <std::size_t BucketCount>
class static_bucket_queue_adapter {
public:
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
    cpp_pq::static_bucket_priority_queue<std::uint64_t, BucketCount> queue_;
};

class dynamic_bucket_queue_adapter {
public:
    explicit dynamic_bucket_queue_adapter(std::size_t bucket_count = 0)
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

class registered_bucket_queue_adapter {
public:
    struct registered_item {
        cpp_pq::registered_bucket_priority_queue<std::uint64_t>::priority_handle handle;
        std::uint64_t value;
    };

    explicit registered_bucket_queue_adapter(std::span<const item> items) {
        registrations_.reserve(items.size());

        for (const auto& next_item : items) {
            if (registrations_.find(next_item.priority) == registrations_.end()) {
                registrations_.emplace(next_item.priority, queue_.register_priority(next_item.priority));
            }
        }
    }

    [[nodiscard]] std::vector<registered_item> bind_items(std::span<const item> items) const {
        std::vector<registered_item> bound_items;
        bound_items.reserve(items.size());

        for (const auto& next_item : items) {
            bound_items.push_back(registered_item{
                registrations_.at(next_item.priority),
                next_item.value,
            });
        }

        return bound_items;
    }

    void push(const registered_item& next_item) {
        queue_.push(next_item.handle, next_item.value);
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
    cpp_pq::registered_bucket_priority_queue<std::uint64_t> queue_{};
    std::unordered_map<std::size_t, cpp_pq::registered_bucket_priority_queue<std::uint64_t>::priority_handle> registrations_{};
};

struct priority_entry {
    std::size_t priority;
    std::uint64_t value;
    std::uint64_t sequence;
};

struct higher_priority {
    [[nodiscard]] bool operator()(const priority_entry& left, const priority_entry& right) const noexcept {
        return std::tie(left.priority, left.sequence) < std::tie(right.priority, right.sequence);
    }
};

class std_priority_queue_adapter {
public:
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
    std::priority_queue<priority_entry, std::vector<priority_entry>, higher_priority> queue_{};
    std::uint64_t next_sequence_{0};
};

template <typename Queue, typename Range>
std::uint64_t run_bulk_round(Queue& queue, const Range& items) {
    for (const auto& next_item : items) {
        queue.push(next_item);
    }

    std::uint64_t checksum = 0;
    while (!queue.empty()) {
        checksum = mix_checksum(checksum, queue.top_value());
        queue.pop();
    }

    return checksum;
}

template <typename Queue, typename SeedRange, typename NextRange>
std::uint64_t run_steady_state_round(Queue& queue, const SeedRange& seed_items, const NextRange& next_items) {
    for (const auto& next_item : seed_items) {
        queue.push(next_item);
    }

    std::uint64_t checksum = 0;
    for (const auto& next_item : next_items) {
        checksum = mix_checksum(checksum, queue.top_value());
        queue.pop();
        queue.push(next_item);
    }

    while (!queue.empty()) {
        checksum = mix_checksum(checksum, queue.top_value());
        queue.pop();
    }

    return checksum;
}

template <typename Workload>
benchmark_result measure(
    std::string workload_name,
    std::string container_name,
    std::size_t bucket_count,
    std::size_t rounds,
    std::size_t operations_per_round,
    Workload&& workload
) {
    [[maybe_unused]] const auto warmup_checksum = workload();

    std::uint64_t checksum = 0;
    const auto start = clock_type::now();

    for (std::size_t round = 0; round < rounds; ++round) {
        checksum = mix_checksum(checksum, workload());
    }

    const auto finish = clock_type::now();
    const auto elapsed = std::chrono::duration<double>(finish - start).count();

    return benchmark_result{
        std::move(workload_name),
        std::move(container_name),
        bucket_count,
        rounds * operations_per_round,
        elapsed,
        checksum,
    };
}

template <std::size_t BucketCount>
void run_suite(const benchmark_config& config, std::vector<benchmark_result>& results) {
    const auto bulk_items = make_items(config.bulk_items, 0xBADC0FFEULL + BucketCount, BucketCount);
    const auto steady_seed_items = make_items(config.steady_state_size, 0x12345678ULL + BucketCount, BucketCount);
    const auto steady_operation_items = make_items(config.steady_state_operations, 0xCAFEBABELL + BucketCount, BucketCount);
    auto steady_registration_items = steady_seed_items;
    steady_registration_items.insert(steady_registration_items.end(), steady_operation_items.begin(), steady_operation_items.end());
    registered_bucket_queue_adapter registered_bulk_queue(bulk_items);
    const auto registered_bulk_items = registered_bulk_queue.bind_items(bulk_items);
    registered_bucket_queue_adapter registered_steady_queue(steady_registration_items);
    const auto registered_steady_seed_items = registered_steady_queue.bind_items(steady_seed_items);
    const auto registered_steady_operation_items = registered_steady_queue.bind_items(steady_operation_items);

    const auto bulk_operations = config.bulk_items * 3;
    const auto steady_state_operations = (config.steady_state_size * 3) + (config.steady_state_operations * 3);

    {
        std_priority_queue_adapter queue;
        results.push_back(measure(
            "bulk-fill-drain",
            "std::priority_queue",
            BucketCount,
            config.bulk_rounds,
            bulk_operations,
            [&]() {
                return run_bulk_round(queue, bulk_items);
            }
        ));
    }

    {
        auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
        results.push_back(measure(
            "bulk-fill-drain",
            "cpp_pq::static_bucket_priority_queue",
            BucketCount,
            config.bulk_rounds,
            bulk_operations,
            [&]() {
                return run_bulk_round(*queue, bulk_items);
            }
        ));
    }

    {
        dynamic_bucket_queue_adapter queue(BucketCount);
        results.push_back(measure(
            "bulk-fill-drain",
            "cpp_pq::dynamic_bucket_priority_queue",
            BucketCount,
            config.bulk_rounds,
            bulk_operations,
            [&]() {
                return run_bulk_round(queue, bulk_items);
            }
        ));
    }

    {
        results.push_back(measure(
            "bulk-fill-drain",
            "cpp_pq::registered_bucket_priority_queue",
            BucketCount,
            config.bulk_rounds,
            bulk_operations,
            [&]() {
                return run_bulk_round(registered_bulk_queue, registered_bulk_items);
            }
        ));
    }

    {
        std_priority_queue_adapter queue;
        results.push_back(measure(
            "steady-state",
            "std::priority_queue",
            BucketCount,
            config.steady_state_rounds,
            steady_state_operations,
            [&]() {
                return run_steady_state_round(queue, steady_seed_items, steady_operation_items);
            }
        ));
    }

    {
        auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
        results.push_back(measure(
            "steady-state",
            "cpp_pq::static_bucket_priority_queue",
            BucketCount,
            config.steady_state_rounds,
            steady_state_operations,
            [&]() {
                return run_steady_state_round(*queue, steady_seed_items, steady_operation_items);
            }
        ));
    }

    {
        dynamic_bucket_queue_adapter queue(BucketCount);
        results.push_back(measure(
            "steady-state",
            "cpp_pq::dynamic_bucket_priority_queue",
            BucketCount,
            config.steady_state_rounds,
            steady_state_operations,
            [&]() {
                return run_steady_state_round(queue, steady_seed_items, steady_operation_items);
            }
        ));
    }

    {
        results.push_back(measure(
            "steady-state",
            "cpp_pq::registered_bucket_priority_queue",
            BucketCount,
            config.steady_state_rounds,
            steady_state_operations,
            [&]() {
                return run_steady_state_round(registered_steady_queue, registered_steady_seed_items, registered_steady_operation_items);
            }
        ));
    }
}

template <std::size_t BucketCount>
void run_edge_case_suite(const benchmark_config& config, std::vector<benchmark_result>& results) {
    const auto sparse_items = make_items(config.construct_sparse_items, 0x0B5EEDULL + BucketCount, BucketCount);
    const auto unique_priority_items = make_unique_priority_items(BucketCount, 0x51A1EULL + BucketCount);
    const auto growth_ramp_items = make_growth_ramp_items(config.growth_ramp_items, 0x6A0EULL + BucketCount, BucketCount);
    registered_bucket_queue_adapter sparse_registered_queue(sparse_items);
    const auto sparse_registered_items = sparse_registered_queue.bind_items(sparse_items);
    registered_bucket_queue_adapter unique_registered_queue(unique_priority_items);
    const auto unique_registered_items = unique_registered_queue.bind_items(unique_priority_items);
    registered_bucket_queue_adapter growth_registered_queue(growth_ramp_items);
    const auto growth_registered_items = growth_registered_queue.bind_items(growth_ramp_items);

    {
        const auto operations_per_round = sparse_items.size() * 3;

        results.push_back(measure(
            "construct-sparse",
            "std::priority_queue",
            BucketCount,
            config.construct_sparse_rounds,
            operations_per_round,
            [&]() {
                std_priority_queue_adapter queue;
                return run_bulk_round(queue, sparse_items);
            }
        ));

        results.push_back(measure(
            "construct-sparse",
            "cpp_pq::static_bucket_priority_queue",
            BucketCount,
            config.construct_sparse_rounds,
            operations_per_round,
            [&]() {
                auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
                return run_bulk_round(*queue, sparse_items);
            }
        ));

        results.push_back(measure(
            "construct-sparse",
            "cpp_pq::dynamic_bucket_priority_queue",
            BucketCount,
            config.construct_sparse_rounds,
            operations_per_round,
            [&]() {
                dynamic_bucket_queue_adapter queue(BucketCount);
                return run_bulk_round(queue, sparse_items);
            }
        ));

        results.push_back(measure(
            "construct-sparse",
            "cpp_pq::registered_bucket_priority_queue",
            BucketCount,
            config.construct_sparse_rounds,
            operations_per_round,
            [&]() {
                return run_bulk_round(sparse_registered_queue, sparse_registered_items);
            }
        ));
    }

    {
        const auto operations_per_round = unique_priority_items.size() * 3;

        results.push_back(measure(
            "unique-priority",
            "std::priority_queue",
            BucketCount,
            config.unique_priority_rounds,
            operations_per_round,
            [&]() {
                std_priority_queue_adapter queue;
                return run_bulk_round(queue, unique_priority_items);
            }
        ));

        results.push_back(measure(
            "unique-priority",
            "cpp_pq::static_bucket_priority_queue",
            BucketCount,
            config.unique_priority_rounds,
            operations_per_round,
            [&]() {
                auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
                return run_bulk_round(*queue, unique_priority_items);
            }
        ));

        results.push_back(measure(
            "unique-priority",
            "cpp_pq::dynamic_bucket_priority_queue",
            BucketCount,
            config.unique_priority_rounds,
            operations_per_round,
            [&]() {
                dynamic_bucket_queue_adapter queue(BucketCount);
                return run_bulk_round(queue, unique_priority_items);
            }
        ));

        results.push_back(measure(
            "unique-priority",
            "cpp_pq::registered_bucket_priority_queue",
            BucketCount,
            config.unique_priority_rounds,
            operations_per_round,
            [&]() {
                return run_bulk_round(unique_registered_queue, unique_registered_items);
            }
        ));
    }

    {
        const auto operations_per_round = growth_ramp_items.size() * 3;

        results.push_back(measure(
            "growth-ramp",
            "std::priority_queue",
            BucketCount,
            config.growth_ramp_rounds,
            operations_per_round,
            [&]() {
                std_priority_queue_adapter queue;
                return run_bulk_round(queue, growth_ramp_items);
            }
        ));

        results.push_back(measure(
            "growth-ramp",
            "cpp_pq::static_bucket_priority_queue",
            BucketCount,
            config.growth_ramp_rounds,
            operations_per_round,
            [&]() {
                auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
                return run_bulk_round(*queue, growth_ramp_items);
            }
        ));

        results.push_back(measure(
            "growth-ramp",
            "cpp_pq::dynamic_bucket_priority_queue",
            BucketCount,
            config.growth_ramp_rounds,
            operations_per_round,
            [&]() {
                dynamic_bucket_queue_adapter queue;
                return run_bulk_round(queue, growth_ramp_items);
            }
        ));

        results.push_back(measure(
            "growth-ramp",
            "cpp_pq::registered_bucket_priority_queue",
            BucketCount,
            config.growth_ramp_rounds,
            operations_per_round,
            [&]() {
                return run_bulk_round(growth_registered_queue, growth_registered_items);
            }
        ));
    }
}

void print_results(const benchmark_config& config, std::span<const benchmark_result> results) {
    std::cout << "Bounded-priority queue microbenchmark\n";
    std::cout << "bulk_items=" << config.bulk_items
              << ", bulk_rounds=" << config.bulk_rounds
              << ", steady_state_size=" << config.steady_state_size
              << ", steady_state_operations=" << config.steady_state_operations
              << ", steady_state_rounds=" << config.steady_state_rounds
              << ", edge_bucket_count=" << config.edge_bucket_count
              << ", construct_sparse_items=" << config.construct_sparse_items
              << ", unique_priority_items=" << config.edge_bucket_count
              << ", growth_ramp_items=" << config.growth_ramp_items << "\n\n";

    std::cout << std::left
              << std::setw(18) << "workload"
              << std::setw(10) << "buckets"
              << std::setw(45) << "container"
              << std::setw(14) << "total ms"
              << std::setw(14) << "ns/op"
              << "checksum"
              << '\n';

    for (const auto& result : results) {
        const auto total_ms = result.seconds * 1'000.0;
        const auto ns_per_operation = (result.seconds * 1'000'000'000.0) / static_cast<double>(result.operations);

        std::cout << std::left
                  << std::setw(18) << result.workload
                  << std::setw(10) << result.bucket_count
                  << std::setw(45) << result.container
                  << std::setw(14) << std::fixed << std::setprecision(3) << total_ms
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_per_operation
                  << result.checksum
                  << '\n';
    }
}

} // namespace

int main() {
    const benchmark_config config{};
    std::vector<benchmark_result> results;
    results.reserve(64);

    run_suite<16>(config, results);
    run_suite<64>(config, results);
    run_suite<256>(config, results);
    run_suite<1024>(config, results);
    run_suite<100000>(config, results);
    run_edge_case_suite<100000>(config, results);

    print_results(config, results);
    return 0;
}
