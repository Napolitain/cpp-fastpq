#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
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
    std::size_t construct_sparse_items = 2'048;
    std::size_t construct_sparse_rounds = 50;
    std::size_t unique_priority_rounds = 5;
    std::size_t growth_ramp_items = 4'096;
    std::size_t growth_ramp_rounds = 50;
};

struct benchmark_result {
    std::string workload;
    std::string variant;
    std::size_t bucket_count;
    std::size_t operations;
    double seconds;
    std::uint64_t checksum;
};

struct gain_row {
    std::string workload;
    std::size_t bucket_count;
    std::string variant;
    double baseline_ns_per_op;
    double variant_ns_per_op;
    double delta_ns_per_op;
    double percent_gain;
    double speedup;
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

class exact_growth_dynamic_adapter {
public:
    explicit exact_growth_dynamic_adapter(std::size_t bucket_count = 0)
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
    cpp_pq::dynamic_bucket_priority_queue_exact_growth<std::uint64_t> queue_;
};

class default_dynamic_adapter {
public:
    explicit default_dynamic_adapter(std::size_t bucket_count = 0)
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

class paged_dynamic_adapter {
public:
    explicit paged_dynamic_adapter(std::size_t bucket_count = 0)
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

class paged_geometric_dynamic_adapter {
public:
    explicit paged_geometric_dynamic_adapter(std::size_t bucket_count = 0)
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
    cpp_pq::paged_dynamic_bucket_priority_queue_geometric<std::uint64_t> queue_;
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
    std::string variant_name,
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
        std::move(variant_name),
        bucket_count,
        rounds * operations_per_round,
        elapsed,
        checksum,
    };
}

template <typename Factory>
void run_all_variants(
    std::vector<benchmark_result>& results,
    std::string workload_name,
    std::size_t bucket_count,
    std::size_t rounds,
    std::size_t operations_per_round,
    Factory&& factory
) {
    results.push_back(measure(
        workload_name,
        "cpp_pq::dynamic_bucket_priority_queue_exact_growth",
        bucket_count,
        rounds,
        operations_per_round,
        [&]() {
            return factory.template operator()<exact_growth_dynamic_adapter>();
        }
    ));

    results.push_back(measure(
        workload_name,
        "cpp_pq::dynamic_bucket_priority_queue",
        bucket_count,
        rounds,
        operations_per_round,
        [&]() {
            return factory.template operator()<default_dynamic_adapter>();
        }
    ));

    results.push_back(measure(
        workload_name,
        "cpp_pq::paged_dynamic_bucket_priority_queue",
        bucket_count,
        rounds,
        operations_per_round,
        [&]() {
            return factory.template operator()<paged_dynamic_adapter>();
        }
    ));

    results.push_back(measure(
        workload_name,
        "cpp_pq::paged_dynamic_bucket_priority_queue_geometric",
        bucket_count,
        rounds,
        operations_per_round,
        [&]() {
            return factory.template operator()<paged_geometric_dynamic_adapter>();
        }
    ));
}

void run_dense_suite(const benchmark_config& config, std::size_t bucket_count, std::vector<benchmark_result>& results) {
    const auto bulk_items = make_items(config.bulk_items, 0xBADC0FFEULL + bucket_count, bucket_count);
    const auto steady_seed_items = make_items(config.steady_state_size, 0x12345678ULL + bucket_count, bucket_count);
    const auto steady_operation_items = make_items(config.steady_state_operations, 0xCAFEBABELL + bucket_count, bucket_count);

    run_all_variants(
        results,
        "bulk-fill-drain",
        bucket_count,
        config.bulk_rounds,
        config.bulk_items * 3,
        [&]<typename Queue>() {
            Queue queue(bucket_count);
            return run_bulk_round(queue, bulk_items);
        }
    );

    run_all_variants(
        results,
        "steady-state",
        bucket_count,
        config.steady_state_rounds,
        (config.steady_state_size * 3) + (config.steady_state_operations * 3),
        [&]<typename Queue>() {
            Queue queue(bucket_count);
            return run_steady_state_round(queue, steady_seed_items, steady_operation_items);
        }
    );
}

void run_edge_suite(const benchmark_config& config, std::size_t bucket_count, std::vector<benchmark_result>& results) {
    const auto sparse_items = make_items(config.construct_sparse_items, 0x0B5EEDULL + bucket_count, bucket_count);
    const auto unique_priority_items = make_unique_priority_items(bucket_count, 0x51A1EULL + bucket_count);
    const auto growth_ramp_items = make_growth_ramp_items(config.growth_ramp_items, 0x6A0EULL + bucket_count, bucket_count);

    run_all_variants(
        results,
        "construct-sparse",
        bucket_count,
        config.construct_sparse_rounds,
        sparse_items.size() * 3,
        [&]<typename Queue>() {
            Queue queue(bucket_count);
            return run_bulk_round(queue, sparse_items);
        }
    );

    run_all_variants(
        results,
        "unique-priority",
        bucket_count,
        config.unique_priority_rounds,
        unique_priority_items.size() * 3,
        [&]<typename Queue>() {
            Queue queue(bucket_count);
            return run_bulk_round(queue, unique_priority_items);
        }
    );

    run_all_variants(
        results,
        "growth-ramp",
        bucket_count,
        config.growth_ramp_rounds,
        growth_ramp_items.size() * 3,
        [&]<typename Queue>() {
            Queue queue;
            return run_bulk_round(queue, growth_ramp_items);
        }
    );
}

[[nodiscard]] double ns_per_operation(const benchmark_result& result) {
    return (result.seconds * 1'000'000'000.0) / static_cast<double>(result.operations);
}

std::vector<gain_row> compute_gains(std::span<const benchmark_result> results) {
    std::unordered_map<std::string, benchmark_result> baselines;
    std::vector<gain_row> gains;

    for (const auto& result : results) {
        const auto key = result.workload + "|" + std::to_string(result.bucket_count);

        if (result.variant == "cpp_pq::dynamic_bucket_priority_queue_exact_growth") {
            baselines.insert_or_assign(key, result);
        }
    }

    for (const auto& result : results) {
        if (result.variant == "cpp_pq::dynamic_bucket_priority_queue_exact_growth") {
            continue;
        }

        const auto key = result.workload + "|" + std::to_string(result.bucket_count);
        const auto baseline_it = baselines.find(key);
        if (baseline_it == baselines.end()) {
            continue;
        }

        const auto baseline_ns = ns_per_operation(baseline_it->second);
        const auto variant_ns = ns_per_operation(result);
        const auto delta_ns = baseline_ns - variant_ns;

        gains.push_back(gain_row{
            result.workload,
            result.bucket_count,
            result.variant,
            baseline_ns,
            variant_ns,
            delta_ns,
            (delta_ns / baseline_ns) * 100.0,
            baseline_ns / variant_ns,
        });
    }

    return gains;
}

void write_gains_csv(const std::filesystem::path& output_path, std::span<const gain_row> gains) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "workload,buckets,variant,baseline_ns_per_op,variant_ns_per_op,delta_ns_per_op,percent_gain,speedup\n";

    for (const auto& gain : gains) {
        output << gain.workload << ','
               << gain.bucket_count << ','
               << gain.variant << ','
               << std::fixed << std::setprecision(6) << gain.baseline_ns_per_op << ','
               << std::fixed << std::setprecision(6) << gain.variant_ns_per_op << ','
               << std::fixed << std::setprecision(6) << gain.delta_ns_per_op << ','
               << std::fixed << std::setprecision(6) << gain.percent_gain << ','
               << std::fixed << std::setprecision(6) << gain.speedup
               << '\n';
    }
}

void print_results(const benchmark_config& config, std::span<const benchmark_result> results) {
    std::cout << "Dynamic priority queue variant benchmark\n";
    std::cout << "bulk_items=" << config.bulk_items
              << ", bulk_rounds=" << config.bulk_rounds
              << ", steady_state_size=" << config.steady_state_size
              << ", steady_state_operations=" << config.steady_state_operations
              << ", steady_state_rounds=" << config.steady_state_rounds
              << ", construct_sparse_items=" << config.construct_sparse_items
              << ", growth_ramp_items=" << config.growth_ramp_items << "\n\n";

    std::cout << std::left
              << std::setw(18) << "workload"
              << std::setw(10) << "buckets"
              << std::setw(47) << "variant"
              << std::setw(14) << "total ms"
              << std::setw(14) << "ns/op"
              << "checksum"
              << '\n';

    for (const auto& result : results) {
        std::cout << std::left
                  << std::setw(18) << result.workload
                  << std::setw(10) << result.bucket_count
                  << std::setw(47) << result.variant
                  << std::setw(14) << std::fixed << std::setprecision(3) << (result.seconds * 1'000.0)
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_per_operation(result)
                  << result.checksum
                  << '\n';
    }
}

void print_gains(std::span<const gain_row> gains) {
    std::cout << "\nGains vs cpp_pq::dynamic_bucket_priority_queue_exact_growth\n";
    std::cout << std::left
              << std::setw(18) << "workload"
              << std::setw(10) << "buckets"
              << std::setw(47) << "variant"
              << std::setw(14) << "delta ns/op"
              << std::setw(14) << "% gain"
              << "speedup"
              << '\n';

    for (const auto& gain : gains) {
        std::cout << std::left
                  << std::setw(18) << gain.workload
                  << std::setw(10) << gain.bucket_count
                  << std::setw(47) << gain.variant
                  << std::setw(14) << std::fixed << std::setprecision(3) << gain.delta_ns_per_op
                  << std::setw(14) << std::fixed << std::setprecision(3) << gain.percent_gain
                  << gain.speedup
                  << '\n';
    }
}

} // namespace

int main() {
    const benchmark_config config{};
    std::vector<benchmark_result> results;
    results.reserve(52);

    for (const auto bucket_count : {std::size_t{16}, std::size_t{64}, std::size_t{256}, std::size_t{1024}, std::size_t{100000}}) {
        run_dense_suite(config, bucket_count, results);
    }

    run_edge_suite(config, 100000, results);

    const auto gains = compute_gains(results);
    const auto output_path = std::filesystem::current_path() / "dynamic_variant_gains.csv";
    write_gains_csv(output_path, gains);

    print_results(config, results);
    print_gains(gains);
    std::cout << "\nCSV written to " << output_path.string() << '\n';
    return 0;
}
