#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
#include <folly/FBVector.h>
#endif

#include "cpp_pq/bucket_priority_queue.hpp"

namespace {

struct item {
    std::size_t priority;
    std::uint64_t value;
};

enum class range_mode : std::uint8_t {
    known,
    growth,
};

enum class workload_kind : std::uint8_t {
    bulk_uniform,
    steady_uniform,
    steady_hot_set,
    bulk_sparse_reused,
    unique_priority,
    growth_ramp,
};

enum class frontier_pattern : std::uint8_t {
    strided,
    page_clustered,
    random,
};

struct ratio_case {
    std::string_view label;
    std::size_t numerator;
    std::size_t denominator;
};

constexpr auto core_ratio_cases = std::to_array<ratio_case>({
    {"dense", 1, 8},
    {"balanced", 1, 1},
    {"sparse", 8, 1},
});

constexpr auto growth_ramp_ratio_cases = std::to_array<ratio_case>({
    {"balanced", 1, 1},
    {"sparse", 8, 1},
    {"very-sparse", 64, 1},
});

constexpr std::size_t min_bucket_count = 16;
constexpr std::size_t max_bucket_count = 1'000'000;

struct benchmark_config {
    std::array<std::size_t, 4> bulk_item_counts{1'000, 10'000, 100'000, 1'000'000};
    std::array<std::size_t, 3> steady_live_counts{1'000, 10'000, 100'000};
    std::array<std::size_t, 3> edge_item_counts{1'000, 10'000, 100'000};
    std::array<std::size_t, 5> focused_bucket_counts{16, 64, 256, 1'024, 100'000};
    std::size_t target_total_operations = 1'000'000;
    std::size_t target_edge_operations = 750'000;
    std::size_t max_rounds = 512;
    std::size_t steady_state_multiplier = 10;
    std::size_t steady_state_min_operations = 10'000;
    std::size_t steady_state_max_operations = 1'000'000;
    std::size_t sparse_reused_priority_count = 16;
    std::size_t hot_set_priority_count = 8;
    double hot_set_probability = 0.90;
    std::size_t focused_bulk_items = 100'000;
    std::size_t focused_bulk_rounds = 10;
    std::size_t focused_steady_live_items = 4'096;
    std::size_t focused_steady_operations = 100'000;
    std::size_t focused_steady_rounds = 10;
    std::size_t construct_sparse_items = 2'048;
    std::size_t construct_sparse_bucket_count = 100'000;
    std::size_t construct_sparse_rounds = 50;
    std::size_t focused_unique_priority_items = 100'000;
    std::size_t focused_unique_priority_rounds = 5;
    std::size_t focused_growth_ramp_items = 4'096;
    std::size_t focused_growth_ramp_bucket_count = 100'000;
    std::size_t focused_growth_ramp_rounds = 50;
    double compact_case_weight = 8.0;
    std::array<std::size_t, 2> frontier_item_counts{10'000, 100'000};
    std::array<std::size_t, 8> frontier_gap_factors{1, 2, 4, 8, 16, 64, 256, 1'024};
    std::size_t frontier_items_per_used_priority = 100;
};

struct benchmark_case {
    range_mode mode;
    workload_kind kind;
    std::string workload;
    std::string shape;
    std::size_t item_count;
    std::size_t bucket_count;
    std::size_t steady_state_operations;
    std::size_t rounds;
    std::uint64_t seed_tag;
    double score_weight{1.0};
};

struct benchmark_result {
    std::string range;
    std::string workload;
    std::string shape;
    std::size_t item_count;
    std::size_t bucket_count;
    std::string container;
    std::size_t operations;
    double seconds;
    std::uint64_t checksum;
    double score_weight;
};

struct gain_row {
    std::string range;
    std::string workload;
    std::string shape;
    std::size_t item_count;
    std::size_t bucket_count;
    std::string container;
    double baseline_ns_per_op;
    double variant_ns_per_op;
    double delta_ns_per_op;
    double percent_gain;
    double speedup;
};

struct score_row {
    std::string range;
    std::string workload;
    std::string shape;
    std::size_t item_count;
    std::size_t bucket_count;
    std::string container;
    double std_ns_per_op;
    double variant_ns_per_op;
    double score;
    double case_weight;
};

struct score_summary_row {
    std::string container;
    double weighted_score;
    double total_weight;
    std::size_t case_count;
};

struct frontier_benchmark_case {
    range_mode mode;
    std::string workload;
    frontier_pattern pattern;
    std::size_t item_count;
    std::size_t active_priority_count;
    std::size_t bucket_count;
    std::size_t gap_factor;
    std::size_t steady_state_operations;
    std::size_t rounds;
    std::uint64_t seed_tag;
};

struct frontier_result {
    std::string range;
    std::string workload;
    std::string pattern;
    std::size_t item_count;
    std::size_t active_priority_count;
    std::size_t bucket_count;
    std::size_t gap_factor;
    std::string container;
    std::size_t operations;
    double seconds;
    std::uint64_t checksum;
};

struct frontier_score_row {
    std::string range;
    std::string workload;
    std::string pattern;
    std::size_t item_count;
    std::size_t active_priority_count;
    std::size_t bucket_count;
    std::size_t gap_factor;
    std::string container;
    double baseline_ns_per_op;
    double variant_ns_per_op;
    double score;
    double case_weight;
};

struct frontier_gap_summary_row {
    std::size_t gap_factor;
    std::string container;
    double average_score;
    std::size_t case_count;
};

using clock_type = std::chrono::steady_clock;

[[nodiscard]] std::string_view range_mode_name(range_mode mode) noexcept {
    switch (mode) {
    case range_mode::known:
        return "known-range";
    case range_mode::growth:
        return "growth-range";
    }

    return "unknown";
}

[[nodiscard]] std::string_view frontier_pattern_name(frontier_pattern pattern) noexcept {
    switch (pattern) {
    case frontier_pattern::strided:
        return "strided";
    case frontier_pattern::page_clustered:
        return "page-clustered";
    case frontier_pattern::random:
        return "random";
    }

    return "unknown";
}

std::uint64_t mix_checksum(std::uint64_t checksum, std::uint64_t value) {
    checksum ^= value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
    return checksum;
}

[[nodiscard]] std::size_t clamp_bucket_count(std::size_t bucket_count) noexcept {
    return std::clamp(bucket_count, min_bucket_count, max_bucket_count);
}

[[nodiscard]] std::size_t bucket_count_for_ratio(std::size_t items, std::size_t numerator, std::size_t denominator) noexcept {
    const auto requested = (items * numerator + denominator - 1) / denominator;
    return clamp_bucket_count(requested);
}

[[nodiscard]] std::size_t scaled_rounds(
    std::size_t operations_per_round,
    std::size_t target_total_operations,
    std::size_t max_rounds
) noexcept {
    if (operations_per_round == 0) {
        return 1;
    }

    const auto scaled = std::max<std::size_t>(target_total_operations / operations_per_round, 1);
    return std::min(scaled, max_rounds);
}

[[nodiscard]] std::size_t steady_state_operation_count(const benchmark_config& config, std::size_t live_items) noexcept {
    const auto scaled = live_items * config.steady_state_multiplier;
    return std::clamp(scaled, config.steady_state_min_operations, config.steady_state_max_operations);
}

[[nodiscard]] std::uint64_t case_seed(const benchmark_case& next_case, std::uint64_t salt) noexcept {
    auto seed = salt;
    seed = mix_checksum(seed, static_cast<std::uint64_t>(next_case.mode));
    seed = mix_checksum(seed, static_cast<std::uint64_t>(next_case.kind));
    seed = mix_checksum(seed, next_case.item_count);
    seed = mix_checksum(seed, next_case.bucket_count);
    seed = mix_checksum(seed, next_case.steady_state_operations);
    seed = mix_checksum(seed, next_case.seed_tag);
    return seed;
}

[[nodiscard]] std::uint64_t frontier_case_seed(const frontier_benchmark_case& next_case, std::uint64_t salt) noexcept {
    auto seed = salt;
    seed = mix_checksum(seed, static_cast<std::uint64_t>(next_case.mode));
    seed = mix_checksum(seed, static_cast<std::uint64_t>(next_case.pattern));
    seed = mix_checksum(seed, next_case.item_count);
    seed = mix_checksum(seed, next_case.active_priority_count);
    seed = mix_checksum(seed, next_case.bucket_count);
    seed = mix_checksum(seed, next_case.gap_factor);
    seed = mix_checksum(seed, next_case.steady_state_operations);
    seed = mix_checksum(seed, next_case.seed_tag);
    return seed;
}

std::vector<std::size_t> make_sparse_priority_set(std::size_t bucket_count, std::size_t distinct_priority_count) {
    if (bucket_count == 0 || distinct_priority_count == 0) {
        return {};
    }

    const auto actual_count = std::min(bucket_count, distinct_priority_count);
    std::vector<std::size_t> priorities;
    priorities.reserve(actual_count);

    if (actual_count == 1) {
        priorities.push_back(bucket_count - 1);
        return priorities;
    }

    for (std::size_t index = 0; index < actual_count; ++index) {
        priorities.push_back((index * (bucket_count - 1)) / (actual_count - 1));
    }

    return priorities;
}

std::vector<item> make_uniform_items(std::size_t count, std::uint64_t seed, std::size_t bucket_count) {
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

std::vector<item> make_hot_set_items(
    std::size_t count,
    std::uint64_t seed,
    std::size_t bucket_count,
    std::size_t hot_priority_count,
    double hot_probability
) {
    std::mt19937_64 generator(seed);
    std::bernoulli_distribution use_hot_priority(hot_probability);
    std::uniform_int_distribution<std::size_t> full_distribution(0, bucket_count - 1);

    const auto actual_hot_priority_count = std::min(bucket_count, std::max<std::size_t>(hot_priority_count, 1));
    const auto hot_priority_floor = bucket_count - actual_hot_priority_count;
    std::uniform_int_distribution<std::size_t> hot_distribution(hot_priority_floor, bucket_count - 1);

    std::vector<item> items;
    items.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        const auto priority = use_hot_priority(generator) ? hot_distribution(generator) : full_distribution(generator);
        items.push_back(item{
            priority,
            generator(),
        });
    }

    return items;
}

std::vector<item> make_sparse_reused_items(
    std::size_t count,
    std::uint64_t seed,
    std::size_t bucket_count,
    std::size_t distinct_priority_count
) {
    std::mt19937_64 generator(seed);
    const auto priorities = make_sparse_priority_set(bucket_count, distinct_priority_count);
    std::uniform_int_distribution<std::size_t> priority_distribution(0, priorities.size() - 1);

    std::vector<item> items;
    items.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        items.push_back(item{
            priorities[priority_distribution(generator)],
            generator(),
        });
    }

    return items;
}

std::vector<std::size_t> make_page_clustered_priority_set(std::size_t bucket_count, std::size_t distinct_priority_count) {
    if (bucket_count == 0 || distinct_priority_count == 0) {
        return {};
    }

    const auto actual_count = std::min(bucket_count, distinct_priority_count);
    if (actual_count == 1) {
        return {bucket_count - 1};
    }

    const auto page_count = cpp_pq::detail::words_for_bits(bucket_count);
    const auto required_page_count = std::min(page_count, cpp_pq::detail::words_for_bits(actual_count));

    std::vector<std::size_t> page_indices;
    page_indices.reserve(required_page_count);
    if (required_page_count == 1) {
        page_indices.push_back(page_count - 1);
    } else {
        for (std::size_t index = 0; index < required_page_count; ++index) {
            page_indices.push_back((index * (page_count - 1)) / (required_page_count - 1));
        }
    }

    std::vector<std::size_t> priorities;
    priorities.reserve(actual_count);
    std::size_t remaining = actual_count;

    for (std::size_t page_slot = 0; page_slot < page_indices.size(); ++page_slot) {
        const auto page_index = page_indices[page_slot];
        const auto page_start = page_index * cpp_pq::detail::occupancy_word_bits;
        const auto page_end = std::min(bucket_count, page_start + cpp_pq::detail::occupancy_word_bits);
        const auto page_capacity = page_end - page_start;
        const auto pages_left = page_indices.size() - page_slot;
        const auto page_take = std::min(page_capacity, (remaining + pages_left - 1) / pages_left);

        if (page_slot + 1 == page_indices.size()) {
            for (std::size_t offset = 0; offset < page_take; ++offset) {
                priorities.push_back(page_end - page_take + offset);
            }
        } else {
            for (std::size_t offset = 0; offset < page_take; ++offset) {
                priorities.push_back(page_start + offset);
            }
        }

        remaining -= page_take;
    }

    return priorities;
}

std::vector<std::size_t> make_random_priority_set(std::size_t bucket_count, std::size_t distinct_priority_count, std::uint64_t seed) {
    if (bucket_count == 0 || distinct_priority_count == 0) {
        return {};
    }

    const auto actual_count = std::min(bucket_count, distinct_priority_count);
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<std::size_t> distribution(0, bucket_count - 1);

    std::vector<std::size_t> priorities;
    priorities.reserve(actual_count);
    priorities.push_back(bucket_count - 1);

    std::unordered_set<std::size_t> seen;
    seen.reserve(actual_count * 2);
    seen.insert(bucket_count - 1);

    while (priorities.size() < actual_count) {
        const auto priority = distribution(generator);
        if (seen.insert(priority).second) {
            priorities.push_back(priority);
        }
    }

    std::sort(priorities.begin(), priorities.end());
    return priorities;
}

std::vector<std::size_t> make_frontier_priority_set(
    frontier_pattern pattern,
    std::size_t bucket_count,
    std::size_t distinct_priority_count,
    std::uint64_t seed
) {
    switch (pattern) {
    case frontier_pattern::strided:
        return make_sparse_priority_set(bucket_count, distinct_priority_count);
    case frontier_pattern::page_clustered:
        return make_page_clustered_priority_set(bucket_count, distinct_priority_count);
    case frontier_pattern::random:
        return make_random_priority_set(bucket_count, distinct_priority_count, seed);
    }

    return {};
}

std::vector<item> make_items_from_priority_set(std::size_t count, std::uint64_t seed, std::span<const std::size_t> priorities) {
    if (count == 0 || priorities.empty()) {
        return {};
    }

    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<std::size_t> priority_distribution(0, priorities.size() - 1);

    std::vector<item> items;
    items.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        items.push_back(item{
            priorities[priority_distribution(generator)],
            generator(),
        });
    }

    return items;
}

std::vector<item> make_unique_priority_items(std::size_t count, std::uint64_t seed, std::size_t bucket_count) {
    std::mt19937_64 generator(seed);
    const auto priorities = make_sparse_priority_set(bucket_count, count);

    std::vector<item> items;
    items.reserve(priorities.size());

    for (const auto priority : priorities) {
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

class exact_growth_dynamic_bucket_queue_adapter {
public:
    explicit exact_growth_dynamic_bucket_queue_adapter(std::size_t bucket_count = 0)
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

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
class fbvector_dynamic_bucket_queue_adapter {
public:
    explicit fbvector_dynamic_bucket_queue_adapter(std::size_t bucket_count = 0)
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
    cpp_pq::dynamic_bucket_priority_queue_base<std::uint64_t, true, folly::fbvector> queue_;
};
#endif

class paged_dynamic_bucket_queue_adapter {
public:
    explicit paged_dynamic_bucket_queue_adapter(std::size_t bucket_count = 0)
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

class paged_geometric_dynamic_bucket_queue_adapter {
public:
    explicit paged_geometric_dynamic_bucket_queue_adapter(std::size_t bucket_count = 0)
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
    const benchmark_case& next_case,
    std::string container_name,
    std::size_t operations_per_round,
    Workload&& workload
) {
    [[maybe_unused]] const auto warmup_checksum = workload();

    std::uint64_t checksum = 0;
    const auto start = clock_type::now();

    for (std::size_t round = 0; round < next_case.rounds; ++round) {
        checksum = mix_checksum(checksum, workload());
    }

    const auto finish = clock_type::now();
    const auto elapsed = std::chrono::duration<double>(finish - start).count();

    return benchmark_result{
        std::string(range_mode_name(next_case.mode)),
        next_case.workload,
        next_case.shape,
        next_case.item_count,
        next_case.bucket_count,
        std::move(container_name),
        next_case.rounds * operations_per_round,
        elapsed,
        checksum,
        next_case.score_weight,
    };
}

template <typename Workload>
frontier_result measure_frontier(
    const frontier_benchmark_case& next_case,
    std::string container_name,
    std::size_t operations_per_round,
    Workload&& workload
) {
    [[maybe_unused]] const auto warmup_checksum = workload();

    std::uint64_t checksum = 0;
    const auto start = clock_type::now();

    for (std::size_t round = 0; round < next_case.rounds; ++round) {
        checksum = mix_checksum(checksum, workload());
    }

    const auto finish = clock_type::now();
    const auto elapsed = std::chrono::duration<double>(finish - start).count();

    return frontier_result{
        std::string(range_mode_name(next_case.mode)),
        next_case.workload,
        std::string(frontier_pattern_name(next_case.pattern)),
        next_case.item_count,
        next_case.active_priority_count,
        next_case.bucket_count,
        next_case.gap_factor,
        std::move(container_name),
        next_case.rounds * operations_per_round,
        elapsed,
        checksum,
    };
}

template <std::size_t BucketCount>
void benchmark_bulk_case(
    const benchmark_case& next_case,
    std::span<const item> items,
    std::vector<benchmark_result>& results
) {
    const auto operations_per_round = items.size() * 3;

    auto add_result = [&](std::string_view container_name, auto&& workload) {
        results.push_back(measure(
            next_case,
            std::string(container_name),
            operations_per_round,
            std::forward<decltype(workload)>(workload)
        ));
    };

    add_result("std::priority_queue", [&]() {
        std_priority_queue_adapter queue;
        return run_bulk_round(queue, items);
    });

    add_result("cpp_pq::static_bucket_priority_queue", [&]() {
        auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
        return run_bulk_round(*queue, items);
    });

    if (next_case.mode == range_mode::known) {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter queue(BucketCount);
            return run_bulk_round(queue, items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_bulk_round(queue, items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_bulk_round(queue, items);
        });
    } else {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter dynamic_queue;
            return run_bulk_round(dynamic_queue, items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter fbvector_queue;
            return run_bulk_round(fbvector_queue, items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });
    }
}

template <std::size_t BucketCount>
void benchmark_steady_case(
    const benchmark_case& next_case,
    std::span<const item> seed_items,
    std::span<const item> operation_items,
    std::vector<benchmark_result>& results
) {
    const auto operations_per_round = (seed_items.size() * 3) + (operation_items.size() * 3);

    auto add_result = [&](std::string_view container_name, auto&& workload) {
        results.push_back(measure(
            next_case,
            std::string(container_name),
            operations_per_round,
            std::forward<decltype(workload)>(workload)
        ));
    };

    add_result("std::priority_queue", [&]() {
        std_priority_queue_adapter queue;
        return run_steady_state_round(queue, seed_items, operation_items);
    });

    add_result("cpp_pq::static_bucket_priority_queue", [&]() {
        auto queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
        return run_steady_state_round(*queue, seed_items, operation_items);
    });

    if (next_case.mode == range_mode::known) {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter queue(BucketCount);
            return run_steady_state_round(queue, seed_items, operation_items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_steady_state_round(queue, seed_items, operation_items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue(BucketCount);
            return run_steady_state_round(queue, seed_items, operation_items);
        });
    } else {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter dynamic_queue;
            return run_steady_state_round(dynamic_queue, seed_items, operation_items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter fbvector_queue;
            return run_steady_state_round(fbvector_queue, seed_items, operation_items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });
    }
}

void benchmark_frontier_bulk_case(
    const frontier_benchmark_case& next_case,
    std::span<const item> items,
    std::vector<frontier_result>& results
) {
    const auto operations_per_round = items.size() * 3;

    auto add_result = [&](std::string_view container_name, auto&& workload) {
        results.push_back(measure_frontier(
            next_case,
            std::string(container_name),
            operations_per_round,
            std::forward<decltype(workload)>(workload)
        ));
    };

    add_result("std::priority_queue", [&]() {
        std_priority_queue_adapter queue;
        return run_bulk_round(queue, items);
    });

    if (next_case.mode == range_mode::known) {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_bulk_round(queue, items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_bulk_round(queue, items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_bulk_round(queue, items);
        });
    } else {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue;
            return run_bulk_round(queue, items);
        });
    }
}

void benchmark_frontier_steady_case(
    const frontier_benchmark_case& next_case,
    std::span<const item> seed_items,
    std::span<const item> operation_items,
    std::vector<frontier_result>& results
) {
    const auto operations_per_round = (seed_items.size() * 3) + (operation_items.size() * 3);

    auto add_result = [&](std::string_view container_name, auto&& workload) {
        results.push_back(measure_frontier(
            next_case,
            std::string(container_name),
            operations_per_round,
            std::forward<decltype(workload)>(workload)
        ));
    };

    add_result("std::priority_queue", [&]() {
        std_priority_queue_adapter queue;
        return run_steady_state_round(queue, seed_items, operation_items);
    });

    if (next_case.mode == range_mode::known) {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_steady_state_round(queue, seed_items, operation_items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_steady_state_round(queue, seed_items, operation_items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue(next_case.bucket_count);
            return run_steady_state_round(queue, seed_items, operation_items);
        });
    } else {
        add_result("cpp_pq::dynamic_bucket_priority_queue_exact_growth", [&]() {
            exact_growth_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            fbvector_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });
#endif

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue", [&]() {
            paged_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });

        add_result("cpp_pq::paged_dynamic_bucket_priority_queue_geometric", [&]() {
            paged_geometric_dynamic_bucket_queue_adapter queue;
            return run_steady_state_round(queue, seed_items, operation_items);
        });
    }
}

template <std::size_t BucketCount>
void run_case(
    const benchmark_config& config,
    const benchmark_case& next_case,
    std::vector<benchmark_result>& results
) {
    switch (next_case.kind) {
    case workload_kind::bulk_uniform: {
        const auto items = make_uniform_items(next_case.item_count, case_seed(next_case, 0xBADC0FFEULL), BucketCount);
        benchmark_bulk_case<BucketCount>(next_case, items, results);
        return;
    }

    case workload_kind::steady_uniform: {
        const auto seed_items = make_uniform_items(next_case.item_count, case_seed(next_case, 0x12345678ULL), BucketCount);
        const auto operation_items = make_uniform_items(
            next_case.steady_state_operations,
            case_seed(next_case, 0xCAFEBABELL),
            BucketCount
        );
        benchmark_steady_case<BucketCount>(next_case, seed_items, operation_items, results);
        return;
    }

    case workload_kind::steady_hot_set: {
        const auto seed_items = make_hot_set_items(
            next_case.item_count,
            case_seed(next_case, 0xA11CEULL),
            BucketCount,
            config.hot_set_priority_count,
            config.hot_set_probability
        );
        const auto operation_items = make_hot_set_items(
            next_case.steady_state_operations,
            case_seed(next_case, 0xFACEFEEDULL),
            BucketCount,
            config.hot_set_priority_count,
            config.hot_set_probability
        );
        benchmark_steady_case<BucketCount>(next_case, seed_items, operation_items, results);
        return;
    }

    case workload_kind::bulk_sparse_reused: {
        const auto items = make_sparse_reused_items(
            next_case.item_count,
            case_seed(next_case, 0x0B5EEDULL),
            BucketCount,
            config.sparse_reused_priority_count
        );
        benchmark_bulk_case<BucketCount>(next_case, items, results);
        return;
    }

    case workload_kind::unique_priority: {
        const auto items = make_unique_priority_items(next_case.item_count, case_seed(next_case, 0x51A1EULL), BucketCount);
        benchmark_bulk_case<BucketCount>(next_case, items, results);
        return;
    }

    case workload_kind::growth_ramp: {
        const auto items = make_growth_ramp_items(next_case.item_count, case_seed(next_case, 0x6A0EULL), BucketCount);
        benchmark_bulk_case<BucketCount>(next_case, items, results);
        return;
    }
    }
}

template <typename Fn>
void dispatch_bucket_count(std::size_t bucket_count, Fn&& fn) {
    switch (bucket_count) {
    case 16:
        fn.template operator()<16>();
        return;
    case 64:
        fn.template operator()<64>();
        return;
    case 100:
        fn.template operator()<100>();
        return;
    case 256:
        fn.template operator()<256>();
        return;
    case 1'024:
        fn.template operator()<1'024>();
        return;
    case 125:
        fn.template operator()<125>();
        return;
    case 1'000:
        fn.template operator()<1'000>();
        return;
    case 1'250:
        fn.template operator()<1'250>();
        return;
    case 8'000:
        fn.template operator()<8'000>();
        return;
    case 10'000:
        fn.template operator()<10'000>();
        return;
    case 12'500:
        fn.template operator()<12'500>();
        return;
    case 64'000:
        fn.template operator()<64'000>();
        return;
    case 80'000:
        fn.template operator()<80'000>();
        return;
    case 100'000:
        fn.template operator()<100'000>();
        return;
    case 125'000:
        fn.template operator()<125'000>();
        return;
    case 640'000:
        fn.template operator()<640'000>();
        return;
    case 800'000:
        fn.template operator()<800'000>();
        return;
    case 1'000'000:
        fn.template operator()<1'000'000>();
        return;
    default:
        throw std::invalid_argument("Unsupported benchmark bucket count");
    }
}

std::vector<benchmark_case> build_cases(const benchmark_config& config) {
    std::vector<benchmark_case> cases;
    std::uint64_t next_seed_tag = 1;

    auto append_case = [&](range_mode mode,
                           workload_kind kind,
                           std::string workload,
                           std::string shape,
                           std::size_t item_count,
                           std::size_t bucket_count,
                           std::size_t steady_state_operations,
                           std::size_t rounds) {
        cases.push_back(benchmark_case{
            mode,
            kind,
            std::move(workload),
            std::move(shape),
            item_count,
            bucket_count,
            steady_state_operations,
            rounds,
            next_seed_tag++,
        });
    };

    for (const auto mode : {range_mode::known, range_mode::growth}) {
        for (const auto item_count : config.bulk_item_counts) {
            std::size_t previous_bucket_count = 0;

            for (const auto& ratio : core_ratio_cases) {
                const auto bucket_count = bucket_count_for_ratio(item_count, ratio.numerator, ratio.denominator);
                if (bucket_count == previous_bucket_count) {
                    continue;
                }

                previous_bucket_count = bucket_count;
                const auto operations_per_round = item_count * 3;
                append_case(
                    mode,
                    workload_kind::bulk_uniform,
                    "bulk-fill-drain",
                    std::string(ratio.label),
                    item_count,
                    bucket_count,
                    0,
                    scaled_rounds(operations_per_round, config.target_total_operations, config.max_rounds)
                );
            }
        }

        for (const auto live_items : config.steady_live_counts) {
            std::size_t previous_bucket_count = 0;
            const auto steady_operations = steady_state_operation_count(config, live_items);

            for (const auto& ratio : core_ratio_cases) {
                const auto bucket_count = bucket_count_for_ratio(live_items, ratio.numerator, ratio.denominator);
                if (bucket_count == previous_bucket_count) {
                    continue;
                }

                previous_bucket_count = bucket_count;
                const auto operations_per_round = (live_items * 3) + (steady_operations * 3);
                append_case(
                    mode,
                    workload_kind::steady_uniform,
                    "steady-state",
                    std::string(ratio.label),
                    live_items,
                    bucket_count,
                    steady_operations,
                    scaled_rounds(operations_per_round, config.target_total_operations, config.max_rounds)
                );
            }

            const auto hot_set_bucket_count = bucket_count_for_ratio(live_items, 1, 1);
            const auto hot_set_operations_per_round = (live_items * 3) + (steady_operations * 3);
            append_case(
                mode,
                workload_kind::steady_hot_set,
                "steady-state",
                "hot-set",
                live_items,
                hot_set_bucket_count,
                steady_operations,
                scaled_rounds(hot_set_operations_per_round, config.target_total_operations, config.max_rounds)
            );
        }

        for (const auto item_count : config.edge_item_counts) {
            const auto bucket_count = bucket_count_for_ratio(item_count, 64, 1);
            const auto operations_per_round = item_count * 3;
            append_case(
                mode,
                workload_kind::bulk_sparse_reused,
                "bulk-fill-drain",
                "sparse-reused",
                item_count,
                bucket_count,
                0,
                scaled_rounds(operations_per_round, config.target_total_operations, config.max_rounds)
            );
        }
    }

    for (const auto item_count : config.edge_item_counts) {
        const auto operations_per_round = item_count * 3;
        append_case(
            range_mode::known,
            workload_kind::unique_priority,
            "unique-priority",
            "edge",
            item_count,
            bucket_count_for_ratio(item_count, 1, 1),
            0,
            scaled_rounds(operations_per_round, config.target_edge_operations, config.max_rounds)
        );
    }

    for (const auto item_count : config.bulk_item_counts) {
        const auto bucket_count = bucket_count_for_ratio(item_count, 1, 100);
        const auto operations_per_round = item_count * 3;
        append_case(
            range_mode::known,
            workload_kind::bulk_uniform,
            "bulk-fill-drain",
            "compact-100x",
            item_count,
            bucket_count,
            0,
            scaled_rounds(operations_per_round, config.target_total_operations, config.max_rounds)
        );
    }

    for (const auto live_items : config.steady_live_counts) {
        const auto steady_operations = steady_state_operation_count(config, live_items);
        const auto operations_per_round = (live_items * 3) + (steady_operations * 3);
        append_case(
            range_mode::known,
            workload_kind::steady_uniform,
            "steady-state",
            "compact-100x",
            live_items,
            bucket_count_for_ratio(live_items, 1, 100),
            steady_operations,
            scaled_rounds(operations_per_round, config.target_total_operations, config.max_rounds)
        );
    }

    for (const auto bucket_count : config.focused_bucket_counts) {
        append_case(
            range_mode::known,
            workload_kind::bulk_uniform,
            "bulk-fill-drain",
            "focused-range",
            config.focused_bulk_items,
            bucket_count,
            0,
            config.focused_bulk_rounds
        );

        append_case(
            range_mode::known,
            workload_kind::steady_uniform,
            "steady-state",
            "focused-range",
            config.focused_steady_live_items,
            bucket_count,
            config.focused_steady_operations,
            config.focused_steady_rounds
        );
    }

    append_case(
        range_mode::known,
        workload_kind::bulk_uniform,
        "construct-sparse",
        "focused-range",
        config.construct_sparse_items,
        config.construct_sparse_bucket_count,
        0,
        config.construct_sparse_rounds
    );

    append_case(
        range_mode::known,
        workload_kind::unique_priority,
        "unique-priority",
        "focused-range",
        config.focused_unique_priority_items,
        clamp_bucket_count(config.focused_unique_priority_items),
        0,
        config.focused_unique_priority_rounds
    );

    for (const auto item_count : config.edge_item_counts) {
        std::size_t previous_bucket_count = 0;

        for (const auto& ratio : growth_ramp_ratio_cases) {
            const auto bucket_count = bucket_count_for_ratio(item_count, ratio.numerator, ratio.denominator);
            if (bucket_count == previous_bucket_count) {
                continue;
            }

            previous_bucket_count = bucket_count;
            const auto operations_per_round = std::min(item_count, bucket_count) * 3;
            append_case(
                range_mode::growth,
                workload_kind::growth_ramp,
                "growth-ramp",
                std::string(ratio.label),
                item_count,
                bucket_count,
                0,
                scaled_rounds(operations_per_round, config.target_edge_operations, config.max_rounds)
            );
        }
    }

    append_case(
        range_mode::growth,
        workload_kind::growth_ramp,
        "growth-ramp",
        "focused-range",
        config.focused_growth_ramp_items,
        config.focused_growth_ramp_bucket_count,
        0,
        config.focused_growth_ramp_rounds
    );

    for (auto& next_case : cases) {
        if (next_case.shape == "compact-100x") {
            next_case.score_weight = config.compact_case_weight;
        }
    }

    return cases;
}

std::vector<frontier_benchmark_case> build_frontier_cases(const benchmark_config& config) {
    std::vector<frontier_benchmark_case> cases;
    std::uint64_t next_seed_tag = 1;

    auto append_case = [&](range_mode mode,
                           std::string workload,
                           frontier_pattern pattern,
                           std::size_t item_count,
                           std::size_t active_priority_count,
                           std::size_t bucket_count,
                           std::size_t gap_factor,
                           std::size_t steady_state_operations,
                           std::size_t rounds) {
        cases.push_back(frontier_benchmark_case{
            mode,
            std::move(workload),
            pattern,
            item_count,
            active_priority_count,
            bucket_count,
            gap_factor,
            steady_state_operations,
            rounds,
            next_seed_tag++,
        });
    };

    for (const auto mode : {range_mode::known, range_mode::growth}) {
        for (const auto item_count : config.frontier_item_counts) {
            const auto active_priority_count = std::max<std::size_t>(1, item_count / config.frontier_items_per_used_priority);
            const auto steady_operations = steady_state_operation_count(config, item_count);

            for (const auto gap_factor : config.frontier_gap_factors) {
                const auto requested_bucket_count = active_priority_count * gap_factor;
                const auto bucket_count = clamp_bucket_count(requested_bucket_count);
                const auto bulk_operations_per_round = item_count * 3;
                const auto steady_operations_per_round = (item_count * 3) + (steady_operations * 3);

                for (const auto pattern : {frontier_pattern::strided, frontier_pattern::page_clustered, frontier_pattern::random}) {
                    append_case(
                        mode,
                        "frontier-bulk",
                        pattern,
                        item_count,
                        active_priority_count,
                        bucket_count,
                        gap_factor,
                        0,
                        scaled_rounds(bulk_operations_per_round, config.target_total_operations, config.max_rounds)
                    );

                    append_case(
                        mode,
                        "frontier-steady",
                        pattern,
                        item_count,
                        active_priority_count,
                        bucket_count,
                        gap_factor,
                        steady_operations,
                        scaled_rounds(steady_operations_per_round, config.target_total_operations, config.max_rounds)
                    );
                }
            }
        }
    }

    return cases;
}

void run_frontier_case(
    const benchmark_config& config,
    const frontier_benchmark_case& next_case,
    std::vector<frontier_result>& results
) {
    const auto priorities = make_frontier_priority_set(
        next_case.pattern,
        next_case.bucket_count,
        next_case.active_priority_count,
        frontier_case_seed(next_case, 0xD15EA5EULL)
    );

    if (next_case.workload == "frontier-bulk") {
        const auto items = make_items_from_priority_set(
            next_case.item_count,
            frontier_case_seed(next_case, 0xB01DFACEULL),
            priorities
        );
        benchmark_frontier_bulk_case(next_case, items, results);
        return;
    }

    if (next_case.workload == "frontier-steady") {
        const auto seed_items = make_items_from_priority_set(
            next_case.item_count,
            frontier_case_seed(next_case, 0x51EEDULL),
            priorities
        );
        const auto operation_items = make_items_from_priority_set(
            next_case.steady_state_operations,
            frontier_case_seed(next_case, 0x0A71A710ULL),
            priorities
        );
        benchmark_frontier_steady_case(next_case, seed_items, operation_items, results);
        return;
    }

    throw std::invalid_argument("Unsupported frontier workload");
}

[[nodiscard]] double ns_per_operation(const benchmark_result& result) {
    return (result.seconds * 1'000'000'000.0) / static_cast<double>(result.operations);
}

[[nodiscard]] double ns_per_operation(const frontier_result& result) {
    return (result.seconds * 1'000'000'000.0) / static_cast<double>(result.operations);
}

[[nodiscard]] std::string gain_key(const benchmark_result& result) {
    return result.range + "|" + result.workload + "|" + result.shape + "|" + std::to_string(result.item_count) + "|" +
           std::to_string(result.bucket_count);
}

std::vector<gain_row> compute_gains(std::span<const benchmark_result> results) {
    static constexpr std::string_view baseline_container = "cpp_pq::dynamic_bucket_priority_queue_exact_growth";

    std::unordered_map<std::string, benchmark_result> baselines;
    std::vector<gain_row> gains;

    for (const auto& result : results) {
        if (result.container == baseline_container) {
            baselines.insert_or_assign(gain_key(result), result);
        }
    }

    for (const auto& result : results) {
        if (result.container == baseline_container) {
            continue;
        }

        const auto baseline_it = baselines.find(gain_key(result));
        if (baseline_it == baselines.end()) {
            continue;
        }

        const auto baseline_ns = ns_per_operation(baseline_it->second);
        const auto variant_ns = ns_per_operation(result);
        const auto delta_ns = baseline_ns - variant_ns;

        gains.push_back(gain_row{
            result.range,
            result.workload,
            result.shape,
            result.item_count,
            result.bucket_count,
            result.container,
            baseline_ns,
            variant_ns,
            delta_ns,
            (delta_ns / baseline_ns) * 100.0,
            baseline_ns / variant_ns,
        });
    }

    return gains;
}

std::vector<score_row> compute_scores(std::span<const benchmark_result> results) {
    static constexpr std::string_view baseline_container = "std::priority_queue";

    std::unordered_map<std::string, benchmark_result> baselines;
    std::vector<score_row> scores;
    scores.reserve(results.size());

    for (const auto& result : results) {
        if (result.container == baseline_container) {
            baselines.insert_or_assign(gain_key(result), result);
        }
    }

    for (const auto& result : results) {
        const auto baseline_it = baselines.find(gain_key(result));
        if (baseline_it == baselines.end()) {
            continue;
        }

        const auto std_ns = ns_per_operation(baseline_it->second);
        const auto variant_ns = ns_per_operation(result);
        if (std_ns <= 0.0 || variant_ns <= 0.0) {
            continue;
        }

        scores.push_back(score_row{
            result.range,
            result.workload,
            result.shape,
            result.item_count,
            result.bucket_count,
            result.container,
            std_ns,
            variant_ns,
            100.0 * (std_ns / variant_ns),
            result.score_weight,
        });
    }

    return scores;
}

std::vector<score_summary_row> compute_score_summary(std::span<const score_row> scores) {
    struct aggregate {
        double weighted_score_sum{0.0};
        double total_weight{0.0};
        std::size_t case_count{0};
    };

    std::unordered_map<std::string, aggregate> aggregates;

    for (const auto& score : scores) {
        auto& aggregate = aggregates[score.container];
        aggregate.weighted_score_sum += score.score * score.case_weight;
        aggregate.total_weight += score.case_weight;
        ++aggregate.case_count;
    }

    std::vector<score_summary_row> summary;
    summary.reserve(aggregates.size());

    for (const auto& [container, aggregate] : aggregates) {
        if (aggregate.total_weight <= 0.0) {
            continue;
        }

        summary.push_back(score_summary_row{
            container,
            aggregate.weighted_score_sum / aggregate.total_weight,
            aggregate.total_weight,
            aggregate.case_count,
        });
    }

    std::sort(summary.begin(), summary.end(), [](const score_summary_row& left, const score_summary_row& right) {
        if (left.weighted_score == right.weighted_score) {
            return left.container < right.container;
        }

        return left.weighted_score > right.weighted_score;
    });

    return summary;
}

[[nodiscard]] std::string frontier_key(const frontier_result& result) {
    return result.range + "|" + result.workload + "|" + result.pattern + "|" + std::to_string(result.item_count) + "|" +
           std::to_string(result.active_priority_count) + "|" + std::to_string(result.bucket_count);
}

std::vector<frontier_score_row> compute_frontier_scores(std::span<const frontier_result> results) {
    static constexpr std::string_view baseline_container = "cpp_pq::dynamic_bucket_priority_queue";

    std::unordered_map<std::string, frontier_result> baselines;
    std::vector<frontier_score_row> scores;
    scores.reserve(results.size());

    for (const auto& result : results) {
        if (result.container == baseline_container) {
            baselines.insert_or_assign(frontier_key(result), result);
        }
    }

    for (const auto& result : results) {
        const auto baseline_it = baselines.find(frontier_key(result));
        if (baseline_it == baselines.end()) {
            continue;
        }

        const auto baseline_ns = ns_per_operation(baseline_it->second);
        const auto variant_ns = ns_per_operation(result);
        if (baseline_ns <= 0.0 || variant_ns <= 0.0) {
            continue;
        }

        scores.push_back(frontier_score_row{
            result.range,
            result.workload,
            result.pattern,
            result.item_count,
            result.active_priority_count,
            result.bucket_count,
            result.gap_factor,
            result.container,
            baseline_ns,
            variant_ns,
            100.0 * (baseline_ns / variant_ns),
            1.0,
        });
    }

    return scores;
}

std::vector<score_summary_row> compute_frontier_score_summary(std::span<const frontier_score_row> scores) {
    struct aggregate {
        double weighted_score_sum{0.0};
        double total_weight{0.0};
        std::size_t case_count{0};
    };

    std::unordered_map<std::string, aggregate> aggregates;

    for (const auto& score : scores) {
        auto& aggregate = aggregates[score.container];
        aggregate.weighted_score_sum += score.score * score.case_weight;
        aggregate.total_weight += score.case_weight;
        ++aggregate.case_count;
    }

    std::vector<score_summary_row> summary;
    summary.reserve(aggregates.size());

    for (const auto& [container, aggregate] : aggregates) {
        if (aggregate.total_weight <= 0.0) {
            continue;
        }

        summary.push_back(score_summary_row{
            container,
            aggregate.weighted_score_sum / aggregate.total_weight,
            aggregate.total_weight,
            aggregate.case_count,
        });
    }

    std::sort(summary.begin(), summary.end(), [](const score_summary_row& left, const score_summary_row& right) {
        if (left.weighted_score == right.weighted_score) {
            return left.container < right.container;
        }

        return left.weighted_score > right.weighted_score;
    });

    return summary;
}

std::vector<frontier_gap_summary_row> compute_frontier_gap_summary(std::span<const frontier_score_row> scores) {
    struct aggregate {
        double score_sum{0.0};
        std::size_t case_count{0};
    };

    std::unordered_map<std::string, aggregate> aggregates;

    for (const auto& score : scores) {
        const auto key = std::to_string(score.gap_factor) + "|" + score.container;
        auto& aggregate = aggregates[key];
        aggregate.score_sum += score.score;
        ++aggregate.case_count;
    }

    std::vector<frontier_gap_summary_row> summary;
    summary.reserve(aggregates.size());

    for (const auto& [key, aggregate] : aggregates) {
        const auto delimiter = key.find('|');
        summary.push_back(frontier_gap_summary_row{
            static_cast<std::size_t>(std::stoull(key.substr(0, delimiter))),
            key.substr(delimiter + 1),
            aggregate.score_sum / static_cast<double>(aggregate.case_count),
            aggregate.case_count,
        });
    }

    std::sort(summary.begin(), summary.end(), [](const frontier_gap_summary_row& left, const frontier_gap_summary_row& right) {
        if (left.gap_factor != right.gap_factor) {
            return left.gap_factor < right.gap_factor;
        }

        if (left.average_score == right.average_score) {
            return left.container < right.container;
        }

        return left.average_score > right.average_score;
    });

    return summary;
}

void write_gains_csv(const std::filesystem::path& output_path, std::span<const gain_row> gains) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "range,workload,shape,items,buckets,container,baseline_ns_per_op,variant_ns_per_op,delta_ns_per_op,percent_gain,speedup\n";

    for (const auto& gain : gains) {
        output << gain.range << ','
               << gain.workload << ','
               << gain.shape << ','
               << gain.item_count << ','
               << gain.bucket_count << ','
               << gain.container << ','
               << std::fixed << std::setprecision(6) << gain.baseline_ns_per_op << ','
               << std::fixed << std::setprecision(6) << gain.variant_ns_per_op << ','
               << std::fixed << std::setprecision(6) << gain.delta_ns_per_op << ','
               << std::fixed << std::setprecision(6) << gain.percent_gain << ','
               << std::fixed << std::setprecision(6) << gain.speedup
               << '\n';
    }
}

void write_scores_csv(const std::filesystem::path& output_path, std::span<const score_row> scores) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "range,workload,shape,items,buckets,container,std_ns_per_op,variant_ns_per_op,score,case_weight\n";

    for (const auto& score : scores) {
        output << score.range << ','
               << score.workload << ','
               << score.shape << ','
               << score.item_count << ','
               << score.bucket_count << ','
               << score.container << ','
               << std::fixed << std::setprecision(6) << score.std_ns_per_op << ','
               << std::fixed << std::setprecision(6) << score.variant_ns_per_op << ','
               << std::fixed << std::setprecision(6) << score.score << ','
               << std::fixed << std::setprecision(3) << score.case_weight
               << '\n';
    }
}

void write_score_summary_csv(const std::filesystem::path& output_path, std::span<const score_summary_row> summary) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "container,weighted_score,total_weight,case_count\n";

    for (const auto& row : summary) {
        output << row.container << ','
               << std::fixed << std::setprecision(6) << row.weighted_score << ','
               << std::fixed << std::setprecision(3) << row.total_weight << ','
               << row.case_count
               << '\n';
    }
}

void write_frontier_results_csv(const std::filesystem::path& output_path, std::span<const frontier_result> results) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "range,workload,pattern,items,active_priorities,buckets,gap_factor,container,total_ms,ns_per_op,checksum\n";

    for (const auto& result : results) {
        output << result.range << ','
               << result.workload << ','
               << result.pattern << ','
               << result.item_count << ','
               << result.active_priority_count << ','
               << result.bucket_count << ','
               << result.gap_factor << ','
               << result.container << ','
               << std::fixed << std::setprecision(6) << (result.seconds * 1'000.0) << ','
               << std::fixed << std::setprecision(6) << ns_per_operation(result) << ','
               << result.checksum
               << '\n';
    }
}

void write_frontier_scores_csv(const std::filesystem::path& output_path, std::span<const frontier_score_row> scores) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "range,workload,pattern,items,active_priorities,buckets,gap_factor,container,baseline_ns_per_op,variant_ns_per_op,score\n";

    for (const auto& score : scores) {
        output << score.range << ','
               << score.workload << ','
               << score.pattern << ','
               << score.item_count << ','
               << score.active_priority_count << ','
               << score.bucket_count << ','
               << score.gap_factor << ','
               << score.container << ','
               << std::fixed << std::setprecision(6) << score.baseline_ns_per_op << ','
               << std::fixed << std::setprecision(6) << score.variant_ns_per_op << ','
               << std::fixed << std::setprecision(6) << score.score
               << '\n';
    }
}

void write_frontier_gap_summary_csv(const std::filesystem::path& output_path, std::span<const frontier_gap_summary_row> summary) {
    std::ofstream output(output_path, std::ios::trunc);
    output << "gap_factor,container,average_score,case_count\n";

    for (const auto& row : summary) {
        output << row.gap_factor << ','
               << row.container << ','
               << std::fixed << std::setprecision(6) << row.average_score << ','
               << row.case_count
               << '\n';
    }
}

void print_results(const benchmark_config& config, std::span<const benchmark_result> results) {
    std::cout << "Unified bounded-priority queue benchmark matrix\n";
    std::cout << "bulk_items={1k,10k,100k,1m}, steady_live={1k,10k,100k}, core_ratios={1/8,1,8}\n";
    std::cout << "known-range: dynamic variants are pre-sized to the final bucket range\n";
    std::cout << "growth-range: dynamic variants start empty each measured run and pay the expansion cost in-band\n";
    std::cout << "focused-range: preserves the prior dynamic-variant sweep at buckets={16,64,256,1024,100k}\n";
    std::cout << "compact-100x: compact priorities with about 100 items per bucket and score weight "
              << config.compact_case_weight << '\n';
#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
    std::cout << "fbvector variant: enabled\n";
#else
    std::cout << "fbvector variant: disabled (configure with -DCPP_PQ_ENABLE_FOLLY_FBVECTOR=ON)\n";
#endif
    std::cout << "steady_state_multiplier=" << config.steady_state_multiplier
              << ", steady_state_max_operations=" << config.steady_state_max_operations
              << ", sparse_reused_priorities=" << config.sparse_reused_priority_count
              << ", hot_set_probability=" << config.hot_set_probability
              << ", focused_construct_sparse_items=" << config.construct_sparse_items
              << ", focused_growth_ramp_items=" << config.focused_growth_ramp_items
              << '\n'
              << '\n';

    std::cout << std::left
              << std::setw(13) << "range"
              << std::setw(18) << "workload"
              << std::setw(15) << "shape"
              << std::setw(10) << "items"
              << std::setw(10) << "buckets"
              << std::setw(56) << "container"
              << std::setw(14) << "total ms"
              << std::setw(14) << "ns/op"
              << "checksum"
              << '\n';

    for (const auto& result : results) {
        std::cout << std::left
                  << std::setw(13) << result.range
                  << std::setw(18) << result.workload
                  << std::setw(15) << result.shape
                  << std::setw(10) << result.item_count
                  << std::setw(10) << result.bucket_count
                  << std::setw(56) << result.container
                  << std::setw(14) << std::fixed << std::setprecision(3) << (result.seconds * 1'000.0)
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_per_operation(result)
                  << result.checksum
                  << '\n';
    }
}

void print_score_summary(std::span<const score_summary_row> summary) {
    std::cout << "\nWeighted score vs std::priority_queue (std = 100)\n";
    std::cout << std::left
              << std::setw(56) << "container"
              << std::setw(16) << "weighted score"
              << std::setw(14) << "total weight"
              << "cases"
              << '\n';

    for (const auto& row : summary) {
        std::cout << std::left
                  << std::setw(56) << row.container
                  << std::setw(16) << std::fixed << std::setprecision(3) << row.weighted_score
                  << std::setw(14) << std::fixed << std::setprecision(3) << row.total_weight
                  << row.case_count
                  << '\n';
    }
}

void print_frontier_score_summary(std::span<const score_summary_row> summary) {
    std::cout << "\nSparsity frontier score vs cpp_pq::dynamic_bucket_priority_queue (dynamic = 100)\n";
    std::cout << std::left
              << std::setw(56) << "container"
              << std::setw(16) << "average score"
              << std::setw(14) << "total weight"
              << "cases"
              << '\n';

    for (const auto& row : summary) {
        std::cout << std::left
                  << std::setw(56) << row.container
                  << std::setw(16) << std::fixed << std::setprecision(3) << row.weighted_score
                  << std::setw(14) << std::fixed << std::setprecision(3) << row.total_weight
                  << row.case_count
                  << '\n';
    }
}

void print_frontier_gap_summary(std::span<const frontier_gap_summary_row> summary) {
    std::cout << "\nAverage frontier score by requested gap factor (dynamic = 100)\n";
    std::cout << std::left
              << std::setw(10) << "gap"
              << std::setw(56) << "container"
              << std::setw(16) << "avg score"
              << "cases"
              << '\n';

    for (const auto& row : summary) {
        std::cout << std::left
                  << std::setw(10) << row.gap_factor
                  << std::setw(56) << row.container
                  << std::setw(16) << std::fixed << std::setprecision(3) << row.average_score
                  << row.case_count
                  << '\n';
    }
}

void print_gains(std::span<const gain_row> gains) {
    std::cout << "\nGains vs cpp_pq::dynamic_bucket_priority_queue_exact_growth\n";
    std::cout << std::left
              << std::setw(13) << "range"
              << std::setw(18) << "workload"
              << std::setw(15) << "shape"
              << std::setw(10) << "items"
              << std::setw(10) << "buckets"
              << std::setw(56) << "container"
              << std::setw(14) << "delta ns/op"
              << std::setw(14) << "% gain"
              << "speedup"
              << '\n';

    for (const auto& gain : gains) {
        std::cout << std::left
                  << std::setw(13) << gain.range
                  << std::setw(18) << gain.workload
                  << std::setw(15) << gain.shape
                  << std::setw(10) << gain.item_count
                  << std::setw(10) << gain.bucket_count
                  << std::setw(56) << gain.container
                  << std::setw(14) << std::fixed << std::setprecision(3) << gain.delta_ns_per_op
                  << std::setw(14) << std::fixed << std::setprecision(3) << gain.percent_gain
                  << gain.speedup
                  << '\n';
    }
}

} // namespace

int main() {
    const benchmark_config config{};
    const auto cases = build_cases(config);
    const auto frontier_cases = build_frontier_cases(config);

    std::vector<benchmark_result> results;
    std::vector<frontier_result> frontier_results;
#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
    constexpr std::size_t benchmark_variant_count = 7;
    constexpr std::size_t frontier_variant_count = 6;
#else
    constexpr std::size_t benchmark_variant_count = 6;
    constexpr std::size_t frontier_variant_count = 5;
#endif
    results.reserve(cases.size() * benchmark_variant_count);
    frontier_results.reserve(frontier_cases.size() * frontier_variant_count);

    for (const auto& next_case : cases) {
        dispatch_bucket_count(next_case.bucket_count, [&]<std::size_t BucketCount>() {
            run_case<BucketCount>(config, next_case, results);
        });
    }

    for (const auto& next_case : frontier_cases) {
        run_frontier_case(config, next_case, frontier_results);
    }

    const auto gains = compute_gains(results);
    const auto scores = compute_scores(results);
    const auto score_summary = compute_score_summary(scores);
    const auto frontier_scores = compute_frontier_scores(frontier_results);
    const auto frontier_score_summary = compute_frontier_score_summary(frontier_scores);
    const auto frontier_gap_summary = compute_frontier_gap_summary(frontier_scores);
    const auto gains_output_path = std::filesystem::current_path() / "benchmark_priority_queue_gains.csv";
    const auto scores_output_path = std::filesystem::current_path() / "benchmark_priority_queue_scores.csv";
    const auto score_summary_output_path = std::filesystem::current_path() / "benchmark_priority_queue_score_summary.csv";
    const auto frontier_results_output_path = std::filesystem::current_path() / "benchmark_priority_queue_frontier_results.csv";
    const auto frontier_scores_output_path = std::filesystem::current_path() / "benchmark_priority_queue_frontier_scores.csv";
    const auto frontier_gap_summary_output_path =
        std::filesystem::current_path() / "benchmark_priority_queue_frontier_gap_summary.csv";
    write_gains_csv(gains_output_path, gains);
    write_scores_csv(scores_output_path, scores);
    write_score_summary_csv(score_summary_output_path, score_summary);
    write_frontier_results_csv(frontier_results_output_path, frontier_results);
    write_frontier_scores_csv(frontier_scores_output_path, frontier_scores);
    write_frontier_gap_summary_csv(frontier_gap_summary_output_path, frontier_gap_summary);

    print_results(config, results);
    print_score_summary(score_summary);
    print_frontier_score_summary(frontier_score_summary);
    print_frontier_gap_summary(frontier_gap_summary);
    print_gains(gains);
    std::cout << "\nCSVs written to "
              << gains_output_path.string()
              << ", "
              << scores_output_path.string()
              << ", and "
              << score_summary_output_path.string()
              << ", "
              << frontier_results_output_path.string()
              << ", "
              << frontier_scores_output_path.string()
              << ", and "
              << frontier_gap_summary_output_path.string()
              << '\n';
    return 0;
}
