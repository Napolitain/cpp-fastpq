#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
    std::size_t target_total_operations = 1'000'000;
    std::size_t target_edge_operations = 750'000;
    std::size_t max_rounds = 512;
    std::size_t steady_state_multiplier = 10;
    std::size_t steady_state_min_operations = 10'000;
    std::size_t steady_state_max_operations = 1'000'000;
    std::size_t sparse_reused_priority_count = 16;
    std::size_t hot_set_priority_count = 8;
    double hot_set_probability = 0.90;
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

    std_priority_queue_adapter std_queue;
    add_result("std::priority_queue", [&]() {
        return run_bulk_round(std_queue, items);
    });

    auto static_queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
    add_result("cpp_pq::static_bucket_priority_queue", [&]() {
        return run_bulk_round(*static_queue, items);
    });

    if (next_case.mode == range_mode::known) {
        dynamic_bucket_queue_adapter dynamic_queue(BucketCount);
        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            return run_bulk_round(dynamic_queue, items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        fbvector_dynamic_bucket_queue_adapter fbvector_queue(BucketCount);
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            return run_bulk_round(fbvector_queue, items);
        });
#endif
    } else {
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

    std_priority_queue_adapter std_queue;
    add_result("std::priority_queue", [&]() {
        return run_steady_state_round(std_queue, seed_items, operation_items);
    });

    auto static_queue = std::make_unique<static_bucket_queue_adapter<BucketCount>>();
    add_result("cpp_pq::static_bucket_priority_queue", [&]() {
        return run_steady_state_round(*static_queue, seed_items, operation_items);
    });

    if (next_case.mode == range_mode::known) {
        dynamic_bucket_queue_adapter dynamic_queue(BucketCount);
        add_result("cpp_pq::dynamic_bucket_priority_queue", [&]() {
            return run_steady_state_round(dynamic_queue, seed_items, operation_items);
        });

#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
        fbvector_dynamic_bucket_queue_adapter fbvector_queue(BucketCount);
        add_result("cpp_pq::dynamic_bucket_priority_queue_folly_fbvector", [&]() {
            return run_steady_state_round(fbvector_queue, seed_items, operation_items);
        });
#endif
    } else {
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

    return cases;
}

[[nodiscard]] double ns_per_operation(const benchmark_result& result) {
    return (result.seconds * 1'000'000'000.0) / static_cast<double>(result.operations);
}

void print_results(const benchmark_config& config, std::span<const benchmark_result> results) {
    std::cout << "Bounded-priority queue benchmark matrix\n";
    std::cout << "bulk_items={1k,10k,100k,1m}, steady_live={1k,10k,100k}, core_ratios={1/8,1,8}\n";
    std::cout << "known-range: dynamic variants are pre-sized to the final bucket range\n";
    std::cout << "growth-range: dynamic variants start empty each measured run and pay the expansion cost in-band\n";
#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
    std::cout << "fbvector variant: enabled\n";
#else
    std::cout << "fbvector variant: disabled (configure with -DCPP_PQ_ENABLE_FOLLY_FBVECTOR=ON)\n";
#endif
    std::cout << "steady_state_multiplier=" << config.steady_state_multiplier
              << ", steady_state_max_operations=" << config.steady_state_max_operations
              << ", sparse_reused_priorities=" << config.sparse_reused_priority_count
              << ", hot_set_probability=" << config.hot_set_probability
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

} // namespace

int main() {
    const benchmark_config config{};
    const auto cases = build_cases(config);

    std::vector<benchmark_result> results;
#if defined(CPP_PQ_HAS_FOLLY_FBVECTOR)
    constexpr std::size_t benchmark_variant_count = 4;
#else
    constexpr std::size_t benchmark_variant_count = 3;
#endif
    results.reserve(cases.size() * benchmark_variant_count);

    for (const auto& next_case : cases) {
        dispatch_bucket_count(next_case.bucket_count, [&]<std::size_t BucketCount>() {
            run_case<BucketCount>(config, next_case, results);
        });
    }

    print_results(config, results);
    return 0;
}
