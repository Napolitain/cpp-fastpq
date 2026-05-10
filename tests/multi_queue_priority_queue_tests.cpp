#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "cpp_pq/bucket_priority_queue.hpp"

namespace {

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string("Expectation failed: ") + std::string(message));
    }
}

template <typename Fn>
void expect_out_of_range(Fn&& fn, std::string_view message) {
    try {
        fn();
    } catch (const std::out_of_range&) {
        return;
    }

    throw std::runtime_error(std::string("Expected std::out_of_range: ") + std::string(message));
}

template <typename Fn>
void expect_logic_error(Fn&& fn, std::string_view message) {
    try {
        fn();
    } catch (const std::logic_error&) {
        return;
    }

    throw std::runtime_error(std::string("Expected std::logic_error: ") + std::string(message));
}

void test_basic_priority_behavior() {
    cpp_pq::bucket_priority_queue<int, 8> queue;

    expect(queue.empty(), "queue starts empty");
    expect(queue.size() == 0, "queue size starts at zero");
    expect_out_of_range([&]() { (void)queue.top(); }, "top() on empty queue");
    expect_out_of_range([&]() { queue.pop(); }, "pop() on empty queue");

    queue.push(1, 10);
    queue.push(6, 60);
    queue.push(3, 30);

    expect(!queue.empty(), "queue becomes non-empty after pushes");
    expect(queue.size() == 3, "queue tracks size");
    expect(queue.top() == 10, "lowest numeric priority value is returned");
    expect(queue.top_priority() == 1, "lowest numeric priority index is tracked");

    queue.push(1, 11);
    expect(queue.top() == 10, "equal-priority values keep FIFO insertion order");

    queue.pop();
    expect(queue.size() == 3, "pop reduces size by one");
    expect(queue.top() == 11, "removing the oldest equal-priority element reveals the next one");

    queue.emplace(0, 70);
    expect(queue.top() == 70, "emplace inserts directly into the correct bucket");
    expect(queue.top_priority() == 0, "emplace updates the tracked priority");
}

void test_clear_and_bounds() {
    cpp_pq::bucket_priority_queue<int, 4> queue;

    queue.push(0, 1);
    queue.push(3, 2);
    queue.clear();

    expect(queue.empty(), "clear resets emptiness");
    expect(queue.size() == 0, "clear resets size");
    expect_out_of_range([&]() { (void)queue.top_priority(); }, "top_priority() on empty queue");
    expect_out_of_range([&]() { queue.push(4, 7); }, "push() with out-of-range priority");
}

void test_alias_type() {
    cpp_pq::multi_queue_priority_queue<int, 2> queue;
    queue.push(1, 99);
    expect(queue.top() == 99, "multi_queue alias maps to the bucketed implementation");
}

void test_bulk_priority_behavior() {
    cpp_pq::bulk_bucket_priority_queue<int, 8> queue;

    queue.push(1, 10);
    queue.push(6, 60);
    queue.push(3, 30);
    queue.push(1, 11);

    expect(queue.top() == 10, "bulk queue returns the lowest numeric priority value");
    expect(queue.top_priority() == 1, "bulk queue tracks the lowest numeric priority index");

    queue.pop();
    expect(queue.size() == 3, "bulk pop reduces size by one");
    expect(queue.top() == 11, "bulk queue preserves FIFO order within a priority");

    queue.pop();
    expect(queue.top_priority() == 3, "bulk queue scans to the next populated priority");
}

void test_bulk_rejects_push_during_drain() {
    cpp_pq::bulk_bucket_priority_queue<int, 4> queue;

    queue.push(3, 30);
    queue.push(1, 10);
    queue.pop();

    expect_logic_error([&]() { queue.push(2, 20); }, "bulk push after drain starts");

    queue.pop();
    expect(queue.empty(), "bulk queue fully drained");

    queue.push(2, 200);
    expect(queue.top_priority() == 2, "bulk queue accepts a new fill phase after draining");
}

void test_public_alias_mappings() {
    static_assert(std::is_same_v<
                  cpp_pq::static_bucket_priority_queue<int, 8>,
                  cpp_pq::bucket_priority_queue<int, 8>>);
    static_assert(std::is_same_v<
                  cpp_pq::bulk_multi_queue_priority_queue<int, 8>,
                  cpp_pq::bulk_bucket_priority_queue<int, 8>>);
    static_assert(std::is_same_v<
                  cpp_pq::multi_queue_priority_queue<int, 8>,
                  cpp_pq::bucket_priority_queue<int, 8>>);
    static_assert(std::is_same_v<
                  cpp_pq::dynamic_bucket_priority_queue_exact_growth<int>,
                  cpp_pq::dynamic_bucket_priority_queue_base<int, false>>);
    static_assert(std::is_same_v<
                  cpp_pq::dynamic_bucket_priority_queue_geometric<int>,
                  cpp_pq::dynamic_bucket_priority_queue_base<int, true>>);
    static_assert(std::is_same_v<
                  cpp_pq::dynamic_bucket_priority_queue<int>,
                  cpp_pq::dynamic_bucket_priority_queue_geometric<int>>);
    static_assert(std::is_same_v<
                  cpp_pq::dynamic_multi_queue_priority_queue<int>,
                  cpp_pq::dynamic_bucket_priority_queue<int>>);
    static_assert(std::is_same_v<
                  cpp_pq::paged_dynamic_bucket_priority_queue<int>,
                  cpp_pq::paged_dynamic_bucket_priority_queue_base<int, false>>);
    static_assert(std::is_same_v<
                  cpp_pq::paged_dynamic_bucket_priority_queue_geometric<int>,
                  cpp_pq::paged_dynamic_bucket_priority_queue_base<int, true>>);
    static_assert(std::is_same_v<
                  cpp_pq::registered_multi_queue_priority_queue<int>,
                  cpp_pq::registered_bucket_priority_queue<int>>);
}

void test_large_static_bucket_count() {
    auto queue = std::make_unique<cpp_pq::static_bucket_priority_queue<int, 100000>>();

    queue->push(17, 17);
    queue->push(4097, 4097);
    queue->push(99999, 99999);

    expect(queue->top_priority() == 17, "large static queue tracks a low priority across multiple bitmap levels");
    expect(queue->top() == 17, "large static queue returns the lowest numeric priority value");

    queue->pop();
    expect(queue->top_priority() == 4097, "large static queue falls back to the next populated priority");
}

void test_dynamic_growth() {
    cpp_pq::dynamic_bucket_priority_queue<int> queue;

    expect(queue.bucket_count() == 0, "dynamic queue starts with zero configured priorities");
    expect_out_of_range([&]() { (void)queue.top(); }, "dynamic top() on empty queue");
    expect_out_of_range([&]() { queue.pop(); }, "dynamic pop() on empty queue");

    queue.push(12, 120);
    expect(queue.bucket_count() == 13, "dynamic queue grows to include the inserted priority");
    expect(queue.top_priority() == 12, "dynamic queue tracks the grown priority");

    queue.push(2, 20);
    queue.emplace(40, 400);
    expect(queue.bucket_count() == 41, "dynamic queue expands again when a larger priority arrives");
    expect(queue.top() == 20, "dynamic queue returns the lowest numeric priority value after expansion");
    expect(queue.top_priority() == 2, "dynamic queue reports the lowest numeric priority");

    queue.pop();
    expect(queue.top_priority() == 12, "dynamic queue falls back correctly after removing the top bucket");

    queue.expand_priorities(128);
    expect(queue.bucket_count() == 128, "dynamic queue can be expanded explicitly");

    queue.clear();
    expect(queue.empty(), "dynamic clear resets emptiness");
    expect(queue.bucket_count() == 128, "dynamic clear preserves the configured priority range");
}

void test_dynamic_alias_type() {
    cpp_pq::dynamic_multi_queue_priority_queue<int> queue;
    queue.push(3, 33);
    expect(queue.top() == 33, "dynamic alias maps to the dynamic bucket implementation");
}

void test_geometric_dynamic_growth() {
    cpp_pq::dynamic_bucket_priority_queue_geometric<int> queue;

    expect(queue.bucket_count() == 0, "geometric dynamic queue starts with zero configured priorities");
    queue.push(12, 120);
    expect(queue.bucket_count() == 13, "geometric dynamic queue tracks the logical priority range");

    queue.push(40, 400);
    expect(queue.bucket_count() == 41, "geometric dynamic queue grows logically to include the inserted priority");
    expect(queue.top_priority() == 12, "geometric dynamic queue reports the lowest numeric priority");
    expect(queue.top() == 120, "geometric dynamic queue returns the lowest numeric priority value");
}

void test_paged_dynamic_sparse_priorities() {
    cpp_pq::paged_dynamic_bucket_priority_queue<int> queue;

    queue.push(1, 10);
    queue.push(10000, 20);
    expect(queue.bucket_count() == 10001, "paged dynamic queue tracks the sparse logical range");
    expect(queue.top_priority() == 1, "paged dynamic queue finds the lowest sparse priority");
    expect(queue.top() == 10, "paged dynamic queue returns the sparse lowest-priority value");

    queue.pop();
    expect(queue.top_priority() == 10000, "paged dynamic queue falls back across sparse pages");
}

void test_paged_geometric_dynamic_sparse_priorities() {
    cpp_pq::paged_dynamic_bucket_priority_queue_geometric<int> queue;

    queue.push(5, 50);
    queue.push(4097, 4097);
    queue.push(99999, 99999);
    expect(queue.bucket_count() == 100000, "paged geometric queue tracks the logical sparse range");
    expect(queue.top_priority() == 5, "paged geometric queue returns the lowest sparse priority");
    expect(queue.top() == 50, "paged geometric queue returns the lowest-priority value");

    queue.clear();
    expect(queue.empty(), "paged geometric queue clear removes all queued items");
    expect(queue.bucket_count() == 100000, "paged geometric queue clear preserves logical capacity");
}

void test_registered_sparse_priorities() {
    cpp_pq::registered_bucket_priority_queue<int> queue;

    auto low = queue.register_priority(1);
    auto high = queue.register_priority(10000);
    auto duplicate_high = queue.register_priority(10000);

    expect(queue.registered_priority_count() == 2, "registered queue stores sparse priorities without materializing the full range");
    expect(high == duplicate_high, "registering the same priority returns the same handle");

    queue.push(low, 10);
    queue.push(high, 20);

    expect(queue.top_priority() == 1, "registered queue returns the lowest sparse priority");
    expect(queue.top() == 10, "registered queue returns the lowest-priority value");

    queue.pop();
    expect(queue.top_priority() == 10000, "registered queue falls back to the next registered sparse priority");
}

void test_registered_handles_survive_clear() {
    cpp_pq::registered_bucket_priority_queue<int> queue;

    const auto urgent = queue.register_priority(42);
    queue.push(urgent, 420);
    queue.clear();

    expect(queue.empty(), "registered queue clear removes queued items");
    expect(queue.registered_priority_count() == 1, "registered priorities remain cached after clear");

    queue.push(urgent, 421);
    expect(queue.top_priority() == 42, "registered handles remain valid after clear");
    expect(queue.top() == 421, "registered handles can be reused after clear");
}

void test_registered_invalid_handle() {
    cpp_pq::registered_bucket_priority_queue<int> queue;
    cpp_pq::registered_bucket_priority_queue<int>::priority_handle invalid_handle;

    try {
        queue.push(invalid_handle, 7);
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("Expected std::invalid_argument for an invalid registered priority handle");
}

void test_registered_alias_type() {
    cpp_pq::registered_multi_queue_priority_queue<int> queue;
    const auto handle = queue.register_priority(3);
    queue.push(handle, 33);
    expect(queue.top() == 33, "registered alias maps to the registered bucket implementation");
}

} // namespace

int main() {
    try {
        test_basic_priority_behavior();
        test_clear_and_bounds();
        test_alias_type();
        test_bulk_priority_behavior();
        test_bulk_rejects_push_during_drain();
        test_public_alias_mappings();
        test_large_static_bucket_count();
        test_dynamic_growth();
        test_dynamic_alias_type();
        test_geometric_dynamic_growth();
        test_paged_dynamic_sparse_priorities();
        test_paged_geometric_dynamic_sparse_priorities();
        test_registered_sparse_priorities();
        test_registered_handles_survive_clear();
        test_registered_invalid_handle();
        test_registered_alias_type();
    } catch (const std::exception& ex) {
        std::cerr << "Test failure: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "All bucket priority queue tests passed.\n";
    return 0;
}
