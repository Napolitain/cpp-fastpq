#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpp_pq {

namespace detail {

inline constexpr std::size_t occupancy_word_bits = 64;

[[nodiscard]] inline constexpr std::size_t words_for_bits(std::size_t bit_count) noexcept {
    return (bit_count + occupancy_word_bits - 1) / occupancy_word_bits;
}

[[nodiscard]] inline constexpr std::size_t word_index(std::size_t bit_index) noexcept {
    return bit_index / occupancy_word_bits;
}

[[nodiscard]] inline constexpr std::uint64_t bit_mask(std::size_t bit_index) noexcept {
    return std::uint64_t{1} << (bit_index % occupancy_word_bits);
}

[[nodiscard]] inline constexpr std::size_t lowest_bit_index(std::uint64_t value) noexcept {
    return static_cast<std::size_t>(std::countr_zero(value));
}

[[nodiscard]] inline std::size_t geometric_capacity(std::size_t required_capacity) noexcept {
    if (required_capacity == 0) {
        return 0;
    }

    return std::bit_ceil(std::max(required_capacity, occupancy_word_bits));
}

template <typename T, template <typename, typename> class BucketContainer = std::vector>
class fifo_bucket {
public:
    using value_type = T;
    using container_type = BucketContainer<T, std::allocator<T>>;

    [[nodiscard]] bool empty() const noexcept {
        return head_ >= values_.size();
    }

    [[nodiscard]] value_type& front() {
        return values_[head_];
    }

    [[nodiscard]] const value_type& front() const {
        return values_[head_];
    }

    template <typename U>
    void push(U&& value) {
        values_.push_back(std::forward<U>(value));
    }

    template <typename... Args>
    value_type& emplace(Args&&... args) {
        values_.emplace_back(std::forward<Args>(args)...);
        return values_.back();
    }

    void pop() {
        ++head_;

        if (empty()) {
            clear();
            return;
        }

        compact_if_needed();
    }

    void clear() {
        values_.clear();
        head_ = 0;
    }

private:
    static constexpr std::size_t min_compact_head = 64;

    void compact_if_needed() {
        if (head_ < min_compact_head || (head_ * 2) < values_.size()) {
            return;
        }

        container_type compacted;
        compacted.reserve(values_.size() - head_);
        const auto active_begin = values_.begin() + static_cast<typename container_type::difference_type>(head_);
        std::move(active_begin, values_.end(), std::back_inserter(compacted));
        values_ = std::move(compacted);
        head_ = 0;
    }

    container_type values_{};
    std::size_t head_{0};
};

template <std::size_t BucketCount>
class static_bucket_index {
    static_assert(BucketCount > 0, "static_bucket_index requires at least one bit");

    static constexpr std::size_t leaf_word_count = words_for_bits(BucketCount);

    [[nodiscard]] static consteval std::size_t compute_level_count() noexcept {
        std::size_t level_count = 1;
        std::size_t word_count = leaf_word_count;

        while (word_count > 1) {
            word_count = words_for_bits(word_count);
            ++level_count;
        }

        return level_count;
    }

    static constexpr std::size_t level_count = compute_level_count();

    [[nodiscard]] static consteval auto make_level_word_counts() noexcept {
        std::array<std::size_t, level_count> counts{};
        counts[0] = leaf_word_count;

        for (std::size_t level = 1; level < level_count; ++level) {
            counts[level] = words_for_bits(counts[level - 1]);
        }

        return counts;
    }

    static constexpr auto level_word_counts_ = make_level_word_counts();

    [[nodiscard]] static consteval std::size_t compute_total_word_count() noexcept {
        std::size_t total = 0;

        for (const auto count : level_word_counts_) {
            total += count;
        }

        return total;
    }

    static constexpr std::size_t total_word_count = compute_total_word_count();

    [[nodiscard]] static consteval auto make_level_offsets() noexcept {
        std::array<std::size_t, level_count> offsets{};
        std::size_t current_offset = 0;

        for (std::size_t level = 0; level < level_count; ++level) {
            offsets[level] = current_offset;
            current_offset += level_word_counts_[level];
        }

        return offsets;
    }

    static constexpr auto level_offsets_ = make_level_offsets();

public:
    void clear() noexcept {
        words_.fill(0);
    }

    void set(std::size_t bit_index) noexcept {
        std::size_t child_index = word_index(bit_index);
        auto* leaf_words = level_data(0);
        const auto previous = leaf_words[child_index];
        leaf_words[child_index] |= bit_mask(bit_index);

        if (previous != 0) {
            return;
        }

        for (std::size_t level = 1; level < level_count; ++level) {
            auto* level_words = level_data(level);
            const auto parent_word_index = word_index(child_index);
            const auto parent_previous = level_words[parent_word_index];
            level_words[parent_word_index] |= bit_mask(child_index);

            if (parent_previous != 0) {
                return;
            }

            child_index = parent_word_index;
        }
    }

    void reset(std::size_t bit_index) noexcept {
        std::size_t child_index = word_index(bit_index);
        auto* leaf_words = level_data(0);
        leaf_words[child_index] &= ~bit_mask(bit_index);

        if (leaf_words[child_index] != 0) {
            return;
        }

        for (std::size_t level = 1; level < level_count; ++level) {
            auto* level_words = level_data(level);
            const auto parent_word_index = word_index(child_index);
            level_words[parent_word_index] &= ~bit_mask(child_index);

            if (level_words[parent_word_index] != 0) {
                return;
            }

            child_index = parent_word_index;
        }
    }

    [[nodiscard]] std::size_t lowest_set_bit() const noexcept {
        std::size_t current_word_index = 0;

        for (std::size_t level = level_count - 1; level > 0; --level) {
            const auto word = level_data(level)[current_word_index];
            current_word_index = (current_word_index * occupancy_word_bits) + lowest_bit_index(word);
        }

        const auto leaf_word = level_data(0)[current_word_index];
        return (current_word_index * occupancy_word_bits) + lowest_bit_index(leaf_word);
    }

private:
    [[nodiscard]] std::uint64_t* level_data(std::size_t level) noexcept {
        return words_.data() + level_offsets_[level];
    }

    [[nodiscard]] const std::uint64_t* level_data(std::size_t level) const noexcept {
        return words_.data() + level_offsets_[level];
    }

    std::array<std::uint64_t, total_word_count> words_{};
};

class dynamic_bucket_index {
public:
    explicit dynamic_bucket_index(std::size_t bit_count = 0) {
        expand(bit_count);
    }

    void expand(std::size_t bit_count) {
        if (bit_count <= bit_count_) {
            return;
        }

        auto previous_levels = std::move(levels_);
        levels_ = make_levels(bit_count);
        bit_count_ = bit_count;

        if (previous_levels.empty()) {
            return;
        }

        const auto leaf_word_copy_count = std::min(previous_levels.front().size(), levels_.front().size());
        std::copy_n(previous_levels.front().begin(), leaf_word_copy_count, levels_.front().begin());
        rebuild_summaries();
    }

    void clear() noexcept {
        for (auto& level : levels_) {
            std::fill(level.begin(), level.end(), std::uint64_t{0});
        }
    }

    void set(std::size_t bit_index) noexcept {
        std::size_t child_index = word_index(bit_index);
        auto& leaf_words = levels_.front();
        const auto previous = leaf_words[child_index];
        leaf_words[child_index] |= bit_mask(bit_index);

        if (previous != 0) {
            return;
        }

        for (std::size_t level = 1; level < levels_.size(); ++level) {
            auto& level_words = levels_[level];
            const auto parent_word_index = word_index(child_index);
            const auto parent_previous = level_words[parent_word_index];
            level_words[parent_word_index] |= bit_mask(child_index);

            if (parent_previous != 0) {
                return;
            }

            child_index = parent_word_index;
        }
    }

    void reset(std::size_t bit_index) noexcept {
        std::size_t child_index = word_index(bit_index);
        auto& leaf_words = levels_.front();
        leaf_words[child_index] &= ~bit_mask(bit_index);

        if (leaf_words[child_index] != 0) {
            return;
        }

        for (std::size_t level = 1; level < levels_.size(); ++level) {
            auto& level_words = levels_[level];
            const auto parent_word_index = word_index(child_index);
            level_words[parent_word_index] &= ~bit_mask(child_index);

            if (level_words[parent_word_index] != 0) {
                return;
            }

            child_index = parent_word_index;
        }
    }

    [[nodiscard]] std::size_t lowest_set_bit() const noexcept {
        std::size_t current_word_index = 0;

        for (std::size_t level = levels_.size() - 1; level > 0; --level) {
            const auto word = levels_[level][current_word_index];
            current_word_index = (current_word_index * occupancy_word_bits) + lowest_bit_index(word);
        }

        const auto leaf_word = levels_.front()[current_word_index];
        return (current_word_index * occupancy_word_bits) + lowest_bit_index(leaf_word);
    }

private:
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>> make_levels(std::size_t bit_count) {
        if (bit_count == 0) {
            return {};
        }

        std::vector<std::size_t> level_word_counts;
        level_word_counts.push_back(words_for_bits(bit_count));

        while (level_word_counts.back() > 1) {
            level_word_counts.push_back(words_for_bits(level_word_counts.back()));
        }

        std::vector<std::vector<std::uint64_t>> levels(level_word_counts.size());
        for (std::size_t level = 0; level < level_word_counts.size(); ++level) {
            levels[level].resize(level_word_counts[level], std::uint64_t{0});
        }

        return levels;
    }

    void rebuild_summaries() noexcept {
        for (std::size_t level = 1; level < levels_.size(); ++level) {
            std::fill(levels_[level].begin(), levels_[level].end(), std::uint64_t{0});
        }

        for (std::size_t level = 1; level < levels_.size(); ++level) {
            const auto& child_words = levels_[level - 1];
            auto& parent_words = levels_[level];

            for (std::size_t child_index = 0; child_index < child_words.size(); ++child_index) {
                if (child_words[child_index] != 0) {
                    parent_words[word_index(child_index)] |= bit_mask(child_index);
                }
            }
        }
    }

    std::size_t bit_count_{0};
    std::vector<std::vector<std::uint64_t>> levels_{};
};

} // namespace detail

// Live fixed-range queue. Use when push/pop operations can be interleaved and
// the priority range is known at compile time. Lower numeric priorities are
// returned first. Equal-priority values are popped FIFO.
template <typename T, std::size_t BucketCount>
class bucket_priority_queue {
    static_assert(BucketCount > 0, "bucket_priority_queue requires at least one bucket");

public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = detail::fifo_bucket<T>;

    [[nodiscard]] static constexpr size_type bucket_count() noexcept {
        return BucketCount;
    }

    [[nodiscard]] static constexpr priority_type min_priority() noexcept {
        return 0;
    }

    [[nodiscard]] static constexpr priority_type max_priority() noexcept {
        return BucketCount - 1;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }

    void clear() {
        for (auto& bucket : buckets_) {
            bucket.clear();
        }

        occupancy_.clear();
        size_ = 0;
    }

    template <typename U>
    void push(priority_type priority, U&& value) {
        validate_priority(priority);
        auto& bucket = buckets_[priority];
        const auto was_empty = bucket.empty();
        bucket.push(std::forward<U>(value));
        if (was_empty) {
            occupancy_.set(priority);
        }
        ++size_;
    }

    template <typename... Args>
    value_type& emplace(priority_type priority, Args&&... args) {
        validate_priority(priority);
        auto& bucket = buckets_[priority];
        const auto was_empty = bucket.empty();
        auto& value = bucket.emplace(std::forward<Args>(args)...);
        if (was_empty) {
            occupancy_.set(priority);
        }
        ++size_;
        return value;
    }

    [[nodiscard]] value_type& top() {
        return bucket_for_top().front();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for_top().front();
    }

    [[nodiscard]] priority_type top_priority() const {
        return top_non_empty_priority();
    }

    void pop() {
        const auto priority = top_non_empty_priority();
        auto& bucket = buckets_[priority];
        bucket.pop();
        --size_;

        if (bucket.empty()) {
            occupancy_.reset(priority);
        }
    }

private:
    [[nodiscard]] bucket_type& bucket_for_top() {
        return buckets_[top_non_empty_priority()];
    }

    [[nodiscard]] const bucket_type& bucket_for_top() const {
        return buckets_[top_non_empty_priority()];
    }

    void validate_priority(priority_type priority) const {
        if (priority >= BucketCount) {
            throw std::out_of_range("priority is out of range for bucket_priority_queue");
        }
    }

    [[nodiscard]] priority_type top_non_empty_priority() const {
        if (empty()) {
            throw std::out_of_range("bucket_priority_queue is empty");
        }

        return occupancy_.lowest_set_bit();
    }

    std::array<bucket_type, BucketCount> buckets_{};
    detail::static_bucket_index<BucketCount> occupancy_{};
    size_type size_{0};
};

// Fill-then-drain fixed-range queue. Use when all pushes for a batch complete
// before popping begins. It avoids maintaining live occupancy metadata and only
// scans buckets while moving between priorities, giving total drain work of
// O(items + buckets). Push throws std::logic_error if called while a non-empty
// drain phase is in progress.
template <typename T, std::size_t BucketCount>
class bulk_bucket_priority_queue {
    static_assert(BucketCount > 0, "bulk_bucket_priority_queue requires at least one bucket");

public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = detail::fifo_bucket<T>;

    [[nodiscard]] static constexpr size_type bucket_count() noexcept {
        return BucketCount;
    }

    [[nodiscard]] static constexpr priority_type min_priority() noexcept {
        return 0;
    }

    [[nodiscard]] static constexpr priority_type max_priority() noexcept {
        return BucketCount - 1;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }

    void clear() {
        for (auto& bucket : buckets_) {
            bucket.clear();
        }

        size_ = 0;
        cursor_ = min_priority();
        draining_ = false;
    }

    template <typename U>
    void push(priority_type priority, U&& value) {
        validate_priority(priority);
        prepare_for_push();
        buckets_[priority].push(std::forward<U>(value));
        ++size_;
    }

    template <typename... Args>
    value_type& emplace(priority_type priority, Args&&... args) {
        validate_priority(priority);
        prepare_for_push();
        auto& bucket = buckets_[priority];
        auto& value = bucket.emplace(std::forward<Args>(args)...);
        ++size_;
        return value;
    }

    [[nodiscard]] value_type& top() {
        return bucket_for_top().front();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for_top().front();
    }

    [[nodiscard]] priority_type top_priority() const {
        return top_non_empty_priority();
    }

    void pop() {
        const auto priority = top_non_empty_priority();
        auto& bucket = buckets_[priority];
        bucket.pop();
        --size_;

        if (bucket.empty() && size_ != 0) {
            ++cursor_;
            advance_cursor();
        }
    }

private:
    void prepare_for_push() {
        if (!draining_) {
            return;
        }
        if (size_ != 0) {
            throw std::logic_error("bulk_bucket_priority_queue cannot be pushed while draining");
        }

        draining_ = false;
        cursor_ = min_priority();
    }

    void start_drain() const {
        if (!draining_) {
            draining_ = true;
            cursor_ = min_priority();
        }

        advance_cursor();
    }

    void advance_cursor() const noexcept {
        while (cursor_ < BucketCount && buckets_[cursor_].empty()) {
            ++cursor_;
        }
    }

    [[nodiscard]] bucket_type& bucket_for_top() {
        return buckets_[top_non_empty_priority()];
    }

    [[nodiscard]] const bucket_type& bucket_for_top() const {
        return buckets_[top_non_empty_priority()];
    }

    void validate_priority(priority_type priority) const {
        if (priority >= BucketCount) {
            throw std::out_of_range("priority is out of range for bulk_bucket_priority_queue");
        }
    }

    [[nodiscard]] priority_type top_non_empty_priority() const {
        if (empty()) {
            throw std::out_of_range("bulk_bucket_priority_queue is empty");
        }

        start_drain();
        return cursor_;
    }

    std::array<bucket_type, BucketCount> buckets_{};
    mutable size_type cursor_{0};
    size_type size_{0};
    mutable bool draining_{false};
};

template <typename T, std::size_t BucketCount>
using static_bucket_priority_queue = bucket_priority_queue<T, BucketCount>;

template <typename T, std::size_t BucketCount>
using bulk_multi_queue_priority_queue = bulk_bucket_priority_queue<T, BucketCount>;

template <typename T, std::size_t BucketCount>
using multi_queue_priority_queue = bucket_priority_queue<T, BucketCount>;

// Advanced customization point. Prefer the aliases below in application code so the
// dynamic queue internals can evolve without forcing call-site changes. BucketContainer
// is expected to be a template<Value, Allocator> with vector-like append,
// iteration, reserve, and move-assignment semantics.
//
// Live dynamic-range queue. Use when push/pop operations can be interleaved and
// the priority range is not fully known up front. The exact-growth alias grows
// to the requested logical range; the default geometric alias keeps spare bucket
// capacity to reduce repeated reallocations.
template <typename T, bool GeometricGrowth, template <typename, typename> class BucketContainer = std::vector>
class dynamic_bucket_priority_queue_base {
public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = detail::fifo_bucket<T, BucketContainer>;

    explicit dynamic_bucket_priority_queue_base(size_type bucket_count = 0) {
        expand_priorities(bucket_count);
    }

    [[nodiscard]] static constexpr priority_type min_priority() noexcept {
        return 0;
    }

    [[nodiscard]] size_type bucket_count() const noexcept {
        return bucket_count_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }

    void clear() {
        for (auto& bucket : buckets_) {
            bucket.clear();
        }

        occupancy_.clear();
        size_ = 0;
    }

    void expand_priorities(size_type bucket_count) {
        if (bucket_count <= bucket_count_) {
            return;
        }

        bucket_count_ = bucket_count;
        reserve_capacity(bucket_count_);
    }

    template <typename U>
    void push(priority_type priority, U&& value) {
        ensure_priority(priority);
        auto& bucket = buckets_[priority];
        const auto was_empty = bucket.empty();
        bucket.push(std::forward<U>(value));
        if (was_empty) {
            occupancy_.set(priority);
        }
        ++size_;
    }

    template <typename... Args>
    value_type& emplace(priority_type priority, Args&&... args) {
        ensure_priority(priority);
        auto& bucket = buckets_[priority];
        const auto was_empty = bucket.empty();
        auto& value = bucket.emplace(std::forward<Args>(args)...);
        if (was_empty) {
            occupancy_.set(priority);
        }
        ++size_;
        return value;
    }

    [[nodiscard]] value_type& top() {
        return bucket_for_top().front();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for_top().front();
    }

    [[nodiscard]] priority_type top_priority() const {
        return top_non_empty_priority();
    }

    void pop() {
        const auto priority = top_non_empty_priority();
        auto& bucket = buckets_[priority];
        bucket.pop();
        --size_;

        if (bucket.empty()) {
            occupancy_.reset(priority);
        }
    }

private:
    void reserve_capacity(size_type bucket_count) {
        if (bucket_count <= capacity_) {
            return;
        }

        capacity_ = GeometricGrowth ? detail::geometric_capacity(bucket_count) : bucket_count;
        buckets_.resize(capacity_);
        occupancy_.expand(capacity_);
    }

    void ensure_priority(priority_type priority) {
        if (priority >= bucket_count_) {
            expand_priorities(priority + 1);
        }
    }

    [[nodiscard]] bucket_type& bucket_for_top() {
        return buckets_[top_non_empty_priority()];
    }

    [[nodiscard]] const bucket_type& bucket_for_top() const {
        return buckets_[top_non_empty_priority()];
    }

    [[nodiscard]] priority_type top_non_empty_priority() const {
        if (empty()) {
            throw std::out_of_range("dynamic_bucket_priority_queue is empty");
        }

        return occupancy_.lowest_set_bit();
    }

    std::vector<bucket_type> buckets_{};
    detail::dynamic_bucket_index occupancy_{};
    size_type bucket_count_{0};
    size_type capacity_{0};
    size_type size_{0};
};

template <typename T>
using dynamic_bucket_priority_queue_exact_growth = dynamic_bucket_priority_queue_base<T, false>;

// Stable public aliases for the flat dynamic queue family.
template <typename T>
using dynamic_bucket_priority_queue_geometric = dynamic_bucket_priority_queue_base<T, true>;

template <typename T>
using dynamic_bucket_priority_queue = dynamic_bucket_priority_queue_geometric<T>;

template <typename T>
using dynamic_multi_queue_priority_queue = dynamic_bucket_priority_queue<T>;

// Advanced sparse-range customization point. Prefer the aliases below in application
// code unless you intentionally want to opt into a specific paged growth policy.
//
// Live sparse-range queue. Use when the logical priority range can be large but
// active priorities tend to cluster into a limited number of 64-priority pages.
template <typename T, bool GeometricGrowth>
class paged_dynamic_bucket_priority_queue_base {
    static constexpr std::size_t page_bucket_count = detail::occupancy_word_bits;

public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = detail::fifo_bucket<T>;

    explicit paged_dynamic_bucket_priority_queue_base(size_type bucket_count = 0) {
        expand_priorities(bucket_count);
    }

    [[nodiscard]] static constexpr priority_type min_priority() noexcept {
        return 0;
    }

    [[nodiscard]] size_type bucket_count() const noexcept {
        return bucket_count_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }

    void clear() {
        for (auto& page : pages_) {
            if (!page) {
                continue;
            }

            for (auto& bucket : page->buckets_) {
                bucket.clear();
            }

            page->occupancy_mask_ = 0;
        }

        non_empty_pages_.clear();
        size_ = 0;
    }

    void expand_priorities(size_type bucket_count) {
        if (bucket_count <= bucket_count_) {
            return;
        }

        bucket_count_ = bucket_count;
        reserve_page_capacity(detail::words_for_bits(bucket_count_));
    }

    template <typename U>
    void push(priority_type priority, U&& value) {
        ensure_priority(priority);

        const auto next_page_index = page_index(priority);
        const auto next_bucket_offset = bucket_offset(priority);
        auto& page = ensure_page(next_page_index);
        auto& bucket = page.buckets_[next_bucket_offset];
        const auto bucket_was_empty = bucket.empty();
        bucket.push(std::forward<U>(value));

        if (bucket_was_empty) {
            page.occupancy_mask_ |= detail::bit_mask(next_bucket_offset);
            non_empty_pages_.set(next_page_index);
        }

        ++size_;
    }

    template <typename... Args>
    value_type& emplace(priority_type priority, Args&&... args) {
        ensure_priority(priority);

        const auto next_page_index = page_index(priority);
        const auto next_bucket_offset = bucket_offset(priority);
        auto& page = ensure_page(next_page_index);
        auto& bucket = page.buckets_[next_bucket_offset];
        const auto bucket_was_empty = bucket.empty();
        auto& value = bucket.emplace(std::forward<Args>(args)...);

        if (bucket_was_empty) {
            page.occupancy_mask_ |= detail::bit_mask(next_bucket_offset);
            non_empty_pages_.set(next_page_index);
        }

        ++size_;
        return value;
    }

    [[nodiscard]] value_type& top() {
        return bucket_for_priority(top_non_empty_priority()).front();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for_priority(top_non_empty_priority()).front();
    }

    [[nodiscard]] priority_type top_priority() const {
        return top_non_empty_priority();
    }

    void pop() {
        const auto priority = top_non_empty_priority();
        auto& page = *pages_[page_index(priority)];
        auto& bucket = page.buckets_[bucket_offset(priority)];
        bucket.pop();
        --size_;

        if (bucket.empty()) {
            page.occupancy_mask_ &= ~detail::bit_mask(bucket_offset(priority));

            if (page.occupancy_mask_ == 0) {
                non_empty_pages_.reset(page_index(priority));
            }
        }
    }

private:
    struct page_state {
        std::array<bucket_type, page_bucket_count> buckets_{};
        std::uint64_t occupancy_mask_{0};
    };

    static void require_valid_page_count(size_type page_count) {
        if (page_count == 0) {
            return;
        }
    }

    void reserve_page_capacity(size_type page_count) {
        if (page_count <= page_capacity_) {
            return;
        }

        require_valid_page_count(page_count);
        page_capacity_ = GeometricGrowth ? detail::geometric_capacity(page_count) : page_count;
        pages_.resize(page_capacity_);
        non_empty_pages_.expand(page_capacity_);
    }

    void ensure_priority(priority_type priority) {
        if (priority >= bucket_count_) {
            expand_priorities(priority + 1);
        }
    }

    [[nodiscard]] static constexpr size_type page_index(priority_type priority) noexcept {
        return detail::word_index(priority);
    }

    [[nodiscard]] static constexpr size_type bucket_offset(priority_type priority) noexcept {
        return priority % page_bucket_count;
    }

    [[nodiscard]] page_state& ensure_page(size_type next_page_index) {
        if (!pages_[next_page_index]) {
            pages_[next_page_index] = std::make_unique<page_state>();
        }

        return *pages_[next_page_index];
    }

    [[nodiscard]] bucket_type& bucket_for_priority(priority_type priority) {
        return pages_[page_index(priority)]->buckets_[bucket_offset(priority)];
    }

    [[nodiscard]] const bucket_type& bucket_for_priority(priority_type priority) const {
        return pages_[page_index(priority)]->buckets_[bucket_offset(priority)];
    }

    [[nodiscard]] priority_type top_non_empty_priority() const {
        if (empty()) {
            throw std::out_of_range("paged_dynamic_bucket_priority_queue is empty");
        }

        const auto lowest_page_index = non_empty_pages_.lowest_set_bit();
        const auto& page = *pages_[lowest_page_index];
        const auto lowest_bucket_offset = detail::lowest_bit_index(page.occupancy_mask_);
        return (lowest_page_index * page_bucket_count) + lowest_bucket_offset;
    }

    std::vector<std::unique_ptr<page_state>> pages_{};
    detail::dynamic_bucket_index non_empty_pages_{};
    size_type bucket_count_{0};
    size_type page_capacity_{0};
    size_type size_{0};
};

template <typename T>
using paged_dynamic_bucket_priority_queue = paged_dynamic_bucket_priority_queue_base<T, false>;

template <typename T>
using paged_dynamic_bucket_priority_queue_geometric = paged_dynamic_bucket_priority_queue_base<T, true>;

// Registered sparse queue. Use when the application can register the small set
// of priorities it will reuse and pass handles on push/emplace. This avoids
// materializing the full priority range and skips per-push range growth checks.
template <typename T>
class registered_bucket_priority_queue {
private:
    struct page_state;

public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = detail::fifo_bucket<T>;

    struct priority_handle {
        priority_handle() = default;

        [[nodiscard]] bool valid() const noexcept {
            return page_ != nullptr;
        }

        [[nodiscard]] priority_type priority() const noexcept {
            return priority_;
        }

        friend bool operator==(const priority_handle&, const priority_handle&) = default;

    private:
        const registered_bucket_priority_queue* owner_{nullptr};
        page_state* page_{nullptr};
        std::size_t bucket_offset_{0};
        priority_type priority_{0};

        priority_handle(
            const registered_bucket_priority_queue* owner,
            page_state* page,
            std::size_t bucket_offset,
            priority_type priority
        )
            : owner_(owner),
              page_(page),
              bucket_offset_(bucket_offset),
              priority_(priority) {
        }

        friend class registered_bucket_priority_queue;
    };

    [[nodiscard]] static constexpr priority_type min_priority() noexcept {
        return 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }

    [[nodiscard]] size_type registered_priority_count() const noexcept {
        return registry_.size();
    }

    [[nodiscard]] priority_handle register_priority(priority_type priority) {
        if (const auto it = registry_.find(priority); it != registry_.end()) {
            return it->second;
        }

        const auto page_index = detail::word_index(priority);
        page_state* page = nullptr;

        if (const auto it = pages_.find(page_index); it != pages_.end()) {
            page = it->second.get();
        } else {
            auto inserted = pages_.emplace(page_index, std::make_unique<page_state>(page_index));
            page = inserted.first->second.get();
        }

        const auto bucket_offset = priority % detail::occupancy_word_bits;
        const auto handle = priority_handle(this, page, bucket_offset, priority);

        registry_.emplace(priority, handle);
        return handle;
    }

    void clear() {
        for (auto& [_, page] : pages_) {
            for (auto& bucket : page->buckets_) {
                bucket.clear();
            }

            page->occupancy_mask_ = 0;
        }

        active_pages_ = {};
        size_ = 0;
    }

    template <typename U>
    void push(const priority_handle& handle, U&& value) {
        validate_handle(handle);
        auto& bucket = bucket_for(handle);
        const auto bucket_was_empty = bucket.empty();
        bucket.push(std::forward<U>(value));

        if (bucket_was_empty) {
            mark_non_empty(handle);
        }

        ++size_;
    }

    template <typename... Args>
    value_type& emplace(const priority_handle& handle, Args&&... args) {
        validate_handle(handle);
        auto& bucket = bucket_for(handle);
        const auto bucket_was_empty = bucket.empty();
        auto& value = bucket.emplace(std::forward<Args>(args)...);

        if (bucket_was_empty) {
            mark_non_empty(handle);
        }

        ++size_;
        return value;
    }

    [[nodiscard]] value_type& top() {
        return bucket_for(top_non_empty_handle()).front();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for(top_non_empty_handle()).front();
    }

    [[nodiscard]] priority_type top_priority() const {
        return top_non_empty_handle().priority_;
    }

    void pop() {
        const auto handle = top_non_empty_handle();
        auto& bucket = bucket_for(handle);
        bucket.pop();
        --size_;

        if (bucket.empty()) {
            mark_empty(handle);
        }
    }

private:
    struct page_state {
        explicit page_state(std::size_t page_index)
            : page_index_(page_index) {
        }

        std::size_t page_index_{0};
        std::uint64_t occupancy_mask_{0};
        std::size_t activation_generation_{0};
        std::array<bucket_type, detail::occupancy_word_bits> buckets_{};
    };

    struct active_page_entry {
        page_state* page{nullptr};
        std::size_t activation_generation{0};
    };

    struct lower_page {
        [[nodiscard]] bool operator()(const active_page_entry& left, const active_page_entry& right) const noexcept {
            return left.page->page_index_ > right.page->page_index_;
        }
    };

    void validate_handle(const priority_handle& handle) const {
        if (!handle.valid() || handle.owner_ != this) {
            throw std::invalid_argument("priority_handle does not belong to this registered_bucket_priority_queue");
        }
    }

    [[nodiscard]] bucket_type& bucket_for(const priority_handle& handle) {
        return handle.page_->buckets_[handle.bucket_offset_];
    }

    [[nodiscard]] const bucket_type& bucket_for(const priority_handle& handle) const {
        return handle.page_->buckets_[handle.bucket_offset_];
    }

    void mark_non_empty(const priority_handle& handle) {
        auto& page = *handle.page_;
        const auto page_was_empty = page.occupancy_mask_ == 0;
        page.occupancy_mask_ |= detail::bit_mask(handle.bucket_offset_);

        if (page_was_empty) {
            ++page.activation_generation_;
            active_pages_.push(active_page_entry{
                &page,
                page.activation_generation_,
            });
        }
    }

    void mark_empty(const priority_handle& handle) noexcept {
        handle.page_->occupancy_mask_ &= ~detail::bit_mask(handle.bucket_offset_);
    }

    void refresh_active_pages() const {
        while (!active_pages_.empty()) {
            const auto& entry = active_pages_.top();
            if (entry.page->activation_generation_ == entry.activation_generation && entry.page->occupancy_mask_ != 0) {
                return;
            }

            active_pages_.pop();
        }
    }

    [[nodiscard]] priority_handle top_non_empty_handle() const {
        if (empty()) {
            throw std::out_of_range("registered_bucket_priority_queue is empty");
        }

        refresh_active_pages();
        const auto& page = *active_pages_.top().page;
        const auto bucket_offset = detail::lowest_bit_index(page.occupancy_mask_);

        return priority_handle(
            this,
            active_pages_.top().page,
            bucket_offset,
            (page.page_index_ * detail::occupancy_word_bits) + bucket_offset
        );
    }

    std::unordered_map<priority_type, priority_handle> registry_{};
    std::unordered_map<std::size_t, std::unique_ptr<page_state>> pages_{};
    mutable std::priority_queue<active_page_entry, std::vector<active_page_entry>, lower_page> active_pages_{};
    size_type size_{0};
};

template <typename T>
using registered_multi_queue_priority_queue = registered_bucket_priority_queue<T>;

} // namespace cpp_pq
