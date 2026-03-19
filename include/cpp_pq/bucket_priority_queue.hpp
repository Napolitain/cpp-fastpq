#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
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

[[nodiscard]] inline constexpr std::size_t highest_bit_index(std::uint64_t value) noexcept {
    return (occupancy_word_bits - 1) - static_cast<std::size_t>(std::countl_zero(value));
}

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

    [[nodiscard]] std::size_t highest_set_bit() const noexcept {
        std::size_t current_word_index = 0;

        for (std::size_t level = level_count - 1; level > 0; --level) {
            const auto word = level_data(level)[current_word_index];
            current_word_index = (current_word_index * occupancy_word_bits) + highest_bit_index(word);
        }

        const auto leaf_word = level_data(0)[current_word_index];
        return (current_word_index * occupancy_word_bits) + highest_bit_index(leaf_word);
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

    [[nodiscard]] std::size_t highest_set_bit() const noexcept {
        std::size_t current_word_index = 0;

        for (std::size_t level = levels_.size() - 1; level > 0; --level) {
            const auto word = levels_[level][current_word_index];
            current_word_index = (current_word_index * occupancy_word_bits) + highest_bit_index(word);
        }

        const auto leaf_word = levels_.front()[current_word_index];
        return (current_word_index * occupancy_word_bits) + highest_bit_index(leaf_word);
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

template <typename T, std::size_t BucketCount>
class bucket_priority_queue {
    static_assert(BucketCount > 0, "bucket_priority_queue requires at least one bucket");

public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = std::vector<T>;

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
        buckets_[priority].push_back(std::forward<U>(value));
        occupancy_.set(priority);
        ++size_;
    }

    template <typename... Args>
    value_type& emplace(priority_type priority, Args&&... args) {
        validate_priority(priority);
        auto& bucket = buckets_[priority];
        bucket.emplace_back(std::forward<Args>(args)...);
        occupancy_.set(priority);
        ++size_;
        return bucket.back();
    }

    [[nodiscard]] value_type& top() {
        return bucket_for_top().back();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for_top().back();
    }

    [[nodiscard]] priority_type top_priority() const {
        return highest_non_empty_priority();
    }

    void pop() {
        const auto priority = highest_non_empty_priority();
        auto& bucket = buckets_[priority];
        bucket.pop_back();
        --size_;

        if (bucket.empty()) {
            occupancy_.reset(priority);
        }
    }

private:
    [[nodiscard]] bucket_type& bucket_for_top() {
        return buckets_[highest_non_empty_priority()];
    }

    [[nodiscard]] const bucket_type& bucket_for_top() const {
        return buckets_[highest_non_empty_priority()];
    }

    void validate_priority(priority_type priority) const {
        if (priority >= BucketCount) {
            throw std::out_of_range("priority is out of range for bucket_priority_queue");
        }
    }

    [[nodiscard]] priority_type highest_non_empty_priority() const {
        if (empty()) {
            throw std::out_of_range("bucket_priority_queue is empty");
        }

        return occupancy_.highest_set_bit();
    }

    std::array<bucket_type, BucketCount> buckets_{};
    detail::static_bucket_index<BucketCount> occupancy_{};
    size_type size_{0};
};

template <typename T, std::size_t BucketCount>
using static_bucket_priority_queue = bucket_priority_queue<T, BucketCount>;

template <typename T, std::size_t BucketCount>
using multi_queue_priority_queue = bucket_priority_queue<T, BucketCount>;

template <typename T>
class dynamic_bucket_priority_queue {
public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = std::vector<T>;

    explicit dynamic_bucket_priority_queue(size_type bucket_count = 0) {
        expand_priorities(bucket_count);
    }

    [[nodiscard]] static constexpr priority_type min_priority() noexcept {
        return 0;
    }

    [[nodiscard]] size_type bucket_count() const noexcept {
        return buckets_.size();
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
        if (bucket_count <= buckets_.size()) {
            return;
        }

        buckets_.resize(bucket_count);
        occupancy_.expand(bucket_count);
    }

    template <typename U>
    void push(priority_type priority, U&& value) {
        ensure_priority(priority);
        buckets_[priority].push_back(std::forward<U>(value));
        occupancy_.set(priority);
        ++size_;
    }

    template <typename... Args>
    value_type& emplace(priority_type priority, Args&&... args) {
        ensure_priority(priority);
        auto& bucket = buckets_[priority];
        bucket.emplace_back(std::forward<Args>(args)...);
        occupancy_.set(priority);
        ++size_;
        return bucket.back();
    }

    [[nodiscard]] value_type& top() {
        return bucket_for_top().back();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for_top().back();
    }

    [[nodiscard]] priority_type top_priority() const {
        return highest_non_empty_priority();
    }

    void pop() {
        const auto priority = highest_non_empty_priority();
        auto& bucket = buckets_[priority];
        bucket.pop_back();
        --size_;

        if (bucket.empty()) {
            occupancy_.reset(priority);
        }
    }

private:
    void ensure_priority(priority_type priority) {
        if (priority >= buckets_.size()) {
            expand_priorities(priority + 1);
        }
    }

    [[nodiscard]] bucket_type& bucket_for_top() {
        return buckets_[highest_non_empty_priority()];
    }

    [[nodiscard]] const bucket_type& bucket_for_top() const {
        return buckets_[highest_non_empty_priority()];
    }

    [[nodiscard]] priority_type highest_non_empty_priority() const {
        if (empty()) {
            throw std::out_of_range("dynamic_bucket_priority_queue is empty");
        }

        return occupancy_.highest_set_bit();
    }

    std::vector<bucket_type> buckets_{};
    detail::dynamic_bucket_index occupancy_{};
    size_type size_{0};
};

template <typename T>
using dynamic_multi_queue_priority_queue = dynamic_bucket_priority_queue<T>;

template <typename T>
class registered_bucket_priority_queue {
private:
    struct page_state;

public:
    using value_type = T;
    using size_type = std::size_t;
    using priority_type = std::size_t;
    using bucket_type = std::vector<T>;

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
        bucket.push_back(std::forward<U>(value));

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
        bucket.emplace_back(std::forward<Args>(args)...);

        if (bucket_was_empty) {
            mark_non_empty(handle);
        }

        ++size_;
        return bucket.back();
    }

    [[nodiscard]] value_type& top() {
        return bucket_for(highest_non_empty_handle()).back();
    }

    [[nodiscard]] const value_type& top() const {
        return bucket_for(highest_non_empty_handle()).back();
    }

    [[nodiscard]] priority_type top_priority() const {
        return highest_non_empty_handle().priority_;
    }

    void pop() {
        const auto handle = highest_non_empty_handle();
        auto& bucket = bucket_for(handle);
        bucket.pop_back();
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

    struct higher_page {
        [[nodiscard]] bool operator()(const active_page_entry& left, const active_page_entry& right) const noexcept {
            return left.page->page_index_ < right.page->page_index_;
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

    [[nodiscard]] priority_handle highest_non_empty_handle() const {
        if (empty()) {
            throw std::out_of_range("registered_bucket_priority_queue is empty");
        }

        refresh_active_pages();
        const auto& page = *active_pages_.top().page;
        const auto bucket_offset = detail::highest_bit_index(page.occupancy_mask_);

        return priority_handle(
            this,
            active_pages_.top().page,
            bucket_offset,
            (page.page_index_ * detail::occupancy_word_bits) + bucket_offset
        );
    }

    std::unordered_map<priority_type, priority_handle> registry_{};
    std::unordered_map<std::size_t, std::unique_ptr<page_state>> pages_{};
    mutable std::priority_queue<active_page_entry, std::vector<active_page_entry>, higher_page> active_pages_{};
    size_type size_{0};
};

template <typename T>
using registered_multi_queue_priority_queue = registered_bucket_priority_queue<T>;

} // namespace cpp_pq
