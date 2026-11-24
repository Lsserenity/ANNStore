#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

#include "spdlog/spdlog.h"

#include "common/query_io_events.hpp"

namespace annfs::tests {

class QueryLatencyTracker : public annfs::io::QueryIoEventListener {
public:
    using Clock = std::chrono::steady_clock;

    explicit QueryLatencyTracker(size_t query_cnt)
        : start_us_(query_cnt, -1),
          end_us_(query_cnt, -1),
          outstanding_ios_(std::make_unique<std::atomic<int>[]>(query_cnt)),
          cpu_done_(std::make_unique<std::atomic<bool>[]>(query_cnt)),
          end_recorded_(std::make_unique<std::atomic<bool>[]>(query_cnt))
    {
        base_tp_ = Clock::now();
        for (size_t i = 0; i < query_cnt; ++i) {
            outstanding_ios_[i].store(0, std::memory_order_relaxed);
            cpu_done_[i].store(false, std::memory_order_relaxed);
            end_recorded_[i].store(false, std::memory_order_relaxed);
        }
    }

    void RecordQueryStart(size_t query_id)
    {
        if (!valid_query(query_id)) {
            return;
        }
        if (start_us_[query_id] < 0) {
            start_us_[query_id] = elapsed_us();
            // SPDLOG_INFO("QueryLatencyTracker start query {} at {} us",
            //             query_id,
            //             start_us_[query_id]);
        }
    }

    void MarkQueryCpuDone(size_t query_id)
    {
        if (!valid_query(query_id)) {
            return;
        }
        cpu_done_[query_id].store(true, std::memory_order_release);
        try_finalize(query_id);
    }

    void OnQueryIoSubmitted(int query_id, int count) override
    {
        if (count <= 0 || !valid_query(query_id)) {
            return;
        }
        outstanding_ios_[query_id].fetch_add(count, std::memory_order_acq_rel);
    }

    void OnQueryIoCompleted(int query_id, int count) override
    {
        if (count <= 0 || !valid_query(query_id)) {
            return;
        }
        for (int i = 0; i < count; ++i) {
            int prev = outstanding_ios_[query_id].fetch_sub(
                1, std::memory_order_acq_rel);
            if (prev <= 0) {
                outstanding_ios_[query_id].store(
                    0, std::memory_order_release);
                break;
            }
            if (prev == 1) {
                try_finalize(query_id);
            }
        }
    }

    const std::vector<long long>& StartTimesUs() const { return start_us_; }
    const std::vector<long long>& EndTimesUs() const { return end_us_; }

private:
    bool valid_query(size_t query_id) const
    {
        return query_id < start_us_.size();
    }

    bool valid_query(int query_id) const
    {
        return query_id >= 0 &&
               static_cast<size_t>(query_id) < start_us_.size();
    }

    long long elapsed_us() const
    {
        auto now = Clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   now - base_tp_)
            .count();
    }

    void try_finalize(size_t query_id)
    {
        if (!valid_query(query_id)) {
            return;
        }
        if (!cpu_done_[query_id].load(std::memory_order_acquire)) {
            return;
        }
        if (outstanding_ios_[query_id].load(std::memory_order_acquire) != 0) {
            return;
        }
        bool expected = false;
        if (end_recorded_[query_id].compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            end_us_[query_id] = elapsed_us();
            // SPDLOG_INFO("QueryLatencyTracker end query {} at {} us",
            //             query_id,
            //             end_us_[query_id]);
        }
    }

    Clock::time_point base_tp_;
    std::vector<long long> start_us_;
    std::vector<long long> end_us_;
    std::unique_ptr<std::atomic<int>[]> outstanding_ios_;
    std::unique_ptr<std::atomic<bool>[]> cpu_done_;
    std::unique_ptr<std::atomic<bool>[]> end_recorded_;
};

class QueryLatencyScope {
public:
    QueryLatencyScope(QueryLatencyTracker& tracker, size_t query_id)
        : tracker_(tracker),
          query_id_(query_id),
          active_(true)
    {}

    ~QueryLatencyScope()
    {
        if (active_) {
            tracker_.MarkQueryCpuDone(query_id_);
        }
    }

    QueryLatencyScope(const QueryLatencyScope&) = delete;
    QueryLatencyScope& operator=(const QueryLatencyScope&) = delete;

    QueryLatencyScope(QueryLatencyScope&&) = delete;
    QueryLatencyScope& operator=(QueryLatencyScope&&) = delete;

private:
    QueryLatencyTracker& tracker_;
    size_t query_id_;
    bool active_;
};

} // namespace annfs::tests
