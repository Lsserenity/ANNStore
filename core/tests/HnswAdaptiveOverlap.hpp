#pragma once

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "early-stop-overlap-support.hpp"
#include "tests/HnswExperimentRunner.hpp"

namespace annfs::tests {

using AdaptiveStableIterConfig = ::AdaptiveStableIterConfig;

inline void PrintAdaptiveOverlapUsage(const char* binary)
{
    std::cerr
        << "Usage: " << binary
        << " <config_json_path> <dataset_name> [num_threads] [repeat_id]\n"
        << "       [--stableiter <value>] [--stable_hops <value>]\n"
        << "       [--stableiter_phase <value>] [--stableiter_init <value>] [--stableiter_min <value>]\n"
        << "       [--stableiter_max <value>] [--stableiter_dec_step <value>] [--stableiter_inc_step <value>]\n"
        << "       [--no_prefetch_ratio_high <value>]\n"
        << "       [--io_issue_ratio_high <value>]\n"
        << "       [--extra_io_ratio_high <value>]  # alias: io_issue_ratio_high = 1 + extra\n"
        << "       [--adjust_query_percent <value>]  # 前多少 query 参与调整（0=不限制）\n"
        << "       [--adaptive_stop_on_target <on|off>]\n"
        << "       [--adaptive_stop_target_phases <value>]  # 连续达标 phase 数\n"
        << "       [--adaptive_stop_on_conflict_phases <value>]  # 连续冲突 phase 数\n"
        << "  or:  " << binary
        << " batch <config_json_path> <batch_plan_json> [same adaptive flags]\n"
        << "  or:  " << binary << " warmup <config_json_path>\n";
}

class AdaptiveOverlapIOStrategy : public ExperimentStrategy {
public:
    AdaptiveOverlapIOStrategy() = default;
    explicit AdaptiveOverlapIOStrategy(AdaptiveStableIterConfig cfg)
        : cfg_(std::move(cfg))
    {}

    std::string Name() const override { return "HnswOffsetOverlapAdaptive"; }
    std::string ResultFolderName() const override
    {
        return "OffsetOverlapAdaptive_results";
    }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    bool RequiresPostQueryIoWait() const override { return true; }

    std::string ModeSubdir(const HnswIoState& state) const override
    {
        std::string worker = state.use_shared_io_worker ? "shared" : "per_thread";
        return "adaptive_overlap_per_thread_wait_all_" + worker;
    }

    void OnBeforeRun(const HnswIoState& state) override
    {
        if (state.pc) {
            cfg_ = state.pc->adaptive_cfg;
        }
        ann_us_total_.store(0, std::memory_order_relaxed);
        issued_prefetch_labels_total_.store(0, std::memory_order_relaxed);
        query_total_.store(0, std::memory_order_relaxed);

        phase_records_.clear();
        phase_query_count_ = 0;
        phase_no_prefetch_count_ = 0;
        phase_hnsw_sum_us_ = 0.0;
        phase_issued_prefetch_labels_sum_ = 0;
        phase_topk_labels_sum_ = 0;
        phase_id_ = 0;
        adjust_frozen_ = false;
        adjusted_query_count_ = 0;
        consecutive_target_phases_ = 0;
        consecutive_conflict_phases_ = 0;

        if (cfg_.adjust_query_percent > 0.0 && cfg_.adjust_query_percent < 100.0) {
            const double qn = static_cast<double>(std::max(state.query_cnt, 0));
            adjust_budget_queries_ = static_cast<std::size_t>(
                std::floor((qn * cfg_.adjust_query_percent) / 100.0));
        } else {
            adjust_budget_queries_ = 0;
        }

        int fallback = state.pc ? state.pc->dataset.stable_iter : 30;
        int initial = cfg_.initial_stable_iter > 0 ? cfg_.initial_stable_iter : fallback;
        current_stable_iter_ = ClampIter(initial);

        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[AdaptiveOverlap] init stable_iter={}, phase={}, min={}, max={}, "
                "ratio_high={}, io_issue_ratio_high={}, dec_step={}, inc_step={}, "
                "adjust_query_percent={}, stableiter_init={}, dataset_stable_iter={}, "
                "stop_on_target={}, stop_target_phases={}, stop_conflict_phases={}",
                current_stable_iter_,
                cfg_.adjust_every_queries,
                cfg_.min_stable_iter,
                cfg_.max_stable_iter,
                cfg_.no_prefetch_ratio_high,
                cfg_.io_issue_ratio_high,
                cfg_.decrease_step,
                cfg_.increase_step,
                cfg_.adjust_query_percent,
                cfg_.initial_stable_iter,
                fallback,
                cfg_.stop_eval_on_target,
                cfg_.stop_eval_target_consecutive_phases,
                cfg_.stop_eval_on_conflict_phases);
        }
    }

    void OnAfterRun(const HnswIoState& state,
                    const std::string& output_dir) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        FlushPhaseIfNeeded(state);
        std::ofstream out(output_dir + "/AdaptiveStableIterPhases.csv");
        if (!out.is_open()) {
            return;
        }
        out << "phase_id,queries,no_prefetch,no_prefetch_ratio,issued_prefetch_labels,topk_labels,io_issue_ratio,extra_io_ratio,avg_hnsw_us,stable_iter_before,stable_iter_after,action\n";
        for (const auto& r : phase_records_) {
            out << r.phase_id << ","
                << r.query_count << ","
                << r.no_prefetch_count << ","
                << r.no_prefetch_ratio << ","
                << r.issued_prefetch_labels << ","
                << r.topk_labels << ","
                << r.io_issue_ratio << ","
                << r.extra_io_ratio << ","
                << r.avg_hnsw_us << ","
                << r.stable_iter_before << ","
                << r.stable_iter_after << ","
                << r.action << "\n";
        }

        const std::size_t qn = query_total_.load(std::memory_order_relaxed);
        const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
        const double avg_ann =
            static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
        std::cout << "[AdaptiveOverlapStrategy] avg_ann_us=" << avg_ann
                  << " issued_prefetch_labels="
                  << issued_prefetch_labels_total_.load(std::memory_order_relaxed)
                  << "\n";
    }

    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>&) override
    {
        auto& ioer = IoWorkerForThread(state, thread_id);
        const int stable_iter = current_stable_iter_;
        const bool adjust_this_query = ShouldAdjustThisQuery();

        AdaptiveMappedQuerySignals sig;
        int pending_prefetch_ios = 0;
        std::size_t issued_prefetch_labels = 0;
        std::size_t mapped_missing = 0;
        std::size_t topk_labels = 0;
        auto ann_start = Clock::now();
        auto sent_prefetch_labels = state.hnsw_index->searchKnnPrefetchMappedBaseline(
            state.QueryVector(row),
            cur_k,
            stable_iter,
            &ioer,
            state.pc->dataset.least_else_size,
            nullptr,
            &pending_prefetch_ios,
            static_cast<int>(row),
            &issued_prefetch_labels,
            &mapped_missing,
            &sig.prefetch_triggered,
            &topk_labels,
            state.pc && !state.pc->skip_recall);
        auto ann_end = Clock::now();
        const double hnsw_us = ElapsedUs(ann_start, ann_end);
        sig.topk_labels = std::max<std::size_t>(topk_labels, 1);
        sig.issued_prefetch_labels = issued_prefetch_labels;
        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[AdaptiveOverlap][query {}] stable_iter={}, triggered={}, "
                "issued_prefetch_labels={}, topk_labels={}, pending={}, missing={}, adjust={}",
                row,
                stable_iter,
                sig.prefetch_triggered,
                sig.issued_prefetch_labels,
                sig.topk_labels,
                pending_prefetch_ios,
                mapped_missing,
                adjust_this_query);
        }

        if (adjust_this_query) {
            ObserveAndAdjust(state, stable_iter, sig, hnsw_us);
            adjusted_query_count_++;
        }

        ExperimentQueryResult result;
        result.hnsw_time_us = hnsw_us;
        result.io_time_us = 0.0;
        result.overlap = std::make_unique<OverlapQueryResult>();
        result.overlap->issued_label_set = std::move(sent_prefetch_labels);
        ann_us_total_.fetch_add(
            static_cast<std::uint64_t>(result.hnsw_time_us),
            std::memory_order_relaxed);
        issued_prefetch_labels_total_.fetch_add(
            static_cast<std::uint64_t>(sig.issued_prefetch_labels),
            std::memory_order_relaxed);
        query_total_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

private:
    using Clock = std::chrono::high_resolution_clock;

    struct AdaptiveMappedQuerySignals {
        std::size_t issued_prefetch_labels = 0;
        std::size_t topk_labels = 0;
        bool prefetch_triggered = false;
    };

    bool ShouldAdjustThisQuery()
    {
        if (adjust_frozen_) {
            return false;
        }
        if (adjust_budget_queries_ > 0 &&
            adjusted_query_count_ >= adjust_budget_queries_) {
            adjust_frozen_ = true;
            return false;
        }
        return true;
    }

    struct PhaseRecord {
        int phase_id = 0;
        int query_count = 0;
        int no_prefetch_count = 0;
        double no_prefetch_ratio = 0.0;
        std::size_t issued_prefetch_labels = 0;
        std::size_t topk_labels = 0;
        double io_issue_ratio = 0.0;
        double extra_io_ratio = 0.0;
        double avg_hnsw_us = 0.0;
        int stable_iter_before = 0;
        int stable_iter_after = 0;
        std::string action;
    };

    static double ElapsedUs(const Clock::time_point& start,
                            const Clock::time_point& end)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    int ClampIter(int v) const
    {
        return std::max(cfg_.min_stable_iter, std::min(cfg_.max_stable_iter, v));
    }

    void ObserveAndAdjust(const HnswIoState& state,
                          int stable_iter_used,
                          const AdaptiveMappedQuerySignals& sig,
                          double hnsw_us)
    {
        std::lock_guard<std::mutex> lock(mu_);
        phase_query_count_ += 1;
        phase_hnsw_sum_us_ += hnsw_us;
        phase_issued_prefetch_labels_sum_ += sig.issued_prefetch_labels;
        phase_topk_labels_sum_ += std::max<std::size_t>(1, sig.topk_labels);
        if (!sig.prefetch_triggered) {
            phase_no_prefetch_count_ += 1;
        }
        if (phase_query_count_ < cfg_.adjust_every_queries) {
            return;
        }
        FinalizeCurrentPhase(state, stable_iter_used);
    }

    void FlushPhaseIfNeeded(const HnswIoState& state)
    {
        if (phase_query_count_ <= 0) {
            return;
        }
        FinalizeCurrentPhase(state, current_stable_iter_);
    }

    void FinalizeCurrentPhase(const HnswIoState& state, int stable_iter_before)
    {
        const double q = static_cast<double>(phase_query_count_);
        const double no_prefetch_ratio = static_cast<double>(phase_no_prefetch_count_) / q;
        const double issued_prefetch_labels =
            static_cast<double>(phase_issued_prefetch_labels_sum_);
        const double topk_labels = static_cast<double>(phase_topk_labels_sum_);
        const double io_issue_ratio =
            topk_labels > 0.0 ? (issued_prefetch_labels / topk_labels) : 0.0;
        const double extra_io_ratio =
            topk_labels > 0.0
                ? std::max(0.0, (issued_prefetch_labels - topk_labels) / topk_labels)
                : 0.0;

        bool should_decrease = no_prefetch_ratio > cfg_.no_prefetch_ratio_high;
        bool should_increase = io_issue_ratio > cfg_.io_issue_ratio_high;
        const bool target_ok = (io_issue_ratio <= cfg_.io_issue_ratio_high) &&
                               (no_prefetch_ratio <= cfg_.no_prefetch_ratio_high);
        const bool is_conflict = should_increase && should_decrease;
        consecutive_target_phases_ = target_ok ? (consecutive_target_phases_ + 1) : 0;
        consecutive_conflict_phases_ = is_conflict ? (consecutive_conflict_phases_ + 1) : 0;

        int stable_iter_after = current_stable_iter_;
        std::string action = "hold";
        if (should_increase && !should_decrease &&
            current_stable_iter_ < cfg_.max_stable_iter) {
            stable_iter_after = ClampIter(current_stable_iter_ + cfg_.increase_step);
            action = "increase";
        } else if (should_decrease && !should_increase &&
                   current_stable_iter_ > cfg_.min_stable_iter) {
            stable_iter_after = ClampIter(current_stable_iter_ - cfg_.decrease_step);
            action = "decrease";
        } else if (is_conflict) {
            action = "hold_conflict";
        }
        current_stable_iter_ = stable_iter_after;

        if (!adjust_frozen_ &&
            cfg_.stop_eval_on_target &&
            consecutive_target_phases_ >= cfg_.stop_eval_target_consecutive_phases) {
            adjust_frozen_ = true;
            action = "freeze_target";
        }
        if (!adjust_frozen_ &&
            cfg_.stop_eval_on_conflict_phases > 0 &&
            consecutive_conflict_phases_ >= cfg_.stop_eval_on_conflict_phases) {
            adjust_frozen_ = true;
            action = "freeze_conflict";
        }

        PhaseRecord rec;
        rec.phase_id = ++phase_id_;
        rec.query_count = phase_query_count_;
        rec.no_prefetch_count = phase_no_prefetch_count_;
        rec.no_prefetch_ratio = no_prefetch_ratio;
        rec.issued_prefetch_labels = phase_issued_prefetch_labels_sum_;
        rec.topk_labels = phase_topk_labels_sum_;
        rec.io_issue_ratio = io_issue_ratio;
        rec.extra_io_ratio = extra_io_ratio;
        rec.avg_hnsw_us = phase_hnsw_sum_us_ / q;
        rec.stable_iter_before = stable_iter_before;
        rec.stable_iter_after = stable_iter_after;
        rec.action = action;
        phase_records_.push_back(rec);

        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[AdaptiveOverlap][phase {}] q={}, no_prefetch={}/{} ({:.3f}), "
                "issued_prefetch_labels={}, topk_labels={}, "
                "io_issue_ratio={:.3f}, extra_io_ratio={:.3f}, stable_iter {} -> {} ({})",
                rec.phase_id,
                rec.query_count,
                rec.no_prefetch_count,
                rec.query_count,
                rec.no_prefetch_ratio,
                rec.issued_prefetch_labels,
                rec.topk_labels,
                rec.io_issue_ratio,
                rec.extra_io_ratio,
                rec.stable_iter_before,
                rec.stable_iter_after,
                rec.action);
        }

        phase_query_count_ = 0;
        phase_no_prefetch_count_ = 0;
        phase_hnsw_sum_us_ = 0.0;
        phase_issued_prefetch_labels_sum_ = 0;
        phase_topk_labels_sum_ = 0;
    }

    AdaptiveStableIterConfig cfg_;
    int current_stable_iter_ = 30;

    std::mutex mu_;
    int phase_id_ = 0;
    int phase_query_count_ = 0;
    int phase_no_prefetch_count_ = 0;
    double phase_hnsw_sum_us_ = 0.0;
    std::size_t phase_issued_prefetch_labels_sum_ = 0;
    std::size_t phase_topk_labels_sum_ = 0;
    std::vector<PhaseRecord> phase_records_;
    bool adjust_frozen_ = false;
    std::size_t adjust_budget_queries_ = 0;
    std::size_t adjusted_query_count_ = 0;
    int consecutive_target_phases_ = 0;
    int consecutive_conflict_phases_ = 0;
    std::atomic<std::uint64_t> ann_us_total_{0};
    std::atomic<std::uint64_t> issued_prefetch_labels_total_{0};
    std::atomic<std::uint64_t> query_total_{0};
};

class AdaptiveEarlyStopOverlapIOStrategy : public ExperimentStrategy {
public:
    AdaptiveEarlyStopOverlapIOStrategy() = default;
    explicit AdaptiveEarlyStopOverlapIOStrategy(AdaptiveStableIterConfig cfg)
        : cfg_(std::move(cfg))
    {}

    std::string Name() const override { return "HnswOffsetEarlyStopOverlapAdaptive"; }
    std::string ResultFolderName() const override
    {
        return "OffsetEarlyStopOverlapAdaptive_results";
    }
    bool RequiresIterDist() const override { return true; }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    bool RequiresPostQueryIoWait() const override { return true; }

    std::string ModeSubdir(const HnswIoState& state) const override
    {
        std::string worker = state.use_shared_io_worker ? "shared" : "per_thread";
        return "adaptive_early_overlap_submit_only_" + worker;
    }

    void OnBeforeRun(const HnswIoState& state) override
    {
        if (state.pc) {
            cfg_ = state.pc->adaptive_cfg;
        }
        ann_us_total_.store(0, std::memory_order_relaxed);
        issued_prefetch_labels_total_.store(0, std::memory_order_relaxed);
        query_total_.store(0, std::memory_order_relaxed);
        early_stop_total_.store(0, std::memory_order_relaxed);
        overlap_trigger_total_.store(0, std::memory_order_relaxed);

        phase_records_.clear();
        phase_query_count_ = 0;
        phase_no_prefetch_count_ = 0;
        phase_early_stop_count_ = 0;
        phase_overlap_trigger_count_ = 0;
        phase_hnsw_sum_us_ = 0.0;
        phase_issued_prefetch_labels_sum_ = 0;
        phase_topk_labels_sum_ = 0;
        phase_id_ = 0;
        adjust_frozen_ = false;
        adjusted_query_count_ = 0;
        consecutive_target_phases_ = 0;
        consecutive_conflict_phases_ = 0;

        if (cfg_.adjust_query_percent > 0.0 && cfg_.adjust_query_percent < 100.0) {
            const double qn = static_cast<double>(std::max(state.query_cnt, 0));
            adjust_budget_queries_ = static_cast<std::size_t>(
                std::floor((qn * cfg_.adjust_query_percent) / 100.0));
        } else {
            adjust_budget_queries_ = 0;
        }

        const int fallback = state.pc ? state.pc->dataset.stable_iter : 30;
        const int initial = cfg_.initial_stable_iter > 0 ? cfg_.initial_stable_iter : fallback;
        current_stable_iter_ = ClampIter(initial);

        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[AdaptiveEarlyStopOverlap] init stable_iter={}, phase={}, min={}, max={}, "
                "ratio_high={}, io_issue_ratio_high={}, dec_step={}, inc_step={}, "
                "adjust_query_percent={}, stableiter_init={}, dataset_stable_iter={}, stop_on_target={}, "
                "stop_target_phases={}, stop_conflict_phases={}",
                current_stable_iter_,
                cfg_.adjust_every_queries,
                cfg_.min_stable_iter,
                cfg_.max_stable_iter,
                cfg_.no_prefetch_ratio_high,
                cfg_.io_issue_ratio_high,
                cfg_.decrease_step,
                cfg_.increase_step,
                cfg_.adjust_query_percent,
                cfg_.initial_stable_iter,
                fallback,
                cfg_.stop_eval_on_target,
                cfg_.stop_eval_target_consecutive_phases,
                cfg_.stop_eval_on_conflict_phases);
        }
    }

    void OnAfterRun(const HnswIoState& state,
                    const std::string& output_dir) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        FlushPhaseIfNeeded(state);
        std::ofstream out(output_dir + "/AdaptiveStableIterPhases.csv");
        if (out.is_open()) {
            out << "phase_id,queries,no_prefetch,early_stop,overlap_trigger,"
                   "no_prefetch_ratio,issued_prefetch_labels,topk_labels,io_issue_ratio,"
                   "extra_io_ratio,avg_hnsw_us,stable_iter_before,"
                   "stable_iter_after,action\n";
            for (const auto& r : phase_records_) {
                out << r.phase_id << ","
                    << r.query_count << ","
                    << r.no_prefetch_count << ","
                    << r.early_stop_count << ","
                    << r.overlap_trigger_count << ","
                    << r.no_prefetch_ratio << ","
                    << r.issued_prefetch_labels << ","
                    << r.topk_labels << ","
                    << r.io_issue_ratio << ","
                    << r.extra_io_ratio << ","
                    << r.avg_hnsw_us << ","
                    << r.stable_iter_before << ","
                    << r.stable_iter_after << ","
                    << r.action << "\n";
            }
        }

        const std::size_t qn = query_total_.load(std::memory_order_relaxed);
        const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
        const double avg_ann =
            static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
        std::cout << "[AdaptiveEarlyStopOverlapStrategy] avg_ann_us=" << avg_ann
                  << " issued_prefetch_labels=" << issued_prefetch_labels_total_.load(std::memory_order_relaxed)
                  << " early_stop=" << early_stop_total_.load(std::memory_order_relaxed)
                  << " overlap_triggered=" << overlap_trigger_total_.load(std::memory_order_relaxed)
                  << "\n";
    }

    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>&) override
    {
        auto& ioer = IoWorkerForThread(state, thread_id);
        const int stable_iter = current_stable_iter_;
        const bool adjust_this_query = ShouldAdjustThisQuery();
        annfs::hnsw::EarlyStopOverlapSignals signals;

        auto ann_start = Clock::now();
        auto pq = annfs::hnsw::SearchKnnEarlyStopOverlapMapped<float>(
            *state.hnsw_index,
            state.QueryVector(row),
            static_cast<std::size_t>(std::max(cur_k, 0)),
            state.pc->dataset.stable_hops,
            stable_iter,
            &ioer,
            state.pc->dataset.least_else_size,
            static_cast<int>(row),
            &signals,
            state.pc && !state.pc->skip_recall);
        auto ann_end = Clock::now();
        const double hnsw_us = ElapsedUs(ann_start, ann_end);
        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[AdaptiveEarlyStopOverlap][query {}] stable_iter={}, early_stop={}, "
                "overlap_triggered={}, issued_prefetch_labels={}, topk_labels={}, pending={}, adjust={}",
                row,
                stable_iter,
                signals.early_stop_triggered,
                signals.overlap_prefetch_triggered,
                signals.issued_prefetch_labels,
                std::max<std::size_t>(signals.topk_labels, 1),
                signals.pending_prefetch_ios,
                adjust_this_query);
        }

        if (adjust_this_query) {
            ObserveAndAdjust(state, stable_iter, signals, hnsw_us);
            adjusted_query_count_++;
        }

        auto result = BuildResult(std::move(pq), hnsw_us);
        result.io_time_us = 0.0;
        result.overlap = std::make_unique<OverlapQueryResult>();
        result.overlap->issued_label_set = std::move(signals.issued_label_set);

        ann_us_total_.fetch_add(
            static_cast<std::uint64_t>(std::max(0.0, result.hnsw_time_us)),
            std::memory_order_relaxed);
        issued_prefetch_labels_total_.fetch_add(
            static_cast<std::uint64_t>(signals.issued_prefetch_labels),
            std::memory_order_relaxed);
        query_total_.fetch_add(1, std::memory_order_relaxed);
        if (signals.early_stop_triggered) {
            early_stop_total_.fetch_add(1, std::memory_order_relaxed);
        }
        if (signals.overlap_prefetch_triggered) {
            overlap_trigger_total_.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

private:
    using Clock = std::chrono::high_resolution_clock;

    struct PhaseRecord {
        int phase_id = 0;
        int query_count = 0;
        int no_prefetch_count = 0;
        int early_stop_count = 0;
        int overlap_trigger_count = 0;
        double no_prefetch_ratio = 0.0;
        std::size_t issued_prefetch_labels = 0;
        std::size_t topk_labels = 0;
        double io_issue_ratio = 0.0;
        double extra_io_ratio = 0.0;
        double avg_hnsw_us = 0.0;
        int stable_iter_before = 0;
        int stable_iter_after = 0;
        std::string action;
    };

    static double ElapsedUs(const Clock::time_point& start,
                            const Clock::time_point& end)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    static ExperimentQueryResult BuildResult(
        std::priority_queue<std::pair<float, hnswlib::labeltype>> pq,
        double elapsed_us)
    {
        ExperimentQueryResult result;
        result.hnsw_time_us = elapsed_us;
        while (!pq.empty()) {
            result.result_ids.push_back(static_cast<unsigned long long>(pq.top().second));
            pq.pop();
        }
        return result;
    }

    int ClampIter(int v) const
    {
        return std::max(cfg_.min_stable_iter, std::min(cfg_.max_stable_iter, v));
    }

    bool ShouldAdjustThisQuery()
    {
        if (adjust_frozen_) {
            return false;
        }
        if (adjust_budget_queries_ > 0 &&
            adjusted_query_count_ >= adjust_budget_queries_) {
            adjust_frozen_ = true;
            return false;
        }
        return true;
    }

    void ObserveAndAdjust(const HnswIoState& state,
                          int stable_iter_used,
                          const annfs::hnsw::EarlyStopOverlapSignals& signals,
                          double hnsw_us)
    {
        std::lock_guard<std::mutex> lock(mu_);
        phase_query_count_ += 1;
        phase_hnsw_sum_us_ += hnsw_us;
        phase_issued_prefetch_labels_sum_ += signals.issued_prefetch_labels;
        phase_topk_labels_sum_ += std::max<std::size_t>(1, signals.topk_labels);
        if (!signals.overlap_prefetch_triggered) {
            phase_no_prefetch_count_ += 1;
        }
        if (signals.early_stop_triggered) {
            phase_early_stop_count_ += 1;
        }
        if (signals.overlap_prefetch_triggered) {
            phase_overlap_trigger_count_ += 1;
        }

        if (phase_query_count_ < cfg_.adjust_every_queries) {
            return;
        }
        FinalizeCurrentPhase(state, stable_iter_used);
    }

    void FlushPhaseIfNeeded(const HnswIoState& state)
    {
        if (phase_query_count_ <= 0) {
            return;
        }
        FinalizeCurrentPhase(state, current_stable_iter_);
    }

    void FinalizeCurrentPhase(const HnswIoState& state, int stable_iter_before)
    {
        const double q = static_cast<double>(phase_query_count_);
        const double no_prefetch_ratio = static_cast<double>(phase_no_prefetch_count_) / q;
        const double issued_prefetch_labels =
            static_cast<double>(phase_issued_prefetch_labels_sum_);
        const double topk_labels = static_cast<double>(phase_topk_labels_sum_);
        const double io_issue_ratio =
            topk_labels > 0.0 ? (issued_prefetch_labels / topk_labels) : 0.0;
        const double extra_io_ratio =
            topk_labels > 0.0
                ? std::max(0.0, (issued_prefetch_labels - topk_labels) / topk_labels)
                : 0.0;

        const bool should_decrease = no_prefetch_ratio > cfg_.no_prefetch_ratio_high;
        const bool should_increase = io_issue_ratio > cfg_.io_issue_ratio_high;
        const bool target_ok = (io_issue_ratio <= cfg_.io_issue_ratio_high) &&
                               (no_prefetch_ratio <= cfg_.no_prefetch_ratio_high);
        const bool is_conflict = should_increase && should_decrease;
        consecutive_target_phases_ = target_ok ? (consecutive_target_phases_ + 1) : 0;
        consecutive_conflict_phases_ = is_conflict ? (consecutive_conflict_phases_ + 1) : 0;

        int stable_iter_after = current_stable_iter_;
        std::string action = "hold";
        if (should_increase && !should_decrease &&
            current_stable_iter_ < cfg_.max_stable_iter) {
            stable_iter_after = ClampIter(current_stable_iter_ + cfg_.increase_step);
            action = "increase";
        } else if (should_decrease && !should_increase &&
                   current_stable_iter_ > cfg_.min_stable_iter) {
            stable_iter_after = ClampIter(current_stable_iter_ - cfg_.decrease_step);
            action = "decrease";
        } else if (is_conflict) {
            action = "hold_conflict";
        }
        current_stable_iter_ = stable_iter_after;

        if (!adjust_frozen_ && cfg_.stop_eval_on_target &&
            consecutive_target_phases_ >= cfg_.stop_eval_target_consecutive_phases) {
            adjust_frozen_ = true;
            action = "freeze_target";
        }
        if (!adjust_frozen_ && cfg_.stop_eval_on_conflict_phases > 0 &&
            consecutive_conflict_phases_ >= cfg_.stop_eval_on_conflict_phases) {
            adjust_frozen_ = true;
            action = "freeze_conflict";
        }

        PhaseRecord rec;
        rec.phase_id = ++phase_id_;
        rec.query_count = phase_query_count_;
        rec.no_prefetch_count = phase_no_prefetch_count_;
        rec.early_stop_count = phase_early_stop_count_;
        rec.overlap_trigger_count = phase_overlap_trigger_count_;
        rec.no_prefetch_ratio = no_prefetch_ratio;
        rec.issued_prefetch_labels = phase_issued_prefetch_labels_sum_;
        rec.topk_labels = phase_topk_labels_sum_;
        rec.io_issue_ratio = io_issue_ratio;
        rec.extra_io_ratio = extra_io_ratio;
        rec.avg_hnsw_us = phase_hnsw_sum_us_ / q;
        rec.stable_iter_before = stable_iter_before;
        rec.stable_iter_after = stable_iter_after;
        rec.action = action;
        phase_records_.push_back(rec);

        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[AdaptiveEarlyStopOverlap][phase {}] q={}, no_prefetch={}/{} ({:.3f}), "
                "early_stop={}, overlap_trigger={}, issued_prefetch_labels={}, topk_labels={}, "
                "io_issue_ratio={:.3f}, extra_io_ratio={:.3f}, stable_iter {} -> {} ({})",
                rec.phase_id,
                rec.query_count,
                rec.no_prefetch_count,
                rec.query_count,
                rec.no_prefetch_ratio,
                rec.early_stop_count,
                rec.overlap_trigger_count,
                rec.issued_prefetch_labels,
                rec.topk_labels,
                rec.io_issue_ratio,
                rec.extra_io_ratio,
                rec.stable_iter_before,
                rec.stable_iter_after,
                rec.action);
        }

        phase_query_count_ = 0;
        phase_no_prefetch_count_ = 0;
        phase_early_stop_count_ = 0;
        phase_overlap_trigger_count_ = 0;
        phase_hnsw_sum_us_ = 0.0;
        phase_issued_prefetch_labels_sum_ = 0;
        phase_topk_labels_sum_ = 0;
    }

    AdaptiveStableIterConfig cfg_;
    int current_stable_iter_ = 30;

    std::mutex mu_;
    int phase_id_ = 0;
    int phase_query_count_ = 0;
    int phase_no_prefetch_count_ = 0;
    int phase_early_stop_count_ = 0;
    int phase_overlap_trigger_count_ = 0;
    double phase_hnsw_sum_us_ = 0.0;
    std::size_t phase_issued_prefetch_labels_sum_ = 0;
    std::size_t phase_topk_labels_sum_ = 0;
    std::vector<PhaseRecord> phase_records_;
    bool adjust_frozen_ = false;
    std::size_t adjust_budget_queries_ = 0;
    std::size_t adjusted_query_count_ = 0;
    int consecutive_target_phases_ = 0;
    int consecutive_conflict_phases_ = 0;
    std::atomic<std::uint64_t> ann_us_total_{0};
    std::atomic<std::uint64_t> issued_prefetch_labels_total_{0};
    std::atomic<std::uint64_t> query_total_{0};
    std::atomic<std::uint64_t> early_stop_total_{0};
    std::atomic<std::uint64_t> overlap_trigger_total_{0};
};

} // namespace annfs::tests
