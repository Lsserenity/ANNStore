#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tests/HnswAdaptiveOverlap.hpp"
#include "tests/HnswExperimentRunner.hpp"
#include "tests/HnswExperimentStrategies.hpp"

namespace annfs::tests {

namespace {

using Clock = std::chrono::high_resolution_clock;

double ElapsedUs(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

} // namespace

// 监测版策略：
// 1) 复用 mapped overlap adaptive 的搜索链路；
// 2) 额外记录 query 级和 phase 级动态指标；
// 3) 输出单独 CSV，避免干扰现有性能/IO基线结果文件。
class MappedAdaptiveOverlapMonitorStrategy : public MappedOverlapIOStrategy {
public:
    MappedAdaptiveOverlapMonitorStrategy() = default;
    explicit MappedAdaptiveOverlapMonitorStrategy(AdaptiveStableIterConfig cfg)
        : cfg_(std::move(cfg))
    {}

    std::string Name() const override { return "HnswMappedOffsetOverlapAdaptiveMonitor"; }
    std::string ResultFolderName() const override
    {
        return "MappedOffsetOverlapAdaptiveMonitor_results";
    }

    bool RequiresPostQueryIoWait() const override
    {
        return true;
    }

    std::string ModeSubdir(const HnswIoState& state) const override
    {
        std::string worker = state.use_shared_io_worker ? "shared" : "per_thread";
        return "mapped_adaptive_overlap_monitor_submit_only_" + worker;
    }

    void OnBeforeRun(const HnswIoState& state) override
    {
        if (state.pc) {
            cfg_ = state.pc->adaptive_cfg;
        }
        MappedOverlapIOStrategy::OnBeforeRun(state);
        int fallback = state.pc ? state.pc->dataset.stable_iter : 10;
        int initial = cfg_.initial_stable_iter > 0 ? cfg_.initial_stable_iter : fallback;
        current_stable_iter_ = ClampIter(initial);

        std::lock_guard<std::mutex> lock(mu_);
        phase_records_.clear();
        query_records_.clear();
        phase_id_ = 0;
        phase_query_count_ = 0;
        phase_issued_prefetch_labels_sum_ = 0.0;
        phase_topk_labels_sum_ = 0.0;
        phase_missing_sum_ = 0.0;
        phase_pending_prefetch_sum_ = 0.0;
        phase_no_prefetch_count_ = 0;
        phase_hnsw_sum_us_ = 0.0;
        adjust_frozen_ = false;
        adjusted_query_count_ = 0;
        consecutive_target_phases_ = 0;
        consecutive_conflict_phases_ = 0;
        stop_on_target_ = cfg_.stop_eval_on_target;
        if (cfg_.adjust_query_percent > 0.0) {
            const double qn = static_cast<double>(std::max(state.query_cnt, 0));
            adjust_budget_queries_ = static_cast<std::size_t>(
                std::floor((qn * cfg_.adjust_query_percent) / 100.0));
        } else {
            adjust_budget_queries_ = 0; // unlimited
        }

        if (state.cfg && state.cfg->logger) {
            state.cfg->logger->info(
                "[MappedAdaptiveMonitor] init stable_iter={}, phase={}, min={}, max={}, "
                "io_issue_ratio_high={}, no_prefetch_ratio_high={}, dec_step={}, inc_step={}, "
                "adjust_query_percent={}, stop_on_target={}, stop_target_phases={}",
                current_stable_iter_,
                cfg_.adjust_every_queries,
                cfg_.min_stable_iter,
                cfg_.max_stable_iter,
                cfg_.io_issue_ratio_high,
                cfg_.no_prefetch_ratio_high,
                cfg_.decrease_step,
                cfg_.increase_step,
                cfg_.adjust_query_percent,
                stop_on_target_,
                cfg_.stop_eval_target_consecutive_phases);
        }
    }

    void OnAfterRun(const HnswIoState& state, const std::string& output_dir) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        FlushPhaseIfNeeded(state);

        {
            std::ofstream qout(output_dir + "/AdaptiveMappedQueryTrace.csv");
            qout << "query_id,stable_iter_used,topk,issued_prefetch_labels,mapped_missing_labels,pending_prefetch_ios,"
                    "prefetch_triggered,no_prefetch,io_issue_ratio,missing_ratio,hnsw_us,adjusted\n";
            qout << std::fixed << std::setprecision(6);
            for (const auto& r : query_records_) {
                qout << r.query_id << ","
                     << r.stable_iter_used << ","
                     << r.topk << ","
                     << r.issued_prefetch_labels << ","
                     << r.mapped_missing_labels << ","
                     << r.pending_prefetch_ios << ","
                     << (r.prefetch_triggered ? 1 : 0) << ","
                     << (r.no_prefetch ? 1 : 0) << ","
                     << r.io_issue_ratio << ","
                     << r.missing_ratio << ","
                     << r.hnsw_us << ","
                     << (r.adjusted ? 1 : 0) << "\n";
            }
        }

        {
            std::ofstream pout(output_dir + "/AdaptiveMappedPhases.csv");
            pout << "phase_id,queries,no_prefetch_count,no_prefetch_ratio,issued_prefetch_labels,topk_labels,mapped_missing_labels,pending_prefetch_ios_sum,"
                    "io_issue_ratio,missing_ratio,avg_pending_prefetch_ios,avg_hnsw_us,stable_iter_before,"
                    "stable_iter_after,action\n";
            pout << std::fixed << std::setprecision(6);
            for (const auto& r : phase_records_) {
                pout << r.phase_id << ","
                     << r.query_count << ","
                     << r.no_prefetch_count << ","
                     << r.no_prefetch_ratio << ","
                     << r.issued_prefetch_labels << ","
                     << r.topk_labels << ","
                     << r.mapped_missing_labels << ","
                     << r.pending_prefetch_ios_sum << ","
                     << r.io_issue_ratio << ","
                     << r.missing_ratio << ","
                     << r.avg_pending_prefetch_ios << ","
                     << r.avg_hnsw_us << ","
                     << r.stable_iter_before << ","
                     << r.stable_iter_after << ","
                     << r.action << "\n";
            }
        }

        std::size_t total_issued_prefetch_labels = 0;
        std::size_t total_mapped_missing_labels = 0;
        double total_hnsw = 0.0;
        for (const auto& r : query_records_) {
            total_issued_prefetch_labels += r.issued_prefetch_labels;
            total_mapped_missing_labels += r.mapped_missing_labels;
            total_hnsw += r.hnsw_us;
        }
        const double avg_hnsw = query_records_.empty()
            ? 0.0
            : (total_hnsw / static_cast<double>(query_records_.size()));

        std::cout << "[MappedAdaptiveMonitor] queries=" << query_records_.size()
                  << " phases=" << phase_records_.size()
                  << " avg_hnsw_us=" << avg_hnsw
                  << " total_issued_prefetch_labels=" << total_issued_prefetch_labels
                  << " mapped_missing_labels=" << total_mapped_missing_labels
                  << " final_stable_iter=" << current_stable_iter_
                  << "\n";
        std::cout << "Adaptive mapped monitor CSV persisted to " << output_dir << "\n";
    }

    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>&) override
    {
        const int stable_iter = CurrentStableIter();
        auto& io_worker = IoWorkerForThread(state, thread_id);

        int pending_prefetch_ios = 0;
        std::size_t issued_prefetch_labels_raw = 0;
        std::size_t mapped_missing = 0;
        std::size_t topk_labels_raw = 0;
        bool prefetch_triggered = false;

        auto ann_start = Clock::now();
        std::unordered_set<unsigned long long> sent_prefetch_labels = state.hnsw_index->searchKnnPrefetchMappedBaseline(
            state.QueryVector(row),
            cur_k,
            stable_iter,
            &io_worker,
            state.pc->dataset.least_else_size,
            nullptr,
            &pending_prefetch_ios,
            static_cast<int>(row),
            &issued_prefetch_labels_raw,
            &mapped_missing,
            &prefetch_triggered,
            &topk_labels_raw,
            state.pc && !state.pc->skip_recall);
        auto ann_end = Clock::now();
        const double hnsw_us = ElapsedUs(ann_start, ann_end);

        const std::size_t effective_topk_labels =
            std::max<std::size_t>(topk_labels_raw, 1);
        const double io_issue_ratio =
            static_cast<double>(issued_prefetch_labels_raw) /
            static_cast<double>(effective_topk_labels);
        const double missing_ratio = (issued_prefetch_labels_raw > 0)
            ? static_cast<double>(mapped_missing) / static_cast<double>(issued_prefetch_labels_raw)
            : 0.0;

        bool adjust_this_query = !adjust_frozen_;
        if (adjust_this_query && adjust_budget_queries_ > 0 &&
            adjusted_query_count_ >= adjust_budget_queries_) {
            adjust_this_query = false;
            adjust_frozen_ = true;
        }
        ObserveAndMaybeAdjust(
            state,
            static_cast<int>(row),
            stable_iter,
            effective_topk_labels,
            issued_prefetch_labels_raw,
            mapped_missing,
            pending_prefetch_ios,
            prefetch_triggered,
            hnsw_us,
            adjust_this_query);
        if (adjust_this_query) {
            adjusted_query_count_++;
        }

        ExperimentQueryResult res;
        res.hnsw_time_us = hnsw_us;
        res.io_time_us = -1.0;
        res.overlap = std::make_unique<OverlapQueryResult>();
        res.overlap->issued_label_set = std::move(sent_prefetch_labels);
        (void)io_issue_ratio;
        (void)missing_ratio;
        return res;
    }

private:
    struct QueryRecord {
        int query_id = -1;
        int stable_iter_used = 0;
        int topk = 0;
        std::size_t issued_prefetch_labels = 0;
        std::size_t mapped_missing_labels = 0;
        int pending_prefetch_ios = 0;
        bool prefetch_triggered = false;
        bool no_prefetch = false;
        double io_issue_ratio = 0.0;
        double missing_ratio = 0.0;
        double hnsw_us = 0.0;
        bool adjusted = false;
    };

    struct PhaseRecord {
        int phase_id = 0;
        int query_count = 0;
        int no_prefetch_count = 0;
        double no_prefetch_ratio = 0.0;
        double issued_prefetch_labels = 0.0;
        double topk_labels = 0.0;
        double mapped_missing_labels = 0.0;
        double pending_prefetch_ios_sum = 0.0;
        double io_issue_ratio = 0.0;
        double missing_ratio = 0.0;
        double avg_pending_prefetch_ios = 0.0;
        double avg_hnsw_us = 0.0;
        int stable_iter_before = 0;
        int stable_iter_after = 0;
        std::string action;
    };

    int ClampIter(int v) const
    {
        return std::max(cfg_.min_stable_iter, std::min(cfg_.max_stable_iter, v));
    }

    int CurrentStableIter()
    {
        std::lock_guard<std::mutex> lock(mu_);
        return current_stable_iter_;
    }

    void ObserveAndMaybeAdjust(
        const HnswIoState& state,
        int query_id,
        int stable_iter_used,
        double topk_labels,
        double issued_prefetch_labels,
        double mapped_missing_labels,
        int pending_prefetch_ios,
        bool prefetch_triggered,
        double hnsw_us,
        bool adjusted)
    {
        std::lock_guard<std::mutex> lock(mu_);

        QueryRecord qrec;
        qrec.query_id = query_id;
        qrec.stable_iter_used = stable_iter_used;
        qrec.topk = static_cast<int>(topk_labels);
        qrec.issued_prefetch_labels = issued_prefetch_labels;
        qrec.mapped_missing_labels = mapped_missing_labels;
        qrec.pending_prefetch_ios = pending_prefetch_ios;
        qrec.prefetch_triggered = prefetch_triggered;
        qrec.no_prefetch = !prefetch_triggered;
        qrec.io_issue_ratio =
            static_cast<double>(issued_prefetch_labels) /
            static_cast<double>(std::max<std::size_t>(topk_labels, 1));
        qrec.missing_ratio = (issued_prefetch_labels > 0)
            ? static_cast<double>(mapped_missing_labels) / static_cast<double>(issued_prefetch_labels)
            : 0.0;
        qrec.hnsw_us = hnsw_us;
        qrec.adjusted = adjusted;
        query_records_.push_back(qrec);

        // 未参与调整的 query 只记录，不进入 phase 统计。
        if (!adjusted) {
            return;
        }

        phase_query_count_ += 1.0;
        phase_issued_prefetch_labels_sum_ += issued_prefetch_labels;
        phase_topk_labels_sum_ += topk_labels;
        phase_missing_sum_ += mapped_missing_labels;
        phase_pending_prefetch_sum_ += static_cast<double>(std::max(0, pending_prefetch_ios));
        if (!prefetch_triggered) {
            phase_no_prefetch_count_ += 1.0;
        }
        phase_hnsw_sum_us_ += hnsw_us;

        if (phase_query_count_ < static_cast<double>(cfg_.adjust_every_queries)) {
            return;
        }
        FinalizePhase(state, stable_iter_used);
    }

    void FlushPhaseIfNeeded(const HnswIoState& state)
    {
        if (phase_query_count_ == 0) {
            return;
        }
        FinalizePhase(state, current_stable_iter_);
    }

    void FinalizePhase(const HnswIoState& state, int stable_iter_before)
    {
        const double qn = phase_query_count_;
        const double io_issue_ratio = (phase_topk_labels_sum_ == 0)
            ? 0.0
            : (phase_issued_prefetch_labels_sum_ / phase_topk_labels_sum_);
        const double no_prefetch_ratio = phase_no_prefetch_count_ / qn;
        const double missing_ratio = (phase_issued_prefetch_labels_sum_ == 0)
            ? 0.0
            : (phase_missing_sum_ / phase_issued_prefetch_labels_sum_);
        const double avg_pending = phase_pending_prefetch_sum_ / qn;
        const double avg_hnsw = phase_hnsw_sum_us_ / qn;
        const bool target_ok = (io_issue_ratio <= cfg_.io_issue_ratio_high) &&
                               (no_prefetch_ratio <= cfg_.no_prefetch_ratio_high);

        const bool should_decrease = no_prefetch_ratio > cfg_.no_prefetch_ratio_high;
        const bool should_increase = io_issue_ratio > cfg_.io_issue_ratio_high;
        const bool is_conflict = should_increase && should_decrease;
        if (target_ok) {
            consecutive_target_phases_++;
        } else {
            consecutive_target_phases_ = 0;
        }
        if (is_conflict) {
            consecutive_conflict_phases_++;
        } else {
            consecutive_conflict_phases_ = 0;
        }

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

        if (!adjust_frozen_ && stop_on_target_ &&
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
        rec.query_count = static_cast<int>(phase_query_count_);
        rec.no_prefetch_count = static_cast<int>(phase_no_prefetch_count_);
        rec.no_prefetch_ratio = no_prefetch_ratio;
        rec.issued_prefetch_labels = phase_issued_prefetch_labels_sum_;
        rec.topk_labels = phase_topk_labels_sum_;
        rec.mapped_missing_labels = phase_missing_sum_;
        rec.pending_prefetch_ios_sum = phase_pending_prefetch_sum_;
        rec.io_issue_ratio = io_issue_ratio;
        rec.missing_ratio = missing_ratio;
        rec.avg_pending_prefetch_ios = avg_pending;
        rec.avg_hnsw_us = avg_hnsw;
        rec.stable_iter_before = stable_iter_before;
        rec.stable_iter_after = stable_iter_after;
        rec.action = action;
        phase_records_.push_back(rec);

        if (state.cfg && state.cfg->logger &&
            state.cfg->logger->should_log(spdlog::level::debug)) {
            state.cfg->logger->debug(
                "[MappedAdaptiveMonitor][phase {}] q={}, no_prefetch={}/{} ({:.3f}), "
                "issued_prefetch_labels={}, topk_labels={}, "
                "mapped_missing_labels={}, io_issue_ratio={:.3f}, missing_ratio={:.3f}, "
                "avg_pending_prefetch={:.3f}, avg_hnsw_us={:.3f}, stable_iter {} -> {} ({})",
                rec.phase_id,
                rec.query_count,
                rec.no_prefetch_count,
                rec.query_count,
                rec.no_prefetch_ratio,
                rec.issued_prefetch_labels,
                rec.topk_labels,
                rec.mapped_missing_labels,
                rec.io_issue_ratio,
                rec.missing_ratio,
                rec.avg_pending_prefetch_ios,
                rec.avg_hnsw_us,
                rec.stable_iter_before,
                rec.stable_iter_after,
                rec.action);
        }

        phase_query_count_ = 0.0;
        phase_issued_prefetch_labels_sum_ = 0.0;
        phase_topk_labels_sum_ = 0.0;
        phase_missing_sum_ = 0.0;
        phase_pending_prefetch_sum_ = 0.0;
        phase_no_prefetch_count_ = 0.0;
        phase_hnsw_sum_us_ = 0.0;
    }

    AdaptiveStableIterConfig cfg_;
    std::mutex mu_;
    int current_stable_iter_ = 10;

    int phase_id_ = 0;
    double phase_query_count_ = 0.0;
    double phase_issued_prefetch_labels_sum_ = 0.0;
    double phase_topk_labels_sum_ = 0.0;
    double phase_missing_sum_ = 0.0;
    double phase_pending_prefetch_sum_ = 0.0;
    double phase_no_prefetch_count_ = 0.0;
    double phase_hnsw_sum_us_ = 0.0;
    bool adjust_frozen_ = false;
    std::size_t adjust_budget_queries_ = 0; // 0 means unlimited
    std::size_t adjusted_query_count_ = 0;
    int consecutive_target_phases_ = 0;
    int consecutive_conflict_phases_ = 0;
    bool stop_on_target_ = false; // monitor 默认关闭自动停止

    std::vector<QueryRecord> query_records_;
    std::vector<PhaseRecord> phase_records_;
};

} // namespace annfs::tests

int main(int argc, char** argv)
{
    auto args_opt = annfs::tests::ParseCliArgs(argc, argv);
    if (!args_opt) {
        annfs::tests::PrintAdaptiveOverlapUsage(argv[0]);
        return 1;
    }
    annfs::tests::MappedAdaptiveOverlapMonitorStrategy strategy;
    return annfs::tests::RunHnswExperiment(*args_opt, strategy);
}
