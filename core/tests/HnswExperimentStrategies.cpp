#include "tests/HnswExperimentStrategies.hpp"
#include "tests/HnswExperimentBatch.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <cstdint>
#include <cmath>
#include <set>
#include <sys/types.h>

#include "json_utils.hpp"
#include "recall.hpp"
#include "global_config.hpp"
#include "early-stop-overlap-support.hpp"
#include "mapped_io_plan.hpp"
#include "mapped_label_offset_resolver.hpp"

namespace annfs::tests {

namespace {
using Clock = std::chrono::high_resolution_clock;

std::string BuildAsyncModeDir(const HnswIoState& state, bool wait_per_query)
{
    std::string worker = state.use_shared_io_worker ? "shared" : "per_thread";
    std::string mode = wait_per_query ? "per_query_wait" : "submit_only";
    return mode + "_" + worker;
}

std::vector<unsigned long long> ExtractIds(
    const std::priority_queue<std::pair<float, hnswlib::labeltype>>& hnsw_queue)
{
    std::vector<unsigned long long> ids;
    auto copy = hnsw_queue;
    ids.reserve(copy.size());
    while (!copy.empty()) {
        ids.push_back(static_cast<unsigned long long>(copy.top().second));
        copy.pop();
    }
    return ids;
}

ExperimentQueryResult BuildResultFromQueue(
    std::priority_queue<std::pair<float, hnswlib::labeltype>>& hnsw_queue,
    double elapsed_us)
{
    ExperimentQueryResult result;
    result.hnsw_time_us = elapsed_us;
    while (!hnsw_queue.empty()) {
        result.result_ids.push_back(
            static_cast<unsigned long long>(hnsw_queue.top().second));
        hnsw_queue.pop();
    }
    return result;
}

ExperimentQueryResult BuildResultFromQueue(
    std::priority_queue<std::pair<float, hnswlib::labeltype>>& hnsw_queue,
    double elapsed_us,
    int iter_count,
    int dist_count)
{
    ExperimentQueryResult result =
        BuildResultFromQueue(hnsw_queue, elapsed_us);
    result.iter_count = iter_count;
    result.dist_count = dist_count;
    return result;
}

struct IterProfile {
    std::vector<int> iter_counts;
    std::vector<bool> found_flags;
};

IterProfile LoadIterCounts(const std::string& json_path,
                           size_t expected_queries)
{
    IterProfile profile;
    profile.iter_counts.assign(expected_queries, 0);
    profile.found_flags.assign(expected_queries, true);
    if (json_path.empty()) {
        return profile;
    }
    try {
        auto raw = annfs::common::read_json(json_path);
        const nlohmann::json* entries = nullptr;
        if (raw.is_array()) {
            entries = &raw;
        } else if (raw.contains("entries")) {
            entries = &raw["entries"];
        }
        if (!entries) {
            return profile;
        }
        for (size_t i = 0; i < expected_queries && i < entries->size(); ++i) {
            const auto& entry = (*entries)[i];
            if (entry.contains("iter_count")) {
                profile.iter_counts[i] = entry["iter_count"].get<int>();
            } else if (entry.contains("iterations")) {
                profile.iter_counts[i] = entry["iterations"].get<int>();
            }
            if (entry.contains("found_gt")) {
                profile.found_flags[i] = entry["found_gt"].get<double>() > 0.5;
            }
        }
    } catch (const std::exception& e) {
        auto logger = GlobalConfig::instance().logger;
        if (logger) {
            logger->error("Failed to load iter json {}: {}", json_path, e.what());
        }
    }
    return profile;
}

int issue_mapped_io_sync(const HnswIoState& state,
                         size_t thread_id,
                         int query_id,
                         const std::vector<unsigned long long>& ids,
                         std::size_t& missing,
                         std::size_t& issued_io)
{
    auto& worker = IoWorkerForThread(state, thread_id);
    if (!state.pc) {
        missing = 0;
        issued_io = 0;
        return 0;
    }
    return worker.mapped_read_sync(ids, *state.pc, query_id, &missing, &issued_io);
}

int issue_mapped_io_async(const HnswIoState& state,
                          size_t thread_id,
                          int query_id,
                          const std::vector<unsigned long long>& ids,
                          bool wait_per_query,
                          std::size_t& missing,
                          std::size_t& issued_io)
{
    auto& worker = IoWorkerForThread(state, thread_id);
    if (!state.pc) {
        missing = 0;
        issued_io = 0;
        return 0;
    }
    return worker.mapped_prefetch(
        ids, *state.pc, query_id, wait_per_query, &missing, &issued_io);
}

} // namespace

ExperimentQueryResult OffsetIOStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnn(
        state.QueryVector(row),
        cur_k);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();
    auto res = BuildResultFromQueue(pq, elapsed);
    auto& ioer = IoWorkerForThread(state, thread_id);
    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto io_start = Clock::now();
    ioer.mapped_read_sync(
        res.result_ids,
        *state.pc,
        static_cast<int>(row),
        &missing,
        &issued_io);
    (void)missing;
    (void)issued_io;
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<
        std::chrono::microseconds>(io_end - io_start).count();
    ann_us_total_.fetch_add(
        static_cast<std::uint64_t>(res.hnsw_time_us),
        std::memory_order_relaxed);
    total_offsets_.fetch_add(
        res.result_ids.size(),
        std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    return res;
}

void OffsetIOStrategy::OnBeforeRun(const HnswIoState& state)
{
    ann_us_total_.store(0, std::memory_order_relaxed);
    total_offsets_.store(0, std::memory_order_relaxed);
    query_total_.store(0, std::memory_order_relaxed);
    std::cout << "[BaseStrategy] dataset=" << state.args.dataset_name
              << " threads=" << state.num_threads
              << " search_ef=" << state.pc->dataset.search_ef
              << " topk=" << state.pc->topk
              << " io_depth=" << GlobalConfig::instance().RING_DEPTH
              << " io_worker_mode=" << (state.use_shared_io_worker ? "shared" : "per_thread")
              << "\n";
}

void OffsetIOStrategy::OnAfterRun(const HnswIoState&,
                                  const std::string&)
{
    const std::size_t qn = query_total_.load(std::memory_order_relaxed);
    const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
    const double avg_ann = static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
    std::cout << "[BaseStrategy] avg_ann_us=" << avg_ann
              << " total_offsets=" << total_offsets_.load(std::memory_order_relaxed)
              << " missing_labels=0\n";
}

ExperimentQueryResult OffsetIOAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnn(
        state.QueryVector(row),
        cur_k);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();

    auto& ioer = IoWorkerForThread(state, thread_id);
    auto io_start = Clock::now();
    std::size_t missing = 0;
    std::size_t issued_io = 0;
    int pending = ioer.mapped_prefetch(
        ExtractIds(pq),
        *state.pc,
        static_cast<int>(row),
        wait_per_query_,
        &missing,
        &issued_io);
    if (wait_per_query_) {
        ioer.wait_all_completions(pending);
    }
    auto io_end = Clock::now();
    double io_elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(io_end - io_start).count();

    auto result = BuildResultFromQueue(pq, elapsed);
    result.io_time_us = io_elapsed;
    return result;
}

void OffsetIOAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    // 是否强制每个查询提交到异步队列后等待全部io处理结束
    wait_per_query_ = state.pc
        ? state.pc->async_submit_and_wait
        : true;
}

bool OffsetIOAsyncStrategy::RequiresPostQueryIoWait() const
{
    return !wait_per_query_;
}

std::string OffsetIOAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    return BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult EarlyStopStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    (void)thread_id;
    int iter_count = 0;
    int dist_count = 0;
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnnEarly(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_hops);
        // dist_count,
        // iter_count);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();
    auto result = BuildResultFromQueue(pq, elapsed, iter_count, dist_count);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto io_start = Clock::now();
    issue_mapped_io_sync(
        state,
        thread_id,
        static_cast<int>(row),
        result.result_ids,
        missing,
        issued_io);
    auto io_end = Clock::now();
    result.io_time_us = std::chrono::duration_cast<
        std::chrono::microseconds>(io_end - io_start).count();

    ann_us_total_.fetch_add(
        static_cast<std::uint64_t>(result.hnsw_time_us),
        std::memory_order_relaxed);
    total_offsets_.fetch_add(issued_io, std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    return result;
}

void EarlyStopStrategy::OnBeforeRun(const HnswIoState&)
{
    ann_us_total_.store(0, std::memory_order_relaxed);
    total_offsets_.store(0, std::memory_order_relaxed);
    query_total_.store(0, std::memory_order_relaxed);
}

void EarlyStopStrategy::OnAfterRun(const HnswIoState&,
                                   const std::string&)
{
    const std::size_t qn = query_total_.load(std::memory_order_relaxed);
    const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
    const double avg_ann =
        static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
    std::cout << "[EarlyStopStrategy] avg_ann_us=" << avg_ann
              << " total_offsets=" << total_offsets_.load(std::memory_order_relaxed)
              << "\n";
}

ExperimentQueryResult EarlyStopAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    int iter_count = 0;
    int dist_count = 0;
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnnEarly(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_hops);
        // dist_count,
        // iter_count);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();

    auto& ioer = IoWorkerForThread(state, thread_id);
    auto io_start = Clock::now();
    std::size_t missing = 0;
    std::size_t issued_io = 0;
    int pending = ioer.mapped_prefetch(
        ExtractIds(pq),
        *state.pc,
        static_cast<int>(row),
        wait_per_query_,
        &missing,
        &issued_io);
    if (wait_per_query_) {
        ioer.wait_all_completions(pending);
    }
    auto io_end = Clock::now();
    double io_elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(io_end - io_start).count();

    auto result = BuildResultFromQueue(pq, elapsed, iter_count, dist_count);
    result.io_time_us = io_elapsed;
    ann_us_total_.fetch_add(
        static_cast<std::uint64_t>(result.hnsw_time_us),
        std::memory_order_relaxed);
    total_offsets_.fetch_add(
        issued_io,
        std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    return result;
}

void EarlyStopAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    ann_us_total_.store(0, std::memory_order_relaxed);
    total_offsets_.store(0, std::memory_order_relaxed);
    query_total_.store(0, std::memory_order_relaxed);
    wait_per_query_ = state.pc
        ? state.pc->async_submit_and_wait
        : true;
}

void EarlyStopAsyncStrategy::OnAfterRun(const HnswIoState&,
                                        const std::string&)
{
    const std::size_t qn = query_total_.load(std::memory_order_relaxed);
    const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
    const double avg_ann =
        static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
    std::cout << "[EarlyStopAsyncStrategy] avg_ann_us=" << avg_ann
              << " total_offsets=" << total_offsets_.load(std::memory_order_relaxed)
              << "\n";
}

bool EarlyStopAsyncStrategy::RequiresPostQueryIoWait() const
{
    return !wait_per_query_;
}

std::string EarlyStopAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    return BuildAsyncModeDir(state, wait_per_query_);
}

void EarlyStopOverlapAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    ann_us_total_.store(0, std::memory_order_relaxed);
    issued_prefetch_labels_total_.store(0, std::memory_order_relaxed);
    query_total_.store(0, std::memory_order_relaxed);
    early_first_total_.store(0, std::memory_order_relaxed);
    overlap_first_total_.store(0, std::memory_order_relaxed);
    overlap_trigger_total_.store(0, std::memory_order_relaxed);
    wait_per_query_ = state.pc ? state.pc->async_submit_and_wait : true;
}

void EarlyStopOverlapAsyncStrategy::OnAfterRun(const HnswIoState&,
                                               const std::string&)
{
    const std::size_t qn = query_total_.load(std::memory_order_relaxed);
    const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
    const double avg_ann =
        static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
    std::cout << "[EarlyStopOverlapAsyncStrategy] avg_ann_us=" << avg_ann
              << " issued_prefetch_labels="
              << issued_prefetch_labels_total_.load(std::memory_order_relaxed)
              << " overlap_triggered=" << overlap_trigger_total_.load(std::memory_order_relaxed)
              << " early_first=" << early_first_total_.load(std::memory_order_relaxed)
              << " overlap_first=" << overlap_first_total_.load(std::memory_order_relaxed)
              << "\n";
}

bool EarlyStopOverlapAsyncStrategy::RequiresPostQueryIoWait() const
{
    return !wait_per_query_;
}

std::string EarlyStopOverlapAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    return "early_overlap_" + BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult EarlyStopOverlapAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    auto& ioer = IoWorkerForThread(state, thread_id);
    annfs::hnsw::EarlyStopOverlapSignals signals;

    auto start = Clock::now();
    auto pq = annfs::hnsw::SearchKnnEarlyStopOverlapMapped<float>(
        *state.hnsw_index,
        state.QueryVector(row),
        static_cast<std::size_t>(std::max(cur_k, 0)),
        state.pc->dataset.stable_hops,
        state.pc->dataset.stable_iter,
        &ioer,
        state.pc->dataset.least_else_size,
        static_cast<int>(row),
        &signals,
        state.pc && !state.pc->skip_recall);
    if (wait_per_query_ && signals.pending_prefetch_ios > 0) {
        ioer.wait_all_completions(signals.pending_prefetch_ios);
    }
    auto end = Clock::now();
    const double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();

    auto result = BuildResultFromQueue(pq, elapsed);
    result.io_time_us = 0.0;
    result.overlap = std::make_unique<OverlapQueryResult>();
    result.overlap->issued_label_set = std::move(signals.issued_label_set);
    if (state.cfg && state.cfg->logger &&
        state.cfg->logger->should_log(spdlog::level::debug)) {
        state.cfg->logger->debug(
            "[EarlyStopOverlapAsync][query {}] stable_iter={}, early_stop={}, "
            "overlap_triggered={}, issued_prefetch_labels={}, topk_labels={}, pending={}",
            row,
            state.pc->dataset.stable_iter,
            signals.early_stop_triggered,
            signals.overlap_prefetch_triggered,
            signals.issued_prefetch_labels,
            std::max<std::size_t>(signals.topk_labels, 1),
            signals.pending_prefetch_ios);
    }

    ann_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::max(0.0, result.hnsw_time_us)),
        std::memory_order_relaxed);
    issued_prefetch_labels_total_.fetch_add(
        signals.issued_prefetch_labels,
        std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    if (signals.overlap_prefetch_triggered) {
        overlap_trigger_total_.fetch_add(1, std::memory_order_relaxed);
    }
    if (signals.early_stop_triggered && !signals.overlap_prefetch_triggered) {
        early_first_total_.fetch_add(1, std::memory_order_relaxed);
    } else if (signals.overlap_prefetch_triggered) {
        overlap_first_total_.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

std::string IdealTimeStrategy::CustomOutputDir(
    const HnswIoState&,
    const HnswIoCliArgs& args) const
{
    const auto& pc = ParamConfig::instance();
    return "./ideal_time_results/" + args.dataset_name + "/" +
           std::to_string(pc.num_threads) + "_" +
           std::to_string(pc.dataset.search_ef) + "_" +
           std::to_string(GlobalConfig::instance().RING_DEPTH) + "/" +
           std::to_string(pc.query_cnt) + "/" +
           std::to_string(args.repeat_id);
}

void IdealTimeStrategy::OnBeforeRun(const HnswIoState& state)
{
    if (iter_json_path_.empty()) {
        throw std::runtime_error(
            "IdealTimeStrategy requires iter_json_path (set via CLI or batch plan)");
    }
    if (iter_counts_.empty()) {
        auto profile = LoadIterCounts(iter_json_path_, state.query_cnt);
        iter_counts_ = std::move(profile.iter_counts);
        found_gt_flags_ = std::move(profile.found_flags);
    }
}

void IdealTimeStrategy::ApplyBatchRunSpec(
    const ExperimentBatchRunSpec& spec)
{
    if (spec.iter_json_path) {
        iter_json_path_ = *spec.iter_json_path;
        iter_counts_.clear();
        found_gt_flags_.clear();
    }
}

ExperimentQueryResult IdealTimeStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    (void)thread_id;
    int iter_limit = 0;
    bool has_profile = (row < iter_counts_.size() && iter_counts_[row] > 0);
    if (has_profile) {
        iter_limit = iter_counts_[row];
    }
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnnIdealTime(
        state.QueryVector(row),
        cur_k,
        iter_limit);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();
    return BuildResultFromQueue(pq, elapsed);
}

std::string IdealTimeAsyncStrategy::CustomOutputDir(
    const HnswIoState&,
    const HnswIoCliArgs& args) const
{
    const auto& pc = ParamConfig::instance();
    return "./ideal_time_results/" + args.dataset_name + "/" +
           std::to_string(pc.num_threads) + "_" +
           std::to_string(pc.dataset.search_ef) + "_" +
           std::to_string(GlobalConfig::instance().RING_DEPTH) + "/" +
           std::to_string(pc.query_cnt) + "/" +
           std::to_string(args.repeat_id);
}

void IdealTimeAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    if (iter_json_path_.empty()) {
        throw std::runtime_error(
            "IdealTimeAsyncStrategy requires iter_json_path (set via CLI or batch plan)");
    }
    if (iter_counts_.empty()) {
        auto profile = LoadIterCounts(iter_json_path_, state.query_cnt);
        iter_counts_ = std::move(profile.iter_counts);
        found_gt_flags_ = std::move(profile.found_flags);
    }
    wait_per_query_ = state.pc
        ? state.pc->async_submit_and_wait
        : true;
}

void IdealTimeAsyncStrategy::ApplyBatchRunSpec(
    const ExperimentBatchRunSpec& spec)
{
    if (spec.iter_json_path) {
        iter_json_path_ = *spec.iter_json_path;
        iter_counts_.clear();
        found_gt_flags_.clear();
    }
}

ExperimentQueryResult IdealTimeAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    int iter_limit = 0;
    bool has_profile = (row < iter_counts_.size() && iter_counts_[row] > 0);
    if (has_profile) {
        iter_limit = iter_counts_[row];
    }
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnnIdealTime(
        state.QueryVector(row),
        cur_k,
        iter_limit);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();

    auto& ioer = IoWorkerForThread(state, thread_id);
    auto io_start = Clock::now();
    std::size_t missing = 0;
    std::size_t issued_io = 0;
    int pending = ioer.mapped_prefetch(
        ExtractIds(pq),
        *state.pc,
        static_cast<int>(row),
        wait_per_query_,
        &missing,
        &issued_io);
    if (wait_per_query_) {
        ioer.wait_all_completions(pending);
    }
    auto io_end = Clock::now();
    double io_elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(io_end - io_start).count();

    auto result = BuildResultFromQueue(pq, elapsed);
    result.io_time_us = io_elapsed;
    return result;
}

bool IdealTimeAsyncStrategy::RequiresPostQueryIoWait() const
{
    return !wait_per_query_;
}

std::string IdealTimeAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    return BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult IdealSpaceStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>& gt_row)
{
    (void)thread_id;
    std::set<hnswlib::labeltype> gt_set;
    for (int i = 0; i < cur_k; ++i) {
        gt_set.insert(gt_row[i]);
    }

    int iter_count = 0;
    int dist_count = 0;
    bool found_all_gt = false;
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnnRecallCost(
        state.QueryVector(row),
        cur_k,
        gt_set,
        iter_count,
        dist_count,
        found_all_gt);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();
    ExperimentQueryResult res =
        BuildResultFromQueue(pq, elapsed, iter_count, dist_count);
    res.found_all_gt = found_all_gt;

    if (state.pc && state.pc->is_debug && state.cfg && state.cfg->logger) {
        double primary_recall =
            annfs::common::get_recall(res.result_ids, gt_row, cur_k);
        double iter_recall = primary_recall;
        if (res.iter_count > 0) {
            auto verify_pq = state.hnsw_index->searchKnnIdealTime(
                state.QueryVector(row),
                cur_k,
                res.iter_count);
            std::vector<unsigned long long> iter_ids;
            while (!verify_pq.empty()) {
                iter_ids.push_back(
                    static_cast<unsigned long long>(verify_pq.top().second));
                verify_pq.pop();
            }
            std::reverse(iter_ids.begin(), iter_ids.end());
            if (!iter_ids.empty()) {
                iter_recall = annfs::common::get_recall(
                    iter_ids,
                    gt_row,
                    std::min<int>(cur_k, iter_ids.size()));
            }
        }
        state.cfg->logger->debug(
            "[IdealSpace][row {}] iter={}, dist={}, found_gt={}, recall={}, "
            "recall@iter={}",
            row,
            res.iter_count,
            res.dist_count,
            res.found_all_gt,
            primary_recall,
            iter_recall);
        if (std::fabs(primary_recall - iter_recall) > 1e-3) {
            state.cfg->logger->warn(
                "[IdealSpace][row {}] recall mismatch: {} vs {} "
                "(iter_count={})",
                row,
                primary_recall,
                iter_recall,
                res.iter_count);
        }
    }

    return res;
}

void IdealSpaceAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    wait_per_query_ = state.pc
        ? state.pc->async_submit_and_wait
        : true;
}

bool IdealSpaceAsyncStrategy::RequiresPostQueryIoWait() const
{
    return !wait_per_query_;
}

std::string IdealSpaceAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    return BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult IdealSpaceAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>& gt_row)
{
    std::set<hnswlib::labeltype> gt_set;
    for (int i = 0; i < cur_k; ++i) {
        gt_set.insert(gt_row[i]);
    }

    int iter_count = 0;
    int dist_count = 0;
    bool found_all_gt = false;
    auto start = Clock::now();
    auto pq = state.hnsw_index->searchKnnRecallCost(
        state.QueryVector(row),
        cur_k,
        gt_set,
        iter_count,
        dist_count,
        found_all_gt);
    auto end = Clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();

    auto& ioer = IoWorkerForThread(state, thread_id);
    auto io_start = Clock::now();
    std::size_t missing = 0;
    std::size_t issued_io = 0;
    int pending = ioer.mapped_prefetch(
        ExtractIds(pq),
        *state.pc,
        static_cast<int>(row),
        wait_per_query_,
        &missing,
        &issued_io);
    if (wait_per_query_) {
        ioer.wait_all_completions(pending);
    }
    auto io_end = Clock::now();
    double io_elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(io_end - io_start).count();

    ExperimentQueryResult res =
        BuildResultFromQueue(pq, elapsed, iter_count, dist_count);
    res.found_all_gt = found_all_gt;
    res.io_time_us = io_elapsed;

    if (state.pc && state.pc->is_debug && state.cfg && state.cfg->logger) {
        double primary_recall =
            annfs::common::get_recall(res.result_ids, gt_row, cur_k);
        double iter_recall = primary_recall;
        if (res.iter_count > 0) {
            auto verify_pq = state.hnsw_index->searchKnnIdealTime(
                state.QueryVector(row),
                cur_k,
                res.iter_count);
            std::vector<unsigned long long> iter_ids;
            while (!verify_pq.empty()) {
                iter_ids.push_back(
                    static_cast<unsigned long long>(verify_pq.top().second));
                verify_pq.pop();
            }
            std::reverse(iter_ids.begin(), iter_ids.end());
            if (!iter_ids.empty()) {
                iter_recall = annfs::common::get_recall(
                    iter_ids,
                    gt_row,
                    std::min<int>(cur_k, iter_ids.size()));
            }
        }
        state.cfg->logger->debug(
            "[IdealSpaceAsync][row {}] iter={}, dist={}, found_gt={}, recall={}, "
            "recall@iter={}",
            row,
            res.iter_count,
            res.dist_count,
            res.found_all_gt,
            primary_recall,
            iter_recall);
        if (std::fabs(primary_recall - iter_recall) > 1e-3) {
            state.cfg->logger->warn(
                "[IdealSpaceAsync][row {}] recall mismatch: {} vs {} "
                "(iter_count={})",
                row,
                primary_recall,
                iter_recall,
                res.iter_count);
        }
    }

    return res;
}

ExperimentQueryResult OverlapIOStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    auto& ioer = IoWorkerForThread(state, thread_id);
    int pending_prefetch_ios = 0;
    std::size_t issued_prefetch_labels = 0;
    std::size_t mapped_missing = 0;
    std::size_t topk_labels = 0;
    auto start = Clock::now();
    auto sent_prefetch_labels = state.hnsw_index->searchKnnPrefetchMappedBaseline(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_iter,
        &ioer,
        state.pc->dataset.least_else_size,
        nullptr,
        &pending_prefetch_ios,
        static_cast<int>(row),
        &issued_prefetch_labels,
        &mapped_missing,
        nullptr,
        &topk_labels,
        state.pc && !state.pc->skip_recall);
    auto end = Clock::now();
    double hnsw_elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();

    ExperimentQueryResult result;
    result.hnsw_time_us = hnsw_elapsed;
    result.overlap = std::make_unique<OverlapQueryResult>();
    result.overlap->issued_label_set = std::move(sent_prefetch_labels);
    result.io_time_us = 0.0;
    return result;
}

std::string OverlapIOStrategy::ModeSubdir(const HnswIoState& state) const
{
    std::string worker = state.use_shared_io_worker ? "shared" : "per_thread";
    return "overlap_" + worker;
}

void MappedOffsetIOStrategy::LoadMap(const HnswIoState& state)
{
    // 读取并加载 label_map（shift_offbits 时跳过）。
    if (!state.pc || !state.pc->UseLabelMap()) {
        map_path_ = "(disabled)";
        return;
    }
    map_path_ = state.pc->label_map_path;
    annfs::io::MappedLabelOffsetResolver::Instance().LoadFromFile(
        map_path_,
        GlobalConfig::instance().LOGIC_BLOCK_SIZE);
}

void MappedOffsetIOStrategy::OnBeforeRun(const HnswIoState& state)
{
    // 实验前初始化：加载 map、重置统计器、打印运行参数。
    LoadMap(state);
    missing_total_.store(0, std::memory_order_relaxed);
    io_offsets_total_.store(0, std::memory_order_relaxed);
    ann_us_total_.store(0, std::memory_order_relaxed);
    map_us_total_.store(0, std::memory_order_relaxed);
    io_us_total_.store(0, std::memory_order_relaxed);
    query_total_.store(0, std::memory_order_relaxed);

    std::cout << "[MappedStrategy] dataset=" << state.args.dataset_name
              << " threads=" << state.num_threads
              << " search_ef=" << state.pc->dataset.search_ef
              << " topk=" << state.pc->topk
              << " io_depth=" << GlobalConfig::instance().RING_DEPTH
              << " io_worker_mode=" << (state.use_shared_io_worker ? "shared" : "per_thread")
              << "\n";
    if (state.pc) {
        std::cout << "[MappedStrategy] use_label_map=" << (GlobalConfig::instance().USE_LABEL_MAP ? "true" : "false")
                  << " label_decode_mode="
                  << GlobalConfig::LabelDecodeModeToString(GlobalConfig::instance().LABEL_DECODE_MODE_CFG)
                  << " mapped_span_mode=" << GlobalConfig::instance().MAPPED_SPAN_MODE
                  << "\n";
    }
}

void MappedOffsetIOStrategy::OnAfterRun(const HnswIoState& state, const std::string&)
{
    // 实验后汇总统计：平均 ANN/MAP/IO 耗时、缺失标签、总 IO。
    const std::size_t qn = query_total_.load(std::memory_order_relaxed);
    const double denom = qn > 0 ? static_cast<double>(qn) : 1.0;
    const double avg_ann = static_cast<double>(ann_us_total_.load(std::memory_order_relaxed)) / denom;
    const double avg_map = static_cast<double>(map_us_total_.load(std::memory_order_relaxed)) / denom;
    const double avg_io = static_cast<double>(io_us_total_.load(std::memory_order_relaxed)) / denom;
    std::cout << "Map path       : " << map_path_ << "\n";
    std::size_t map_size = 0;
    if (state.pc && state.pc->UseLabelMap()) {
        map_size = annfs::io::MappedLabelOffsetResolver::Instance().Size();
    }
    std::cout << "Map size       : " << map_size << "\n";
    std::cout << "Missing labels : " << missing_total_.load(std::memory_order_relaxed) << "\n";
    std::cout << "Total offsets  : " << io_offsets_total_.load(std::memory_order_relaxed) << "\n";
    std::cout << "Block size     : " << state.cfg->LOGIC_BLOCK_SIZE << "\n";
    std::cout << "Avg ANN us(strategy) : " << avg_ann << "\n";
    std::cout << "Avg MAP us(strategy) : " << avg_map << "\n";
    std::cout << "Avg IO us(strategy)  : " << avg_io << "\n";
    std::cout << "[MappedStrategy] avg_ann_us=" << avg_ann
              << " avg_map_us=" << avg_map
              << " avg_io_us=" << avg_io
              << " avg_hnswio_us=" << (avg_ann + avg_io)
              << " total_offsets=" << io_offsets_total_.load(std::memory_order_relaxed)
              << " missing_labels=" << missing_total_.load(std::memory_order_relaxed)
              << "\n";
}

std::vector<unsigned long long> MappedOffsetIOStrategy::DrainLabels(
    std::priority_queue<std::pair<float, hnswlib::labeltype>>& pq) const
{
    // 把 HNSW 返回的优先队列转成 label 列表（u64）。
    std::vector<unsigned long long> labels;
    labels.reserve(pq.size());
    while (!pq.empty()) {
        labels.push_back(static_cast<unsigned long long>(pq.top().second));
        pq.pop();
    }
    return labels;
}

ExperimentQueryResult MappedOffsetIOStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // 基线 mapped 同步策略：
    // ANN 搜索 -> map 规划计时 -> 同步 IO -> 累计统计。
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnn(state.QueryVector(row), cur_k);
    auto ann_end = Clock::now();
    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.result_ids = DrainLabels(pq);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto map_start = Clock::now();
    auto map_end = map_start;
    auto io_start = Clock::now();
    issue_mapped_io_sync(
        state,
        thread_id,
        static_cast<int>(row),
        res.result_ids,
        missing,
        issued_io);
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_io, std::memory_order_relaxed);
    return res;
}

void MappedOffsetIOAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    // 读取 async_submit_and_wait，决定是否每 query 阻塞等待。
    MappedOffsetIOStrategy::OnBeforeRun(state);
    wait_per_query_ = state.pc ? state.pc->async_submit_and_wait : true;
}

bool MappedOffsetIOAsyncStrategy::RequiresPostQueryIoWait() const
{
    // submit-only 模式下，runner 结束时需要统一 drain。
    return !wait_per_query_;
}

std::string MappedOffsetIOAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    // 输出目录后缀：per_query_wait / submit_only + worker 模式。
    return BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult MappedOffsetIOAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // mapped 异步策略：
    // ANN -> map -> async submit（可选每 query wait）-> 统计。
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnn(state.QueryVector(row), cur_k);
    auto ann_end = Clock::now();
    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.result_ids = DrainLabels(pq);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto map_start = Clock::now();
    auto map_end = map_start;
    auto io_start = Clock::now();
    issue_mapped_io_async(
        state,
        thread_id,
        static_cast<int>(row),
        res.result_ids,
        wait_per_query_,
        missing,
        issued_io);
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    if (res.io_time_us >= 0.0) {
        io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    }
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_io, std::memory_order_relaxed);
    return res;
}

ExperimentQueryResult MappedOffsetIOSpanOneShotStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // span-one-shot 策略：
    // 统一映射为 offset 列表，走普通 offset 同步读（精简路径）。
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnn(state.QueryVector(row), cur_k);
    auto ann_end = Clock::now();
    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.result_ids = DrainLabels(pq);

    auto map_start = Clock::now();
    auto plan = annfs::io::BuildReqPlanU64(
        res.result_ids, state.pc ? *state.pc : ParamConfig::instance());
    auto map_end = Clock::now();

    auto& ioer = IoManagerForThread(state, thread_id);
    auto io_start = Clock::now();
    const auto block_size =
        static_cast<std::uint64_t>(GlobalConfig::instance().LOGIC_BLOCK_SIZE);
    std::vector<off_t> offsets;
    offsets.reserve(plan.total_reqs);
    for (const auto& kv : plan.by_blocks) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, kv.first);
        for (const auto& req : kv.second) {
            const off_t base = req.base;
            for (std::uint32_t i = 0; i < blocks; ++i) {
                offsets.push_back(static_cast<off_t>(
                    static_cast<std::uint64_t>(base) +
                    static_cast<std::uint64_t>(i) * block_size));
            }
        }
    }
    if (!offsets.empty()) {
        ioer.batch_read_offset(offsets, static_cast<int>(row));
    }
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(plan.missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(plan.total_reqs, std::memory_order_relaxed);
    return res;
}

ExperimentQueryResult MappedEarlyStopStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // EarlyStop 同步策略：ANN 使用 early-stop，再走 mapped 同步读。
    int iter_count = 0;
    int dist_count = 0;
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnnEarly(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_hops);
        // dist_count,
        // iter_count);
    auto ann_end = Clock::now();

    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.iter_count = iter_count;
    res.dist_count = dist_count;
    res.result_ids = DrainLabels(pq);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto map_start = Clock::now();
    auto map_end = map_start;
    auto io_start = Clock::now();
    issue_mapped_io_sync(
        state,
        thread_id,
        static_cast<int>(row),
        res.result_ids,
        missing,
        issued_io);
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_io, std::memory_order_relaxed);
    return res;
}

void MappedEarlyStopAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    // EarlyStop 异步策略初始化。
    MappedEarlyStopStrategy::OnBeforeRun(state);
    wait_per_query_ = state.pc ? state.pc->async_submit_and_wait : true;
}

bool MappedEarlyStopAsyncStrategy::RequiresPostQueryIoWait() const
{
    // submit-only 模式需要 run 后统一 wait。
    return !wait_per_query_;
}

std::string MappedEarlyStopAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    // 输出目录后缀。
    return BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult MappedEarlyStopAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // EarlyStop 异步策略执行。
    int iter_count = 0;
    int dist_count = 0;
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnnEarly(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_hops);
        // dist_count,
        // iter_count);
    auto ann_end = Clock::now();

    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.iter_count = iter_count;
    res.dist_count = dist_count;
    res.result_ids = DrainLabels(pq);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto map_start = Clock::now();
    auto map_end = map_start;
    auto io_start = Clock::now();
    issue_mapped_io_async(
        state,
        thread_id,
        static_cast<int>(row),
        res.result_ids,
        wait_per_query_,
        missing,
        issued_io);
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    if (res.io_time_us >= 0.0) {
        io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    }
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_io, std::memory_order_relaxed);
    return res;
}

void MappedEarlyStopSpanPoolSyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    // 同步模式统一由 NormalIOWorker 在 worker 内部处理。
    MappedOffsetIOStrategy::OnBeforeRun(state);
}

void MappedEarlyStopSpanPoolSyncStrategy::OnAfterRun(const HnswIoState& state, const std::string& output_dir)
{
    // 复用父类汇总逻辑。
    MappedOffsetIOStrategy::OnAfterRun(state, output_dir);
}

ExperimentQueryResult MappedEarlyStopSpanPoolSyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // EarlyStop + span-pool 同步策略：
    // 统一走 worker 的 mapped 同步入口，策略层不直接操作 span reader。
    int iter_count = 0;
    int dist_count = 0;
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnnEarly(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_hops);
        // dist_count,
        // iter_count);
    auto ann_end = Clock::now();

    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.iter_count = iter_count;
    res.dist_count = dist_count;
    res.result_ids = DrainLabels(pq);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto map_start = Clock::now();
    auto map_end = map_start;

    auto io_start = Clock::now();
    issue_mapped_io_sync(
        state,
        thread_id,
        static_cast<int>(row),
        res.result_ids,
        missing,
        issued_io);
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_io, std::memory_order_relaxed);
    return res;
}

void MappedEarlyStopSpanPoolAsyncStrategy::OnBeforeRun(const HnswIoState& state)
{
    // span-pool 异步策略初始化。
    MappedEarlyStopSpanPoolSyncStrategy::OnBeforeRun(state);
    wait_per_query_ = state.pc ? state.pc->async_submit_and_wait : false;
}

void MappedEarlyStopSpanPoolAsyncStrategy::OnAfterRun(
    const HnswIoState& state,
    const std::string& output_dir)
{
    // submit-only 模式下由 runner 的 post-query drain 统一等待 worker 完成。
    MappedEarlyStopSpanPoolSyncStrategy::OnAfterRun(state, output_dir);
}

std::string MappedEarlyStopSpanPoolAsyncStrategy::ModeSubdir(const HnswIoState& state) const
{
    // 输出目录后缀。
    return BuildAsyncModeDir(state, wait_per_query_);
}

ExperimentQueryResult MappedEarlyStopSpanPoolAsyncStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // EarlyStop + span-pool 异步策略执行。
    int iter_count = 0;
    int dist_count = 0;
    auto ann_start = Clock::now();
    auto pq = state.hnsw_index->searchKnnEarly(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_hops);
        // dist_count,
        // iter_count);
    auto ann_end = Clock::now();

    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.iter_count = iter_count;
    res.dist_count = dist_count;
    res.result_ids = DrainLabels(pq);

    std::size_t missing = 0;
    std::size_t issued_io = 0;
    auto map_start = Clock::now();
    auto map_end = map_start;

    auto io_start = Clock::now();
    issue_mapped_io_async(
        state,
        thread_id,
        static_cast<int>(row),
        res.result_ids,
        wait_per_query_,
        missing,
        issued_io);
    auto io_end = Clock::now();
    res.io_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        io_end - io_start).count();

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(map_end - map_start).count()),
        std::memory_order_relaxed);
    if (res.io_time_us >= 0.0) {
        io_us_total_.fetch_add(static_cast<std::uint64_t>(res.io_time_us), std::memory_order_relaxed);
    }
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_io, std::memory_order_relaxed);
    return res;
}

std::string MappedOverlapIOStrategy::ModeSubdir(const HnswIoState& state) const
{
    // overlap 固定是 submit-only，按 worker 模式区分目录。
    std::string worker = state.use_shared_io_worker ? "shared" : "per_thread";
    return "mapped_overlap_submit_only_" + worker;
}

ExperimentQueryResult MappedOverlapIOStrategy::ExecuteSearch(
    const HnswIoState& state,
    size_t row,
    size_t thread_id,
    int cur_k,
    const std::vector<hnswlib::labeltype>&)
{
    // overlap 基线策略：
    // 在 HNSW 搜索过程中通过 IOuringManager 实时 prefetch（mapped 路径）。
    auto& io_worker = IoWorkerForThread(state, thread_id);
    int pending_prefetch_ios = 0;
    std::size_t issued_prefetch_labels = 0;
    std::size_t mapped_missing = 0;
    std::size_t topk_labels = 0;
    auto ann_start = Clock::now();
    std::unordered_set<unsigned long long> sent_prefetch_labels = state.hnsw_index->searchKnnPrefetchMappedBaseline(
        state.QueryVector(row),
        cur_k,
        state.pc->dataset.stable_iter,
        &io_worker,
        state.pc->dataset.least_else_size,
        nullptr,
        &pending_prefetch_ios,
        static_cast<int>(row),
        &issued_prefetch_labels,
        &mapped_missing,
        nullptr,
        &topk_labels,
        state.pc && !state.pc->skip_recall);
    auto ann_end = Clock::now();
    ExperimentQueryResult res;
    res.hnsw_time_us = std::chrono::duration_cast<std::chrono::microseconds>(ann_end - ann_start).count();
    res.overlap = std::make_unique<OverlapQueryResult>();
    res.overlap->issued_label_set = std::move(sent_prefetch_labels);
    res.io_time_us = 0.0;

    ann_us_total_.fetch_add(static_cast<std::uint64_t>(res.hnsw_time_us), std::memory_order_relaxed);
    map_us_total_.fetch_add(0, std::memory_order_relaxed);
    query_total_.fetch_add(1, std::memory_order_relaxed);
    missing_total_.fetch_add(mapped_missing, std::memory_order_relaxed);
    io_offsets_total_.fetch_add(issued_prefetch_labels, std::memory_order_relaxed);
    return res;
}

} // namespace annfs::tests
