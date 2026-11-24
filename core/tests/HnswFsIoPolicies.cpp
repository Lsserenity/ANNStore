#include "tests/HnswFsIoPolicies.hpp"

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

namespace annfs::tests {

bool FsSyncIoExecutionPolicy::Init(ParamConfig& pc,
                                   GlobalConfig& cfg,
                                   int num_threads,
                                   const nlohmann::json* root)
{
    (void)root;
    (void)pc;
    if (cfg.logger) {
        cfg.logger->info("[FsSyncIO] Init begin num_threads={} dev_path={} ring_depth={} io_worker_mode={}",
                         num_threads,
                         pc.dev_path,
                         GlobalConfig::instance().RING_DEPTH,
                         GlobalConfig::instance().IO_WORKER_MODE);
    }
    // FsSync uses IOuringManager fixed-buffer registration, which needs
    // LOGIC_BLOCK_SIZE initialized from device before buffer alignment.
    get_block_size(pc.dev_path);
    if (cfg.logger) {
        cfg.logger->info("[FsSyncIO] get_block_size done block_size={} disk_size={}",
                         GlobalConfig::instance().LOGIC_BLOCK_SIZE,
                         GlobalConfig::instance().DISK_SIZE);
    }
    cfg_ = &cfg;
    ioers_.clear();
    ioers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        if (cfg.logger) {
            cfg.logger->info("[FsSyncIO] create IOuringManager begin index={}/{}",
                             i + 1,
                             num_threads);
        }
        ioers_.emplace_back(std::make_unique<IOuringManager>());
    }
    return true;
}

void FsSyncIoExecutionPolicy::SetQueryListener(annfs::io::QueryIoEventListener* listener)
{
    for (auto& ioer : ioers_) {
        if (ioer) {
            ioer->set_query_listener(listener);
        }
    }
}

void FsSyncIoExecutionPolicy::Execute(const std::vector<std::string>& path_list,
                                      int row,
                                      size_t thread_id)
{
    if (path_list.empty()) {
        total_queries_executed_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    total_path_candidates_.fetch_add(
        static_cast<long long>(path_list.size()),
        std::memory_order_relaxed);
    total_queries_executed_.fetch_add(1, std::memory_order_relaxed);
    int submitted = ioers_[thread_id]->submit_files_and_wait(path_list, row);
    if (submitted > 0) {
        effective_io_counter_.fetch_add(static_cast<long long>(submitted), std::memory_order_relaxed);
    }
}

void FsSyncIoExecutionPolicy::Finalize() {}

void FsSyncIoExecutionPolicy::EmitSummary(const ParamConfig& pc,
                                          GlobalConfig& cfg) const
{
    long long effective_ios = effective_io_counter_.load(std::memory_order_relaxed);
    long long total_paths = total_path_candidates_.load(std::memory_order_relaxed);
    long long total_queries = total_queries_executed_.load(std::memory_order_relaxed);
    long long expected_ios = static_cast<long long>(pc.topk) *
                             static_cast<long long>(pc.query_cnt);
    long long constant_ios = 100LL * static_cast<long long>(pc.query_cnt);
    std::uint64_t submitted_total = 0;
    std::uint64_t completed_total = 0;
    std::uint64_t wait_calls_total = 0;
    std::uint64_t wait_requested_total = 0;
    bool diagnostics_enabled = false;
    for (const auto& ioer : ioers_) {
        if (!ioer) {
            continue;
        }
        diagnostics_enabled = diagnostics_enabled || ioer->diagnostics_enabled();
        submitted_total += ioer->submitted_total();
        completed_total += ioer->completed_total();
        wait_calls_total += ioer->wait_calls_total();
        wait_requested_total += ioer->wait_requested_total();
    }
    if (cfg.logger) {
        cfg.logger->info(
            "FsIO effective IO submissions: {} (topk-based {} = topk {} x queries {}, constant_100q={})",
            effective_ios,
            expected_ios,
            pc.topk,
            pc.query_cnt,
            constant_ios);
        cfg.logger->info(
            "FsIO submit diagnostics: total_paths={} total_queries={} avg_paths_per_query={:.3f} submit_ratio={:.6f}",
            total_paths,
            total_queries,
            total_queries > 0 ? static_cast<double>(total_paths) / static_cast<double>(total_queries) : 0.0,
            total_paths > 0 ? static_cast<double>(effective_ios) / static_cast<double>(total_paths) : 0.0);
        if (diagnostics_enabled) {
            cfg.logger->debug(
                "FsIO completion diagnostics: submitted_total={} completed_total={} outstanding_delta={} wait_calls={} wait_requested={}",
                submitted_total,
                completed_total,
                static_cast<long long>(submitted_total) - static_cast<long long>(completed_total),
                wait_calls_total,
                wait_requested_total);
        }
    } else {
        std::cout << "[FsIO] effective_io=" << effective_ios
                  << " expected=" << expected_ios
                  << " (topk=" << pc.topk
                  << " queries=" << pc.query_cnt
                  << ") constant_100q=" << constant_ios
                  << std::endl;
    }
    std::cout << "有效IO=" << constant_ios
              << " # 100 x query_cnt (query_cnt="
              << pc.query_cnt << ")" << std::endl;
}

nlohmann::json FsSyncIoExecutionPolicy::BuildStatsMeta(const ParamConfig& pc) const
{
    long long effective_ios = effective_io_counter_.load(std::memory_order_relaxed);
    long long total_paths = total_path_candidates_.load(std::memory_order_relaxed);
    long long total_queries = total_queries_executed_.load(std::memory_order_relaxed);
    long long expected_ios = static_cast<long long>(pc.topk) *
                             static_cast<long long>(pc.query_cnt);
    long long constant_ios = 100LL * static_cast<long long>(pc.query_cnt);
    std::uint64_t submitted_total = 0;
    std::uint64_t completed_total = 0;
    std::uint64_t wait_calls_total = 0;
    std::uint64_t wait_requested_total = 0;
    bool diagnostics_enabled = false;
    for (const auto& ioer : ioers_) {
        if (!ioer) {
            continue;
        }
        diagnostics_enabled = diagnostics_enabled || ioer->diagnostics_enabled();
        submitted_total += ioer->submitted_total();
        completed_total += ioer->completed_total();
        wait_calls_total += ioer->wait_calls_total();
        wait_requested_total += ioer->wait_requested_total();
    }

    if (diagnostics_enabled && cfg_ && cfg_->logger) {
        cfg_->logger->debug(
            "FsIO completion totals: submitted_total={} completed_total={} outstanding_delta={} wait_calls={} wait_requested={}",
            submitted_total,
            completed_total,
            static_cast<long long>(submitted_total) - static_cast<long long>(completed_total),
            wait_calls_total,
            wait_requested_total);
    }

    nlohmann::json meta = {
        {"effective_io_reported", effective_ios},
        {"total_path_candidates", total_paths},
        {"total_queries_executed", total_queries},
        {"expected_topk_io", expected_ios},
        {"ideal_100q_io", constant_ios},
        {"query_cnt", pc.query_cnt},
        {"topk", pc.topk}
    };
    if (diagnostics_enabled) {
        meta["submitted_total"] = submitted_total;
        meta["completed_total"] = completed_total;
        meta["outstanding_delta"] =
            static_cast<long long>(submitted_total) -
            static_cast<long long>(completed_total);
        meta["wait_calls"] = wait_calls_total;
        meta["wait_requested"] = wait_requested_total;
    }
    return meta;
}

bool FsOpenOnlyExecutionPolicy::Init(ParamConfig& pc,
                                     GlobalConfig& cfg,
                                     int num_threads,
                                     const nlohmann::json* root)
{
    (void)cfg;
    (void)num_threads;
    (void)root;
    use_direct_io_ = pc.use_direct_io;
    return true;
}

void FsOpenOnlyExecutionPolicy::SetQueryListener(annfs::io::QueryIoEventListener* listener)
{
    (void)listener;
}

void FsOpenOnlyExecutionPolicy::Execute(const std::vector<std::string>& path_list,
                                        int row,
                                        size_t thread_id)
{
    (void)row;
    (void)thread_id;
    if (path_list.empty()) {
        return;
    }

    const int open_flags = use_direct_io_ ? (O_RDONLY | O_DIRECT) : O_RDONLY;
    long long failed = 0;
    for (const auto& path : path_list) {
        int fd = open(path.c_str(), open_flags);
        if (fd < 0) {
            ++failed;
            continue;
        }
        close(fd);
    }
    if (failed > 0) {
        open_fail_counter_.fetch_add(failed, std::memory_order_relaxed);
    }
    auto& gc = GlobalConfig::instance();
    if (gc.logger && gc.logger->should_log(spdlog::level::debug)) {
        gc.logger->debug(
            "[FS_OPENONLY_QUERY] query_id={} paths={} failed={} use_direct_io={}",
            row,
            path_list.size(),
            failed,
            use_direct_io_ ? "true" : "false");
    }
}

void FsOpenOnlyExecutionPolicy::NotifySkipped(size_t skipped)
{
    (void)skipped;
}

void FsOpenOnlyExecutionPolicy::Finalize() {}

void FsOpenOnlyExecutionPolicy::EmitSummary(const ParamConfig& pc,
                                            GlobalConfig& cfg) const
{
    long long expected_opens = static_cast<long long>(pc.topk) *
                               static_cast<long long>(pc.query_cnt);
    long long open_failures = open_fail_counter_.load(std::memory_order_relaxed);

    if (cfg.logger) {
        cfg.logger->info(
            "FsOpenOnly open/close attempts: topk-based {} = topk {} x queries {}, open_failures={}",
            expected_opens,
            pc.topk,
            pc.query_cnt,
            open_failures);
    } else {
        std::cout << "[FsOpenOnly] expected=" << expected_opens
                  << " (topk=" << pc.topk
                  << " queries=" << pc.query_cnt
                  << ") open_failures=" << open_failures
                  << std::endl;
    }
}

nlohmann::json FsOpenOnlyExecutionPolicy::BuildStatsMeta(const ParamConfig& pc) const
{
    long long expected_opens = static_cast<long long>(pc.topk) *
                               static_cast<long long>(pc.query_cnt);
    long long open_failures = open_fail_counter_.load(std::memory_order_relaxed);

    return {
        {"expected_topk_open_count", expected_opens},
        {"open_failures", open_failures},
        {"query_cnt", pc.query_cnt},
        {"topk", pc.topk}
    };
}

bool FsAsyncIoExecutionPolicy::Init(ParamConfig& pc,
                                    GlobalConfig& cfg,
                                    int num_threads,
                                    const nlohmann::json* root)
{
    cfg_ = &cfg;
    num_threads_ = num_threads;
    effective_io_counter_.store(0, std::memory_order_relaxed);
    total_path_candidates_.store(0, std::memory_order_relaxed);
    total_queries_executed_.store(0, std::memory_order_relaxed);
    execute_wall_us_total_.store(0, std::memory_order_relaxed);
    finalize_wait_us_.store(0, std::memory_order_relaxed);
    async_mode_ = "per_query_wait";
    if (root) {
        async_mode_ = root->value("fs_async_mode", std::string("per_query_wait"));
    }
    std::transform(async_mode_.begin(), async_mode_.end(), async_mode_.begin(),
                   [](char c) { return static_cast<char>(std::tolower(c)); });

    overlap_mode_ = (async_mode_ == "overlap");
    if (!overlap_mode_ &&
        async_mode_ != "per_query_wait" &&
        async_mode_ != "submit_wait") {
        if (cfg.logger) {
            cfg.logger->warn(
                "Unknown fs_async_mode={}, fallback to per_query_wait",
                async_mode_);
        }
        async_mode_ = "per_query_wait";
        overlap_mode_ = false;
    }

    shared_worker_ = (GlobalConfig::instance().IO_WORKER_MODE == "shared");
    if (shared_worker_ && !overlap_mode_) {
        shared_worker_ = false;
        if (cfg.logger) {
            cfg.logger->warn(
                "io_worker_mode=shared requires fs_async_mode=overlap; fallback to per_thread");
        }
    }
    get_block_size(pc.dev_path);
    ioers_.clear();
    int ioer_count = shared_worker_ ? 1 : num_threads;
    ioers_.reserve(ioer_count);

    if (shared_worker_ && cfg.logger) {
        cfg.logger->info("IO worker mode: shared ({} threads -> 1 ring)", num_threads);
    }

    for (int i = 0; i < ioer_count; ++i) {
        auto ioer = std::make_unique<IOuringManager>();
        if (overlap_mode_) {
            ioer->enable_async_submit(shared_worker_);
            ioer->configure_async_submit(pc.async_submit_and_wait);
        }
        ioers_.emplace_back(std::move(ioer));
    }
    return true;
}

void FsAsyncIoExecutionPolicy::SetQueryListener(annfs::io::QueryIoEventListener* listener)
{
    for (auto& ioer : ioers_) {
        if (ioer) {
            ioer->set_query_listener(listener);
        }
    }
}

void FsAsyncIoExecutionPolicy::Execute(const std::vector<std::string>& path_list,
                                       int row,
                                       size_t thread_id)
{
    total_queries_executed_.fetch_add(1, std::memory_order_relaxed);
    if (path_list.empty()) {
        return;
    }
    total_path_candidates_.fetch_add(
        static_cast<long long>(path_list.size()),
        std::memory_order_relaxed);

    IOuringManager& ioer = shared_worker_ ? *ioers_[0] : *ioers_[thread_id];
    auto execute_start = std::chrono::high_resolution_clock::now();
    int submitted = 0;
    if (overlap_mode_) {
        submitted = ioer.prefetch_files_overlap(path_list, row);
    } else {
        submitted = ioer.submit_files_and_wait(path_list, row);
    }
    auto execute_end = std::chrono::high_resolution_clock::now();
    execute_wall_us_total_.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                execute_end - execute_start).count()),
        std::memory_order_relaxed);
    if (submitted > 0) {
        effective_io_counter_.fetch_add(
            static_cast<long long>(submitted),
            std::memory_order_relaxed);
    }
}

void FsAsyncIoExecutionPolicy::Finalize()
{
    if (!overlap_mode_) {
        return;
    }
    auto wait_start = std::chrono::high_resolution_clock::now();
    if (shared_worker_) {
        ioers_[0]->wait_all_completions_all();
    } else {
        for (auto& ioer : ioers_) {
            ioer->wait_all_completions_all();
        }
    }
    auto wait_end = std::chrono::high_resolution_clock::now();
    finalize_wait_us_.store(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                wait_end - wait_start).count()),
        std::memory_order_relaxed);
}

void FsAsyncIoExecutionPolicy::EmitSummary(const ParamConfig& pc,
                                           GlobalConfig& cfg) const
{
    (void)pc;
    const long long total_queries =
        total_queries_executed_.load(std::memory_order_relaxed);
    const long long total_paths =
        total_path_candidates_.load(std::memory_order_relaxed);
    const long long effective_ios =
        effective_io_counter_.load(std::memory_order_relaxed);
    const std::uint64_t execute_wall_us =
        execute_wall_us_total_.load(std::memory_order_relaxed);
    const std::uint64_t finalize_wait_us =
        finalize_wait_us_.load(std::memory_order_relaxed);
    const double denom = total_queries > 0
        ? static_cast<double>(total_queries)
        : 1.0;
    if (cfg.logger) {
        cfg.logger->info(
            "FsIOAsync mode={} worker_mode={} effective_io={} total_paths={} total_queries={} avg_execute_wall_us={:.3f} finalize_wait_us={}",
            async_mode_,
            shared_worker_ ? "shared" : "per_thread",
            effective_ios,
            total_paths,
            total_queries,
            static_cast<double>(execute_wall_us) / denom,
            finalize_wait_us);
    }
}

nlohmann::json FsAsyncIoExecutionPolicy::BuildStatsMeta(const ParamConfig& pc) const
{
    std::uint64_t submitted_total = 0;
    std::uint64_t completed_total = 0;
    std::uint64_t wait_calls_total = 0;
    std::uint64_t wait_requested_total = 0;
    bool diagnostics_enabled = false;
    for (const auto& ioer : ioers_) {
        if (!ioer) {
            continue;
        }
        diagnostics_enabled = diagnostics_enabled || ioer->diagnostics_enabled();
        submitted_total += ioer->submitted_total();
        completed_total += ioer->completed_total();
        wait_calls_total += ioer->wait_calls_total();
        wait_requested_total += ioer->wait_requested_total();
    }

    const long long total_queries =
        total_queries_executed_.load(std::memory_order_relaxed);
    const std::uint64_t execute_wall_us =
        execute_wall_us_total_.load(std::memory_order_relaxed);
    const double denom = total_queries > 0
        ? static_cast<double>(total_queries)
        : 1.0;

    nlohmann::json meta = {
        {"query_cnt", pc.query_cnt},
        {"topk", pc.topk},
        {"async_mode", async_mode_},
        {"overlap_mode", overlap_mode_},
        {"shared_worker", shared_worker_},
        {"worker_count", static_cast<int>(ioers_.size())},
        {"effective_io_reported", effective_io_counter_.load(std::memory_order_relaxed)},
        {"total_path_candidates", total_path_candidates_.load(std::memory_order_relaxed)},
        {"total_queries_executed", total_queries},
        {"execute_wall_us_total", execute_wall_us},
        {"avg_execute_wall_us", static_cast<double>(execute_wall_us) / denom},
        {"finalize_wait_us", finalize_wait_us_.load(std::memory_order_relaxed)}
    };
    if (diagnostics_enabled) {
        meta["submitted_total"] = submitted_total;
        meta["completed_total"] = completed_total;
        meta["outstanding_delta"] =
            static_cast<long long>(submitted_total) -
            static_cast<long long>(completed_total);
        meta["wait_calls"] = wait_calls_total;
        meta["wait_requested"] = wait_requested_total;
    }
    return meta;
}

std::string FsAsyncIoExecutionPolicy::ResultFolder() const
{
    return overlap_mode_ ? "FsAsyncOverlap_results" : "FsAsync_results";
}

std::string FsAsyncIoExecutionPolicy::ModeDir() const
{
    std::string worker_mode_dir = shared_worker_ ? "shared" : "per_thread";
    if (async_mode_.empty()) {
        return worker_mode_dir;
    }
    return async_mode_ + "_" + worker_mode_dir;
}

} // namespace annfs::tests
