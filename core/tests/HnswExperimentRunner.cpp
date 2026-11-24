#include "tests/HnswExperimentRunner.hpp"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include "json_utils.hpp"
#include "recall.hpp"
#include "tests/HnswExperimentBatch.hpp"
#include "tests/HnswInternalMonitors.hpp"
#include "tests/QueryLatencyTracker.hpp"

namespace annfs::tests {

namespace {
using Clock = std::chrono::high_resolution_clock;

long current_tid()
{
    return static_cast<long>(::syscall(SYS_gettid));
}

std::string join_tid_csv(const std::unordered_set<long>& tids)
{
    if (tids.empty()) {
        return {};
    }
    std::vector<long> sorted_tids(tids.begin(), tids.end());
    std::sort(sorted_tids.begin(), sorted_tids.end());
    std::string out;
    out.reserve(sorted_tids.size() * 8);
    for (size_t i = 0; i < sorted_tids.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        out += std::to_string(sorted_tids[i]);
    }
    return out;
}

long long epoch_ms()
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
}

std::string batch_marker_path()
{
    /**
        环境变量：marker标签的输出文件路径
    */
    const char* path = std::getenv("ANNFS_BATCH_MARKER_PATH");
    return path ? std::string(path) : std::string();
}

std::string batch_marker_ready_gate_path()
{
    const char* path = std::getenv("ANNFS_BATCH_MARKER_READY_FILE");
    return path ? std::string(path) : std::string();
}

int marker_ready_wait_timeout_sec()
{
    const char* v = std::getenv("ANNFS_BATCH_MARKER_READY_TIMEOUT_SEC");
    if (!v) {
        return 120;
    }
    try {
        int x = std::stoi(v);
        return x > 0 ? x : 120;
    } catch (...) {
        return 120;
    }
}

void wait_for_marker_ready_gate_if_needed()
{
    const std::string gate = batch_marker_ready_gate_path();
    if (gate.empty()) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(marker_ready_wait_timeout_sec());
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(gate)) {
            std::cout << "[BATCH_MARKER_GATE] ready file detected: "
                      << gate << std::endl;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "[BATCH_MARKER_GATE] timeout waiting ready file: "
              << gate << std::endl;
}

void append_batch_marker(const std::string& path,
                         const nlohmann::json& entry)
{
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << entry.dump() << "\n";
    }
}

void emit_batch_marker(const std::string& path,
                       const nlohmann::json& entry)
{
    if (path.empty()) {
        return;
    }
    std::cout << "[BATCH_MARKER] " << entry.dump() << std::endl;
    append_batch_marker(path, entry);
}

std::string read_proc_field(const std::string& path,
                            const std::string& key)
{
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(key, 0) == 0) {
            return line.substr(key.size());
        }
    }
    return " n/a";
}

void emit_memory_snapshot(const std::string& prefix)
{
    auto logger = GlobalConfig::instance().logger;
    if (!logger) {
        return;
    }
    logger->debug("{} vmrss={} vmhwm={} mem_available={}",
                  prefix,
                  read_proc_field("/proc/self/status", "VmRSS:"),
                  read_proc_field("/proc/self/status", "VmHWM:"),
                  read_proc_field("/proc/meminfo", "MemAvailable:"));
}

} // namespace

std::string ExperimentStrategy::CustomOutputDir(const HnswIoState& state,
                                                const HnswIoCliArgs& args) const
{
    (void)state;
    (void)args;
    return "";
}

int ExecuteHnswExperimentWithState(HnswIoState& state,
                                   ExperimentStrategy& strategy)
{
    //  - 通过ParallelForQueries并行执行查询，每个查询执行策略指定的搜索逻辑，并记录HNSW时间、IO时间、召回率等指标；
    //  - 如果策略需要，在查询后等待所有异步IO完成；
    //  - 计算总的墙钟时间，并将结果持久化到指定目录下的JSON文件中；
    //  - 支持实验策略在运行前后执行自定义逻辑（OnBeforeRun和OnAfterRun）；
    if (strategy.UsesAsyncIoSubmit() &&
        !state.use_shared_io_worker &&
        strategy.Name().find("Overlap") != std::string::npos) {
        state.use_shared_io_worker = true;
        if (state.cfg && state.cfg->logger) {
            state.cfg->logger->info(
                "Force io_worker_mode=shared for async overlap strategy {}",
                strategy.Name());
        }
    }
    const int resolved_run_index = ResolveRunIndex(state.args);
    const auto execute_start_tp = std::chrono::steady_clock::now();
    std::cout << "[RUN] start"
              << " strategy=" << strategy.Name()
              << " dataset=" << state.args.dataset_name
              << " repeat_id=" << state.args.repeat_id
              << " run_index=" << resolved_run_index
              << " threads=" << state.num_threads
              << " query_cnt=" << state.query_cnt
              << "\n"
              << std::flush;

    strategy.OnBeforeRun(state);
    std::cout << "[RUN] after_on_before_run"
              << " run_index=" << resolved_run_index
              << "\n"
              << std::flush;
    emit_memory_snapshot("[RUN_MEM] after_on_before_run run_index=" +
                         std::to_string(resolved_run_index));
    if (state.use_shared_io_worker && !strategy.UsesAsyncIoSubmit()) {
        if (state.cfg && state.cfg->logger) {
            state.cfg->logger->warn(
                "io_worker_mode=shared requires async IO strategies; "
                "fallback to per_thread for {}",
                strategy.Name());
        }
        state.use_shared_io_worker = false;
    }
    if (strategy.UsesAsyncIoSubmit()) {
        bool shared_worker = state.use_shared_io_worker;
        for (auto& ioer : state.ioers) {
            if (ioer) {
                ioer->enable_async_submit(shared_worker);
                if (state.pc) {
                    ioer->configure_async_submit(
                        state.pc->async_submit_and_wait);
                }
            }
        }
    }

    const size_t query_cnt = static_cast<size_t>(state.query_cnt);
    std::vector<std::vector<double>> costs(
        3, std::vector<double>(query_cnt, 0.0));
    std::vector<std::vector<double>> recalls(
        1, std::vector<double>(query_cnt, 0.0));
    std::vector<std::vector<double>> iter_dist_counts;
    size_t iter_metrics = 0;
    if (strategy.RequiresIterDist()) {
        iter_metrics = strategy.ReportsFoundGtFlag() ? 3 : 2;
        iter_dist_counts.assign(
            iter_metrics, std::vector<double>(query_cnt, 0.0));
    }
    std::vector<double> thread_query_io_us(state.num_threads, 0.0);

    std::unique_ptr<QueryLatencyTracker> query_tracker;
    std::unique_ptr<ScopedQueryListenerBinding> listener_binding;
    QueryLatencyTracker* tracker_ptr = nullptr;
    if (state.args.enable_query_timeline) {
        query_tracker = std::make_unique<QueryLatencyTracker>(query_cnt);
        listener_binding = std::make_unique<ScopedQueryListenerBinding>(
            state.ioers, query_tracker.get());
        tracker_ptr = query_tracker.get();
    }

    bool skip_recall = state.pc && state.pc->skip_recall;
    const bool overlap_prefetch_recall_mode =
        strategy.Name().find("OffsetOverlap") != std::string::npos ||
        strategy.Name().find("EarlyStopOverlap") != std::string::npos;
    const bool compute_overlap_recall =
        !skip_recall && overlap_prefetch_recall_mode;
    const bool compute_regular_recall = !skip_recall && !overlap_prefetch_recall_mode;
    if (skip_recall && state.cfg && state.cfg->logger) {
        state.cfg->logger->info("Recall computation disabled (skip_recall=true)");
    }
    if (compute_overlap_recall && state.cfg && state.cfg->logger) {
        state.cfg->logger->info(
            "Overlap issued-label recall computation enabled");
    }


    const std::string marker_path = batch_marker_path();
    const bool has_pc = (state.pc != nullptr);
    if (!has_pc) {
        if (state.cfg && state.cfg->logger) {
            state.cfg->logger->error("PrepareState returned null ParamConfig");
        }
        return 1;
    }
    auto monitors = BuildInternalMonitors(state.args, *state.pc, strategy.Name());
    const bool monitors_enabled = !monitors.empty();
    const bool emit_query_markers = !marker_path.empty() && monitors_enabled;
    const int run_index = resolved_run_index;
    ::setenv("ANNFS_INTERNAL_REPORT_DATASET", state.args.dataset_name.c_str(), 1);
    ::setenv("ANNFS_INTERNAL_REPORT_THREADS",
             std::to_string(state.pc->num_threads).c_str(), 1);
    ::setenv("ANNFS_INTERNAL_REPORT_REPEAT_ID",
             std::to_string(state.args.repeat_id).c_str(), 1);
    ::setenv("ANNFS_INTERNAL_REPORT_RUN_INDEX",
             std::to_string(run_index).c_str(), 1);
    ::setenv("ANNFS_INTERNAL_TARGET_PID",
             std::to_string(static_cast<long long>(::getpid())).c_str(), 1);
    ::unsetenv("ANNFS_INTERNAL_TRACE_MM_PAGECACHE_FILTER_TIDS");
    bool query_phase_active = false;
    // 提交 query 开始的 marker 
    //  存在问题 应该全部对齐 使用一样的 而不是这里使用 steady_clock 而其他地方使用 high 
    std::chrono::steady_clock::time_point query_phase_start_tp;

    for (const auto& m : monitors) {
        if (!m->Enabled()) {
            continue;
        }
        std::cout << "[RUN] monitor_start"
                  << " run_index=" << run_index
                  << " name=" << m->Name()
                  << "\n"
                  << std::flush;
        if (!m->Start()) {
            if (state.cfg && state.cfg->logger) {
                state.cfg->logger->error(
                    "Internal monitor start failed: {}", m->Name());
            }
            return 2;
        }
        std::cout << "[RUN] monitor_started"
                  << " run_index=" << run_index
                  << " name=" << m->Name()
                  << "\n"
                  << std::flush;
    }
    const char* warmup_raw = std::getenv("ANNFS_INTERNAL_MONITOR_WARMUP_MS");
    int monitor_warmup_ms = 0;
    if (warmup_raw && *warmup_raw) {
        try {
            monitor_warmup_ms = std::max(0, std::stoi(warmup_raw));
        } catch (...) {
            monitor_warmup_ms = 0;
        }
    }
    if (monitor_warmup_ms > 0 && monitors_enabled) {
        std::this_thread::sleep_for(std::chrono::milliseconds(monitor_warmup_ms));
        if (state.cfg && state.cfg->logger) {
            state.cfg->logger->info("Internal monitor warmup waited: {} ms",
                                    monitor_warmup_ms);
        }
    }

    if (emit_query_markers) {
        query_phase_active = true;
        query_phase_start_tp = std::chrono::steady_clock::now();
        nlohmann::json start_marker = {
            {"event", "phase_start"},
            {"phase", "query"},
            {"dataset", state.args.dataset_name},
            {"repeat_id", state.args.repeat_id},
            {"num_threads", state.num_threads},
            {"run_index", run_index},
            {"ts_ms", epoch_ms()}
        };
        emit_batch_marker(marker_path, start_marker);
    }

    std::cout << "[RUN] query_phase_start"
              << " run_index=" << run_index
              << " monitors=" << (monitors_enabled ? "true" : "false")
              << " query_cnt=" << query_cnt
              << " threads=" << state.num_threads
              << "\n"
              << std::flush;
    emit_memory_snapshot("[RUN_MEM] query_phase_start run_index=" +
                         std::to_string(run_index));
    
    std::unordered_set<long> monitor_tids;
    std::mutex monitor_tids_mu;
    monitor_tids.insert(current_tid());
    std::atomic<size_t> completed_queries{0};

    auto allstart = Clock::now();

    ParallelForQueries(
        state,
        [&, tracker_ptr](size_t row, size_t threadId) {
            {
                const long tid = current_tid();
                std::lock_guard<std::mutex> lk(monitor_tids_mu);
                monitor_tids.insert(tid);
            }
            const bool need_gt =
                strategy.RequiresGroundTruth() || compute_regular_recall || compute_overlap_recall;
            std::vector<hnswlib::labeltype> gt_tmp;
            std::vector<hnswlib::labeltype> gt_empty;
            const std::vector<hnswlib::labeltype>* gt_row = nullptr;
            if (need_gt) {
                gt_row = FetchGroundTruth(state, row, gt_tmp);
                if (!gt_row) {
                    if (state.cfg && state.cfg->logger) {
                        state.cfg->logger->error(
                            "Ground truth not available for row {}", row);
                    }
                    return;
                }
            }
            int cur_k = need_gt
                ? std::min<int>(gt_row->size(), state.pc->topk)
                : state.pc->topk;
            if (cur_k <= 0) {
                return;
            }
            std::unique_ptr<QueryLatencyScope> query_scope;
            if (tracker_ptr) {
                query_scope = std::make_unique<QueryLatencyScope>(*tracker_ptr, row);
                tracker_ptr->RecordQueryStart(row);
            }
            auto query_hnsw_start = Clock::now();
            ExperimentQueryResult res =
                strategy.ExecuteSearch(
                    state,
                    row,
                    threadId,
                    cur_k,
                    need_gt ? *gt_row : gt_empty);
            auto query_hnsw_end = Clock::now();
            double query_hnsw_cost = std::chrono::duration_cast<
                std::chrono::microseconds>(query_hnsw_end - query_hnsw_start)
                                            .count();
            const bool need_offsets = !strategy.HandlesIO();
            std::vector<unsigned long long> indexs;
            if (compute_regular_recall) {
                indexs.reserve(res.result_ids.size());
            }

            for (auto id : res.result_ids) {
                if (compute_regular_recall) {
                    indexs.push_back(id);
                }
            }
            double io_cost = 0.0;
            if (strategy.HandlesIO()) {
                io_cost = res.io_time_us < 0.0 ? 0.0 : res.io_time_us;
            } else {
                // OffsetIO 统一走 mapped_read_sync 路径（BuildReqPlanU64）
                auto io_start = Clock::now();
                std::size_t missing = 0;
                std::size_t issued_io = 0;
                IoWorkerForThread(state, threadId)
                    .mapped_read_sync(
                        res.result_ids,
                        *state.pc,
                        static_cast<int>(row),
                        &missing,
                        &issued_io);
                auto io_end = Clock::now();
                io_cost = std::chrono::duration_cast<
                    std::chrono::microseconds>(io_end - io_start)
                                     .count();
            }
            double query_io_cost = query_hnsw_cost + io_cost;
            thread_query_io_us[threadId] += query_io_cost;

            double recall = 0.0;
            if ((compute_regular_recall || compute_overlap_recall) && gt_row) {
                if (compute_overlap_recall) {
                    recall = (res.overlap && !res.overlap->issued_label_set.empty())
                        ? annfs::common::overlap_prefetch_recall_at_k(
                              res.overlap->issued_label_set, *gt_row, cur_k)
                        : 0.0;
                } else if (compute_regular_recall) {
                    recall = annfs::common::get_recall<unsigned long long,
                                                       hnswlib::labeltype>(
                        indexs, *gt_row, cur_k);
                }
            }

            costs[0][row] = res.hnsw_time_us;
            costs[1][row] = io_cost;
            costs[2][row] = res.hnsw_time_us + io_cost;
            recalls[0][row] = recall;

            if (strategy.RequiresIterDist()) {
                iter_dist_counts[0][row] = res.iter_count;
                iter_dist_counts[1][row] = res.dist_count;
                if (strategy.ReportsFoundGtFlag()) {
                    iter_dist_counts[2][row] = res.found_all_gt ? 1.0 : 0.0;
                }
            }
            const size_t done =
                completed_queries.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done == query_cnt || done % 1000 == 0) {
                if (state.cfg && state.cfg->logger) {
                    state.cfg->logger->debug(
                        "[RUN] query_progress run_index={} completed={}/{}",
                        run_index,
                        done,
                        query_cnt);
                }
            }
        });
    
    
    double post_query_wait_us = 0.0;
    // 如果是异步非阻塞策略，则需要在全部 query 结束后统一 wait。
    if (strategy.RequiresPostQueryIoWait()) {
        auto post_query_wait_start = Clock::now();
        for (auto& ioer : state.ioers) {
            if (ioer) {
                ioer->wait_all_completions_all();
            }
        }
        auto post_query_wait_end = Clock::now();
        post_query_wait_us = std::chrono::duration_cast<
            std::chrono::microseconds>(
                post_query_wait_end - post_query_wait_start).count();
    }

    auto allend = Clock::now();

    std::cout << "[RUN] query_phase_done"
              << " run_index=" << run_index
              << " completed="
              << completed_queries.load(std::memory_order_relaxed)
              << "/" << query_cnt
              << "\n"
              << std::flush;
    emit_memory_snapshot("[RUN_MEM] query_phase_done run_index=" +
                         std::to_string(run_index));
                         
    std::cout << "[RUN] post_query_wait_done"
                << " run_index=" << run_index
                << " wall_us=" << post_query_wait_us
                << "\n"
                << std::flush;

    // 如果使用marker 则需要在最后面标记marker
    if (query_phase_active) {
        auto query_phase_end_tp = std::chrono::steady_clock::now();
        auto query_phase_wall = std::chrono::duration_cast<
            std::chrono::microseconds>(
                query_phase_end_tp - query_phase_start_tp).count();
        nlohmann::json end_marker = {
            {"event", "phase_end"},
            {"phase", "query"},
            {"dataset", state.args.dataset_name},
            {"repeat_id", state.args.repeat_id},
            {"num_threads", state.num_threads},
            {"run_index", run_index},
            {"ts_ms", epoch_ms()},
            {"wall_time_us", query_phase_wall}
        };
        emit_batch_marker(marker_path, end_marker);
    }
    const std::string trace_tid_csv = join_tid_csv(monitor_tids);
    if (!trace_tid_csv.empty()) {
        ::setenv("ANNFS_INTERNAL_TRACE_MM_PAGECACHE_FILTER_TIDS",
                 trace_tid_csv.c_str(), 1);
    }
    for (auto it = monitors.rbegin(); it != monitors.rend(); ++it) {
        if (!(*it)->Enabled()) {
            continue;
        }
        std::cout << "[RUN] monitor_stop"
                  << " run_index=" << run_index
                  << " name=" << (*it)->Name()
                  << "\n"
                  << std::flush;
        if (!(*it)->Stop()) {
            if (state.cfg && state.cfg->logger) {
                state.cfg->logger->error(
                    "Internal monitor stop failed: {}", (*it)->Name());
            }
            return 3;
        }
        std::cout << "[RUN] monitor_stopped"
                  << " run_index=" << run_index
                  << " name=" << (*it)->Name()
                  << "\n"
                  << std::flush;
    }

    double wall_time_us = std::chrono::duration_cast<
        std::chrono::microseconds>(allend - allstart)
                              .count();


    const HnswIoCliArgs& args = state.args;
    std::string out_dir;
    if (strategy.HasCustomOutputDir()) {
        out_dir = strategy.CustomOutputDir(state, args);
    } else {
        out_dir = BuildOutputDir(state, strategy.ResultFolderName());
    }
    std::string mode_dir = strategy.ModeSubdir(state);
    if (!mode_dir.empty()) {
        out_dir += "/" + mode_dir;
    }
    std::filesystem::create_directories(out_dir);

    annfs::common::generate_json_multi_T<double>(
        costs,
        {"hnsw", "io", "hnswio"},
        state.query_cnt,
        out_dir + "/HNSWIO.json");
    if (compute_regular_recall || compute_overlap_recall) {
        annfs::common::generate_json_multi_T<double>(
            recalls,
            {"recall"},
            state.query_cnt,
            out_dir + "/HNSWIO_Recall.json");
    }
    if (strategy.RequiresIterDist()) {
        std::vector<std::string> labels = {"iter_count", "dist_count"};
        if (strategy.ReportsFoundGtFlag()) {
            labels.push_back("found_gt");
        }
        annfs::common::generate_json_multi_T<double>(
            iter_dist_counts,
            labels,
            state.query_cnt,
            out_dir + "/HNSWIO_IterDistCounts.json");
    }

    if (state.cfg && state.cfg->logger) {
        state.cfg->logger->info(
            "{} total latency (wall): {} us",
            strategy.Name(),
            wall_time_us);
    }
    nlohmann::json timing = {
        {"total_latency_us", wall_time_us},
        {"post_query_wait_us", post_query_wait_us},
        {"thread_query_io_us", thread_query_io_us}
    };
    for (const auto& m : monitors) {
        m->EmitMeta(timing);
    }
    std::ofstream timing_out(out_dir + "/HNSWIO_TotalTime.json");
    if (timing_out.is_open()) {
        timing_out << timing.dump(4);
        if (state.cfg && state.cfg->logger) {
            state.cfg->logger->info(
                "Persisted HNSWIO_TotalTime.json to {}",
                out_dir);
        }
    }
    // 持久化 query track的时间记录 --- 暂时没有用上 之后再检查
    if (query_tracker) {
        PersistQueryTimeline(out_dir, *query_tracker, state.cfg);
    }
    strategy.OnAfterRun(state, out_dir);
    const auto execute_elapsed_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - execute_start_tp).count();
    std::cout << "[RUN] done"
              << " run_index=" << run_index
              << " rc=0"
              << " elapsed_ms=" << execute_elapsed_ms
              << " out_dir=" << out_dir
              << "\n"
              << std::flush;
    return 0;
}

int RunHnswExperiment(const HnswIoCliArgs& args,
                      ExperimentStrategy& strategy)
{
    const char* batch_plan_path = std::getenv("ANNFS_EXPERIMENT_BATCH_PLAN");
    if (batch_plan_path && *batch_plan_path) {
        const ExperimentBatchPlan plan = LoadExperimentBatchPlan(batch_plan_path);
        std::cout << "[BATCH] plan=" << batch_plan_path
                  << " runs=" << plan.runs.size()
                  << " persist_state=" << (plan.persist_state ? "true" : "false")
                  << " reuse_dataset_index="
                  << (plan.reuse_dataset_index_across_thread_runs ? "true" : "false")
                  << " drop_cache_between_runs=" << (plan.drop_cache_between_runs ? "true" : "false")
                  << "\n";
        return RunHnswExperimentBatch(args.config_json_path, plan, strategy);
    }

    /**
        支持CLI参数传递以及具体的执行策略传递；
    */
    // 1. 先构建状态：
    // 实验前的准备工作（PrepareState），包括加载数据集、构建索引、配置IO等；
    HnswIoState state = PrepareState(args);
    // 获取 iostat的 marker 写入的文件路径
    const std::string marker_path = batch_marker_path();
    const bool emit_run_markers = !marker_path.empty() &&
                                  !HasInternalMonitorSelectionEnv();
    const int run_index = ResolveRunIndex(state.args);
    std::chrono::steady_clock::time_point run_start_tp;
    if (emit_run_markers) {
        // Optional external gate: monitoring side can create this file only
        // after blktrace is fully started, avoiding marker/race windows.
        wait_for_marker_ready_gate_if_needed();
        // 根据环境变量决定是否输出批处理marker，并在适当的时机输出开始和结束的marker；
        run_start_tp = std::chrono::steady_clock::now();
        nlohmann::json start_marker = {
            {"event", "start"},
            {"dataset", state.args.dataset_name},
            {"repeat_id", state.args.repeat_id},
            {"num_threads", state.num_threads},
            {"run_index", run_index},
            {"ts_ms", epoch_ms()}
        };
        emit_batch_marker(marker_path, start_marker);
    }
    // 通过ParallelForQueries并行执行查询，每个查询执行策略指定的搜索逻辑，并记录HNSW时间、IO时间、召回率等指标；
    int rc = ExecuteHnswExperimentWithState(state, strategy);

    if (emit_run_markers) {
        auto run_end_tp = std::chrono::steady_clock::now();
        auto wall_time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(run_end_tp - run_start_tp).count();
        nlohmann::json end_marker = {
            {"event", "end"},
            {"dataset", state.args.dataset_name},
            {"repeat_id", state.args.repeat_id},
            {"num_threads", state.num_threads},
            {"run_index", run_index},
            {"ts_ms", epoch_ms()},
            {"wall_time_us", wall_time_us}
        };
        emit_batch_marker(marker_path, end_marker);
    }

    return rc;
}

} // namespace annfs::tests
