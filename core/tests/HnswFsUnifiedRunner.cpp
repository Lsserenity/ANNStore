#include "tests/HnswFsUnifiedRunner.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include "common/parallel.hpp"
#include "common/json_utils.hpp"
#include "recall.hpp"
#include "tests/HnswInternalMonitors.hpp"
#include "tests/HnswFsRunnerShared.hpp"
#include "tests/QueryLatencyTracker.hpp"

namespace fs = std::filesystem;

namespace annfs::tests {
namespace {

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

struct FlopsRule {
    double flops_per_distance = 0.0;
    double int_ops_per_distance = 0.0;
    std::string rule;
};

std::string lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return s;
}

std::string sanitize_path_token(std::string value)
{
    if (value.empty()) {
        return "unknown";
    }
    for (char& c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!(std::isalnum(uc) || c == '-' || c == '_' || c == '.')) {
            c = '_';
        }
    }
    return value;
}

std::string env_or_default(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return value;
}

std::string utc_run_stamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%dT%H%M%SZ");
    return oss.str();
}

FlopsRule ResolveFlopsRule(const FsDatasetResources& resources,
                           const ParamConfig& pc)
{
    FlopsRule rule;
    const int dim = std::max(0, resources.dim);
    const std::string metric = lower_copy(pc.dataset.distance_metric);

    if (resources.use_uint8_data) {
        rule.flops_per_distance = 0.0;
        rule.int_ops_per_distance = static_cast<double>(3LL * dim);
        rule.rule = "uint8_l2_integer_ops_only";
        return rule;
    }

    if (metric == "ip") {
        rule.flops_per_distance = static_cast<double>(2LL * dim + 1LL);
        rule.rule = "float32_ip_2dim_plus_1";
        return rule;
    }

    rule.flops_per_distance = static_cast<double>(3LL * dim);
    rule.rule = "float32_l2_3dim";
    return rule;
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

int RunFsUnifiedWithResources(FsDatasetResources& resources,
                              const HnswIoCliArgs& args,
                              const ExperimentBatchRunSpec* spec,
                              const nlohmann::json& root,
                              IoExecutionPolicy& policy,
                              const FsUnifiedRunOptions& options)
{
    (void)spec;
    auto& pc = ParamConfig::instance();
    auto& cfg = GlobalConfig::instance();

    const int num_threads = pc.num_threads;
    const bool skip_recall = pc.skip_recall;
    int query_cnt = std::min(pc.query_cnt,
                             static_cast<int>(resources.vector_count));
    if (!skip_recall &&
        resources.gt_accessor &&
        resources.gt_accessor->available_rows > 0 &&
        resources.gt_accessor->available_rows < static_cast<size_t>(query_cnt)) {
        query_cnt = static_cast<int>(resources.gt_accessor->available_rows);
    }
    pc.query_cnt = query_cnt;
    resources.hnsw_index->setEf(pc.dataset.search_ef);

    if (!policy.Init(pc, cfg, num_threads, &root)) {
        return 1;
    }
    emit_memory_snapshot("[FS_MEM] after_policy_init dataset=" + args.dataset_name +
                         " repeat_id=" + std::to_string(args.repeat_id));

    std::unique_ptr<QueryLatencyTracker> query_tracker;
    QueryLatencyTracker* tracker_ptr = nullptr;
    if (args.enable_query_timeline) {
        query_tracker = std::make_unique<QueryLatencyTracker>(query_cnt);
        tracker_ptr = query_tracker.get();
    }
    policy.SetQueryListener(tracker_ptr);

    const auto& path_mapper = *(resources.path_mapper);
    std::vector<std::vector<double>> costs(
        3, std::vector<double>(query_cnt, 0.0));
    std::vector<std::vector<double>> recalls;
    if (skip_recall && cfg.logger) {
        cfg.logger->info("Recall computation disabled (skip_recall=true)");
    }
    if (!skip_recall) {
        recalls.assign(1, std::vector<double>(query_cnt, 0.0));
    }

    auto parallel_executor = [&](auto&& fn) {
        switch (pc.parallel_mode) {
            case 0:
                OptimizedParallelFor(0, query_cnt, num_threads,
                                     std::forward<decltype(fn)>(fn));
                break;
            case 1:
                WorkStealingParallelFor(0, query_cnt, num_threads,
                                        std::forward<decltype(fn)>(fn));
                break;
            case 2:
            default:
                StaticPartitionParallelFor(0, query_cnt, num_threads,
                                           std::forward<decltype(fn)>(fn));
                break;
        }
    };

    const long long dist_before = static_cast<long long>(
        resources.hnsw_index->metric_distance_computations.load(
            std::memory_order_relaxed));

    const std::string marker_path = fsrunner::batch_marker_path();
    const int run_index = ResolveRunIndex(args);
    ::setenv("ANNFS_INTERNAL_REPORT_DATASET", args.dataset_name.c_str(), 1);
    ::setenv("ANNFS_INTERNAL_REPORT_THREADS",
             std::to_string(pc.num_threads).c_str(), 1);
    ::setenv("ANNFS_INTERNAL_REPORT_REPEAT_ID",
             std::to_string(args.repeat_id).c_str(), 1);
    ::setenv("ANNFS_INTERNAL_REPORT_RUN_INDEX",
             std::to_string(run_index).c_str(), 1);
    ::setenv("ANNFS_INTERNAL_TARGET_PID",
             std::to_string(static_cast<long long>(::getpid())).c_str(), 1);
    ::unsetenv("ANNFS_INTERNAL_TRACE_MM_PAGECACHE_FILTER_TIDS");
    std::chrono::steady_clock::time_point query_phase_start_tp;
    bool query_phase_active = false;

    auto monitors = BuildInternalMonitors(args, pc, policy.ResultFolder());
    const bool monitors_enabled = !monitors.empty();
    const bool emit_query_markers = !marker_path.empty() && monitors_enabled;

    for (const auto& m : monitors) {
        if (!m->Enabled()) {
            continue;
        }
        if (cfg.logger) {
            cfg.logger->debug("[FS_RUN] monitor_start run_index={} name={}",
                              run_index,
                              m->Name());
        }
        if (!m->Start()) {
            if (cfg.logger) {
                cfg.logger->error("Internal monitor start failed: {}", m->Name());
            }
            policy.SetQueryListener(nullptr);
            return 2;
        }
        if (cfg.logger) {
            cfg.logger->debug("[FS_RUN] monitor_started run_index={} name={}",
                              run_index,
                              m->Name());
        }
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
        if (cfg.logger) {
            cfg.logger->info("Internal monitor warmup waited: {} ms",
                             monitor_warmup_ms);
        }
    }
    if (emit_query_markers) {
        query_phase_active = true;
        query_phase_start_tp = std::chrono::steady_clock::now();
        nlohmann::json start_marker = {
            {"event", "phase_start"},
            {"phase", "query"},
            {"dataset", args.dataset_name},
            {"repeat_id", args.repeat_id},
            {"num_threads", pc.num_threads},
            {"run_index", run_index},
            {"ts_ms", fsrunner::epoch_ms()}
        };
        fsrunner::emit_batch_marker(marker_path, start_marker);
    }

    auto allstart = std::chrono::high_resolution_clock::now();
    std::unordered_set<long> monitor_tids;
    std::mutex monitor_tids_mu;
    monitor_tids.insert(current_tid());
    std::atomic<size_t> completed_queries{0};
    if (cfg.logger) {
        cfg.logger->debug(
            "[FS_RUN] query_phase_start run_index={} monitors={} query_cnt={} threads={}",
            run_index,
            monitors_enabled ? "true" : "false",
            query_cnt,
            num_threads);
    }
    emit_memory_snapshot("[FS_MEM] query_phase_start run_index=" +
                         std::to_string(run_index));
    parallel_executor([&](size_t row, size_t threadId) {
        {
            const long tid = current_tid();
            std::lock_guard<std::mutex> lk(monitor_tids_mu);
            monitor_tids.insert(tid);
        }
        std::unique_ptr<QueryLatencyScope> query_scope;
        if (tracker_ptr) {
            query_scope = std::make_unique<QueryLatencyScope>(
                *tracker_ptr, row);
            tracker_ptr->RecordQueryStart(row);
        }

        std::vector<int64_t> gt_top_k_indices;
        int cur_k = pc.topk;
        if (!skip_recall && resources.gt_accessor) {
            gt_top_k_indices = resources.gt_accessor->fetch(row);
            cur_k = std::min(static_cast<int>(gt_top_k_indices.size()),
                             pc.topk);
        }

        auto hnswst = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> hnsw_result;
        if (options.enable_flops_count) {
            hnsw_result = resources.hnsw_index->searchKnnCollectMetrics(
                const_cast<void*>(resources.QueryPtr(row)),
                cur_k);
        } else {
            hnsw_result = resources.hnsw_index->searchKnn(
                const_cast<void*>(resources.QueryPtr(row)),
                cur_k);
        }
        auto hnswed = std::chrono::high_resolution_clock::now();
        auto hnswcst = std::chrono::duration_cast<std::chrono::microseconds>(
            hnswed - hnswst).count();

        std::vector<unsigned long long> result_ids;
        while (!hnsw_result.empty()) {
            auto tmp = hnsw_result.top().second;
            hnsw_result.pop();
            result_ids.push_back(static_cast<unsigned long long>(tmp));
        }

        if (!skip_recall && resources.gt_accessor) {
            recalls[0][row] = annfs::common::get_recall<unsigned long long, int64_t>(
                result_ids,
                gt_top_k_indices,
                cur_k);
        }

        auto iost = std::chrono::high_resolution_clock::now();
        std::vector<std::string> path_list;
        path_list.reserve(result_ids.size());
        size_t skipped = 0;
        for (auto id : result_ids) {
            auto p = path_mapper.get_path(static_cast<size_t>(id));
            if (p.empty()) {
                ++skipped;
                continue;
            }
            path_list.emplace_back(std::move(p));
        }
        if (skipped > 0 && cfg.logger) {
            cfg.logger->warn(
                "Skipped {} invalid indices (fs_mapper_max={} fs_limit={} fs_round_size={})",
                skipped,
                resources.fs_mapper_max_elements,
                resources.fs_mapper_file_limit,
                resources.fs_round_size);
        }
        policy.NotifySkipped(skipped);

        policy.Execute(path_list, static_cast<int>(row), threadId);
        auto ioed = std::chrono::high_resolution_clock::now();
        auto iocst = std::chrono::duration_cast<std::chrono::microseconds>(
            ioed - iost).count();

        costs[0][row] = hnswcst;
        costs[1][row] = iocst;
        costs[2][row] = hnswcst + iocst;
        const size_t done = completed_queries.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done == query_cnt || done % 1000 == 0) {
            if (cfg.logger) {
                cfg.logger->debug(
                    "[FS_RUN] query_progress run_index={} completed={}/{}",
                    run_index,
                    done,
                    query_cnt);
            }
        }
    });

    auto finalize_wait_start = std::chrono::high_resolution_clock::now();
    policy.Finalize();
    auto finalize_wait_end = std::chrono::high_resolution_clock::now();
    const double finalize_wait_us = std::chrono::duration_cast<
        std::chrono::microseconds>(
        finalize_wait_end - finalize_wait_start).count();

    auto allend = std::chrono::high_resolution_clock::now();
    auto allcst = std::chrono::duration_cast<std::chrono::microseconds>(
        allend - allstart).count();
    if (cfg.logger) {
        cfg.logger->debug(
            "[FS_RUN] query_phase_done run_index={} completed={}/{} wall_us={}",
            run_index,
            completed_queries.load(std::memory_order_relaxed),
            query_cnt,
            allcst);
    }
    emit_memory_snapshot("[FS_MEM] query_phase_done run_index=" +
                         std::to_string(run_index));

    if (query_phase_active) {
        auto query_phase_end_tp = std::chrono::steady_clock::now();
        auto query_phase_wall = std::chrono::duration_cast<
            std::chrono::microseconds>(query_phase_end_tp - query_phase_start_tp).count();
        nlohmann::json end_marker = {
            {"event", "phase_end"},
            {"phase", "query"},
            {"dataset", args.dataset_name},
            {"repeat_id", args.repeat_id},
            {"num_threads", pc.num_threads},
            {"run_index", run_index},
            {"ts_ms", fsrunner::epoch_ms()},
            {"wall_time_us", query_phase_wall}
        };
        fsrunner::emit_batch_marker(marker_path, end_marker);
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
        if (cfg.logger) {
            cfg.logger->debug("[FS_RUN] monitor_stop run_index={} name={}",
                              run_index,
                              (*it)->Name());
        }
        if (!(*it)->Stop()) {
            if (cfg.logger) {
                cfg.logger->error("Internal monitor stop failed: {}", (*it)->Name());
            }
            policy.SetQueryListener(nullptr);
            return 3;
        }
        if (cfg.logger) {
            cfg.logger->debug("[FS_RUN] monitor_stopped run_index={} name={}",
                              run_index,
                              (*it)->Name());
        }
    }

    const long long dist_after = static_cast<long long>(
        resources.hnsw_index->metric_distance_computations.load(
            std::memory_order_relaxed));
    const long long distance_computations_delta =
        std::max(0LL, dist_after - dist_before);

    policy.EmitSummary(pc, cfg);

    const std::string exec_name =
        env_or_default("ANNFS_EXEC_NAME", policy.ResultFolder());
    const std::string run_id =
        env_or_default("ANNFS_END_RESULTS_RUN_ID", utc_run_stamp());
    const std::string results_root =
        env_or_default("ANNFS_END_RESULTS_ROOT", "./json_results");
    const std::string variant =
        env_or_default("ANNFS_END_RESULTS_VARIANT", "");
    std::string exec_token = exec_name;
    if (!variant.empty()) {
        exec_token += "_" + variant;
    }

    fs::path out_dir = fs::path(results_root) /
                       sanitize_path_token(run_id) /
                       sanitize_path_token(exec_token);
    out_dir /= sanitize_path_token(args.dataset_name);
    out_dir /= "t" + std::to_string(pc.num_threads);
    out_dir /= "efS" + std::to_string(pc.dataset.search_ef) +
               "_qcnt" + std::to_string(pc.query_cnt) +
               "_rpt_" + std::to_string(args.repeat_id);
    fs::create_directories(out_dir);

    annfs::common::generate_json_multi_T<double>(
        costs,
        {"hnsw", "io", "hnswio"},
        pc.query_cnt,
        (out_dir / "HNSWIO.json").string());
    if (!skip_recall) {
        annfs::common::generate_json_multi_T<double>(
            recalls,
            {"recall"},
            pc.query_cnt,
            (out_dir / "HNSWIO_Recall.json").string());
    }

    if (query_tracker) {
        PersistQueryTimeline(out_dir.string(), *query_tracker, &cfg);
    }

    nlohmann::json io_meta = policy.BuildStatsMeta(pc);
    io_meta["total_wall_us"] = allcst;
    io_meta["finalize_wait_wall_us"] = finalize_wait_us;

    if (options.enable_flops_count) {
        FlopsRule flops_rule = ResolveFlopsRule(resources, pc);
        const double total_flops =
            static_cast<double>(distance_computations_delta) *
            flops_rule.flops_per_distance;
        const double total_integer_ops =
            static_cast<double>(distance_computations_delta) *
            flops_rule.int_ops_per_distance;

        if (cfg.logger) {
            cfg.logger->info(
                "FsIO FLOPs: distance_calls={} flops_per_distance={} total_flops={} rule={}",
                distance_computations_delta,
                flops_rule.flops_per_distance,
                total_flops,
                flops_rule.rule);
        }
        std::cout << "[FsIO-FLOPs] distance_calls=" << distance_computations_delta
                  << " flops_per_distance=" << flops_rule.flops_per_distance
                  << " total_flops=" << total_flops
                  << " rule=" << flops_rule.rule
                  << std::endl;
        if (flops_rule.int_ops_per_distance > 0.0) {
            std::cout << "[FsIO-FLOPs] uint8 path detected, estimated_integer_ops="
                      << total_integer_ops << std::endl;
        }

        io_meta["distance_computations"] = distance_computations_delta;
        io_meta["flops_per_distance"] = flops_rule.flops_per_distance;
        io_meta["flops_rule"] = flops_rule.rule;
        io_meta["total_flops"] = total_flops;
        io_meta["integer_ops_per_distance"] = flops_rule.int_ops_per_distance;
        io_meta["estimated_total_integer_ops"] = total_integer_ops;
    }

    for (const auto& m : monitors) {
        m->EmitMeta(io_meta);
    }

    std::ofstream io_meta_out(out_dir / policy.StatsFilename());
    if (io_meta_out.is_open()) {
        io_meta_out << io_meta.dump(2) << std::endl;
    }

    if (cfg.logger) {
        cfg.logger->info("{} total latency (wall): {} us", policy.ResultFolder(), allcst);
        cfg.logger->info("{} finalize wait wall: {} us", policy.ResultFolder(), finalize_wait_us);
    }
    std::cout << "[FsUnified] total_wall_us=" << allcst
              << " finalize_wait_us=" << finalize_wait_us
              << std::endl;

    policy.SetQueryListener(nullptr);
    return 0;
}

} // namespace annfs::tests
