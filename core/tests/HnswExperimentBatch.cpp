#include "tests/HnswExperimentBatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "common/json_utils.hpp"
#include "tests/HnswExperimentRunner.hpp"

namespace annfs::tests {
namespace {

using SteadyClock = std::chrono::steady_clock;

long long elapsed_ms_since(const SteadyClock::time_point& start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        SteadyClock::now() - start).count();
}

bool bool_or_default(const nlohmann::json& obj, const char* key, bool fallback)
{
    if (!obj.contains(key)) {
        return fallback;
    }
    try {
        return obj.at(key).get<bool>();
    } catch (...) {
        return fallback;
    }
}

std::optional<int> opt_int(const nlohmann::json& obj, const char* key)
{
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return std::nullopt;
    }
    try {
        return obj.at(key).get<int>();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> opt_string(const nlohmann::json& obj, const char* key)
{
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return std::nullopt;
    }
    try {
        const std::string v = obj.at(key).get<std::string>();
        if (v.empty()) {
            return std::nullopt;
        }
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> opt_double(const nlohmann::json& obj, const char* key)
{
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return std::nullopt;
    }
    try {
        return obj.at(key).get<double>();
    } catch (...) {
        return std::nullopt;
    }
}

bool drop_page_cache_best_effort()
{
    std::ofstream out("/proc/sys/vm/drop_caches", std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out << "3\n";
    out.flush();
    return static_cast<bool>(out);
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

HnswIoCliArgs BuildRunArgs(const std::string& config_path,
                           const ExperimentBatchRunSpec& spec)
{
    HnswIoCliArgs args;
    args.config_json_path = config_path;
    args.dataset_name = spec.dataset_name;
    args.repeat_id = spec.repeat_id > 0 ? spec.repeat_id : 1;
    args.enable_query_timeline = true;
    if (spec.num_threads) {
        args.override_num_threads = *spec.num_threads;
    }
    if (spec.search_ef) {
        args.override_search_ef = *spec.search_ef;
    }
    if (spec.stable_iter) {
        args.override_stable_iter = *spec.stable_iter;
    }
    return args;
}

void ApplyRunSpecToPreparedState(HnswIoState& state,
                                 const ExperimentBatchRunSpec& spec)
{
    const int prev_threads = state.num_threads;
    const int prev_ring_depth = GlobalConfig::instance().RING_DEPTH;
    const bool prev_use_shared_worker = state.use_shared_io_worker;
    const size_t prev_ioer_count = state.ioers.size();
    state.args.repeat_id = spec.repeat_id > 0 ? spec.repeat_id : 1;
    if (state.pc) {
        if (spec.num_threads) {
            state.pc->num_threads = *spec.num_threads;
        }
        if (spec.search_ef) {
            state.pc->dataset.search_ef = *spec.search_ef;
        }
        if (spec.io_depths) {
            GlobalConfig::instance().RING_DEPTH = *spec.io_depths;
        }
        if (spec.query_cnt) {
            state.pc->query_cnt = *spec.query_cnt;
        }
        if (spec.stable_iter) {
            state.pc->dataset.stable_iter = *spec.stable_iter;
        }
        if (spec.adaptive_stop_target_phases) {
            state.pc->adaptive_cfg.stop_eval_target_consecutive_phases =
                *spec.adaptive_stop_target_phases;
        }
        if (spec.adaptive_stop_on_conflict_phases) {
            state.pc->adaptive_cfg.stop_eval_on_conflict_phases =
                *spec.adaptive_stop_on_conflict_phases;
        }
        if (spec.no_prefetch_ratio_high) {
            state.pc->adaptive_cfg.no_prefetch_ratio_high = *spec.no_prefetch_ratio_high;
        }
        if (spec.io_issue_ratio_high) {
            state.pc->adaptive_cfg.io_issue_ratio_high = *spec.io_issue_ratio_high;
        }
        state.num_threads = state.pc->num_threads;
        state.query_cnt = std::min(
            state.pc->query_cnt,
            static_cast<int>(state.VectorCount()));
        if (!state.gt_preloaded.empty()) {
            state.query_cnt = std::min(
                state.query_cnt,
                static_cast<int>(state.gt_preloaded.size()));
        }
        state.pc->query_cnt = state.query_cnt;
        if (state.hnsw_index) {
            state.hnsw_index->setEf(state.pc->dataset.search_ef);
        }
    }

    const bool threads_changed = (prev_threads != state.num_threads);
    const bool ring_depth_changed =
        (prev_ring_depth != GlobalConfig::instance().RING_DEPTH);
    const bool desired_use_shared_worker =
        (GlobalConfig::instance().IO_WORKER_MODE == "shared");
    const int desired_ioer_count =
        desired_use_shared_worker ? 1 : std::max(1, state.num_threads);
    const bool worker_mode_changed =
        (prev_use_shared_worker != desired_use_shared_worker);
    const bool ioer_count_mismatch =
        static_cast<int>(state.ioers.size()) != desired_ioer_count;
    if (threads_changed || ring_depth_changed ||
        worker_mode_changed || ioer_count_mismatch) {
        std::cout << "[BATCH] io_workers_rebuild=1"
                  << " reason="
                  << (threads_changed ? "threads_changed;" : "")
                  << (ring_depth_changed ? "ring_depth_changed;" : "")
                  << (worker_mode_changed ? "worker_mode_changed;" : "")
                  << (ioer_count_mismatch ? "ioer_count_mismatch;" : "")
                  << " old_threads=" << prev_threads
                  << " new_threads=" << state.num_threads
                  << " old_ring_depth=" << prev_ring_depth
                  << " new_ring_depth=" << GlobalConfig::instance().RING_DEPTH
                  << " old_use_shared=" << (prev_use_shared_worker ? "true" : "false")
                  << " new_use_shared=" << (desired_use_shared_worker ? "true" : "false")
                  << " old_ioers=" << prev_ioer_count
                  << " new_ioers=" << desired_ioer_count
                  << "\n";

        for (auto& ioer : state.ioers) {
            if (ioer) {
                ioer->wait_all_completions_all();
            }
        }
        std::vector<std::unique_ptr<IOuringManager>>().swap(state.ioers);

        state.use_shared_io_worker = desired_use_shared_worker;
        for (int i = 0; i < desired_ioer_count; ++i) {
            state.ioers.emplace_back(std::make_unique<IOuringManager>());
        }
    }
}

} // namespace

ExperimentBatchPlan LoadExperimentBatchPlan(const std::string& path)
{
    ExperimentBatchPlan plan;
    const auto root = annfs::common::read_json(path);
    plan.persist_state = bool_or_default(root, "persist_state", true);
    plan.drop_cache_between_runs = bool_or_default(root, "drop_cache_between_runs", true);
    plan.reuse_dataset_index_across_thread_runs = bool_or_default(
        root,
        "reuse_dataset_index_across_thread_runs",
        true);
    if (!root.contains("runs") || !root.at("runs").is_array()) {
        return plan;
    }
    for (const auto& item : root.at("runs")) {
        if (!item.is_object()) {
            continue;
        }
        ExperimentBatchRunSpec spec;
        spec.dataset_name = item.value("dataset_name", "");
        spec.repeat_id = item.value("repeat_id", 1);
        spec.num_threads = opt_int(item, "num_threads");
        spec.search_ef = opt_int(item, "search_ef");
        spec.io_depths = opt_int(item, "io_depths");
        spec.query_cnt = opt_int(item, "query_cnt");
        spec.iter_json_path = opt_string(item, "iter_json_path");
        spec.monitors = opt_string(item, "monitors");
        spec.stable_iter = opt_int(item, "stable_iter");
        spec.adaptive_stop_target_phases = opt_int(item, "adaptive_stop_target_phases");
        spec.adaptive_stop_on_conflict_phases = opt_int(item, "adaptive_stop_on_conflict_phases");
        spec.no_prefetch_ratio_high = opt_double(item, "no_prefetch_ratio_high");
        spec.io_issue_ratio_high = opt_double(item, "io_issue_ratio_high");
        spec.drop_cache = bool_or_default(item, "drop_cache", false);
        if (!spec.dataset_name.empty()) {
            plan.runs.push_back(std::move(spec));
        }
    }
    return plan;
}

int RunHnswExperimentBatch(const std::string& config_path,
                           const ExperimentBatchPlan& plan,
                           ExperimentStrategy& strategy)
{
    if (plan.runs.empty()) {
        std::cerr << "[BATCH] no runs in plan\n";
        return 1;
    }

    int first_error = 0;
    bool has_state = false;
    std::string cached_dataset;
    std::optional<int> cached_threads;
    HnswIoState state;
    const char* kMonitorsEnv = "ANNFS_INTERNAL_MONITORS";
    const char* kRunIndexEnv = "ANNFS_RUN_INDEX_HINT";
    const char* prev_monitors = std::getenv(kMonitorsEnv);
    const char* prev_run_index = std::getenv(kRunIndexEnv);
    const std::optional<std::string> prev_monitors_opt =
        prev_monitors ? std::make_optional(std::string(prev_monitors))
                      : std::nullopt;
    const std::optional<std::string> prev_run_index_opt =
        prev_run_index ? std::make_optional(std::string(prev_run_index))
                       : std::nullopt;

    const auto must_prepare_state =
        [&](const ExperimentBatchRunSpec& spec) -> bool {
            if (!has_state) {
                return true;
            }
            if (!plan.persist_state) {
                return true;
            }
            if (cached_dataset != spec.dataset_name) {
                return true;
            }
            if (!plan.reuse_dataset_index_across_thread_runs &&
                cached_threads != spec.num_threads) {
                return true;
            }
            return false;
        };

    for (size_t i = 0; i < plan.runs.size(); ++i) {
        const auto& spec = plan.runs[i];
        const auto run_start = SteadyClock::now();
        ::setenv(kRunIndexEnv, std::to_string(i).c_str(), 1);
        const bool need_drop_cache =
            spec.drop_cache || plan.drop_cache_between_runs;
        if (need_drop_cache) {
            std::cout << "[BATCH] run_index=" << i
                      << " dropping page cache before run\n"
                      << std::flush;
            const int sync_rc = std::system("sync");
            (void)sync_rc;
            const bool dropped = drop_page_cache_best_effort();
            if (!dropped) {
                std::cerr << "[BATCH] warn: drop_caches failed (need root): run="
                          << i << " dataset=" << spec.dataset_name << "\n"
                          << std::flush;
            }
            std::cout << "[BATCH] run_index=" << i
                      << " drop_cache_done=" << (dropped ? "true" : "false")
                      << " elapsed_ms=" << elapsed_ms_since(run_start)
                      << "\n"
                      << std::flush;
        }

        if (spec.monitors) {
            if (!spec.monitors->empty()) {
                ::setenv(kMonitorsEnv, spec.monitors->c_str(), 1);
            } else {
                ::unsetenv(kMonitorsEnv);
            }
        } else if (prev_monitors_opt) {
            ::setenv(kMonitorsEnv, prev_monitors_opt->c_str(), 1);
        } else {
            ::unsetenv(kMonitorsEnv);
        }

        std::cout << "[BATCH] run_index=" << i
                  << " dataset=" << spec.dataset_name
                  << " repeat_id=" << spec.repeat_id
                  << " reuse_dataset_index="
                  << (plan.reuse_dataset_index_across_thread_runs ? "true" : "false")
                  << " monitors="
                  << (::getenv(kMonitorsEnv) ? ::getenv(kMonitorsEnv) : "")
                  << " threads=" << (spec.num_threads ? std::to_string(*spec.num_threads) : "")
                  << " no_prefetch_ratio_high="
                  << (spec.no_prefetch_ratio_high ? std::to_string(*spec.no_prefetch_ratio_high) : "")
                  << " io_issue_ratio_high="
                  << (spec.io_issue_ratio_high ? std::to_string(*spec.io_issue_ratio_high) : "")
                  << " stable_iter="
                  << (spec.stable_iter ? std::to_string(*spec.stable_iter) : "")
                  << " adaptive_stop_target_phases="
                  << (spec.adaptive_stop_target_phases ? std::to_string(*spec.adaptive_stop_target_phases) : "")
                  << " adaptive_stop_on_conflict_phases="
                  << (spec.adaptive_stop_on_conflict_phases ? std::to_string(*spec.adaptive_stop_on_conflict_phases) : "")
                  << " query_cnt="
                  << (spec.query_cnt ? std::to_string(*spec.query_cnt) : "")
                  << " io_depths="
                  << (spec.io_depths ? std::to_string(*spec.io_depths) : "")
                  << "\n"
                  << std::flush;
        emit_memory_snapshot("[BATCH_MEM] before_prepare run_index=" +
                             std::to_string(i));

        if (must_prepare_state(spec)) {
            const auto prepare_start = SteadyClock::now();
            std::cout << "[BATCH] run_index=" << i
                      << " rebuilding_state=1"
                      << " reason(dataset/run_unit/persist_state changed)"
                      << "\n"
                      << std::flush;
            // Release previous large state (vectors/index/GT/IO workers)
            // before building a new one to avoid transient memory spikes.
            if (has_state) {
                state = HnswIoState {};
                std::cout << "[BATCH] run_index=" << i
                          << " previous_state_released=1\n"
                          << std::flush;
            }
            HnswIoCliArgs args = BuildRunArgs(config_path, spec);
            state = PrepareState(args);
            std::cout << "[BATCH] run_index=" << i
                      << " prepare_state_done=1"
                      << " elapsed_ms=" << elapsed_ms_since(prepare_start)
                      << "\n"
                      << std::flush;
            emit_memory_snapshot("[BATCH_MEM] after_prepare run_index=" +
                                 std::to_string(i));
            has_state = true;
            cached_dataset = spec.dataset_name;
            cached_threads = spec.num_threads;
        } else {
            std::cout << "[BATCH] run_index=" << i
                      << " reusing_state=1"
                      << " dataset=" << cached_dataset
                      << " reuse_dataset_index="
                      << (plan.reuse_dataset_index_across_thread_runs ? "true" : "false")
                      << "\n"
                      << std::flush;
        }
        ApplyRunSpecToPreparedState(state, spec);

        strategy.ApplyBatchRunSpec(spec);
        std::cout << "[BATCH] run_index=" << i
                  << " execute_start=1"
                  << " elapsed_ms=" << elapsed_ms_since(run_start)
                  << "\n"
                  << std::flush;
        emit_memory_snapshot("[BATCH_MEM] before_execute run_index=" +
                             std::to_string(i));
        int rc = ExecuteHnswExperimentWithState(state, strategy);
        std::cout << "[BATCH] run_index=" << i
                  << " completed rc=" << rc
                  << " elapsed_ms=" << elapsed_ms_since(run_start)
                  << "\n"
                  << std::flush;
        emit_memory_snapshot("[BATCH_MEM] after_execute run_index=" +
                             std::to_string(i));
        if (rc != 0 && first_error == 0) {
            first_error = rc;
        }
    }
    if (prev_monitors_opt) {
        ::setenv(kMonitorsEnv, prev_monitors_opt->c_str(), 1);
    } else {
        ::unsetenv(kMonitorsEnv);
    }
    if (prev_run_index_opt) {
        ::setenv(kRunIndexEnv, prev_run_index_opt->c_str(), 1);
    } else {
        ::unsetenv(kRunIndexEnv);
    }
    state = HnswIoState {};
    std::cout << "[BATCH] final_state_released=1\n" << std::flush;
    return first_error;
}

} // namespace annfs::tests
