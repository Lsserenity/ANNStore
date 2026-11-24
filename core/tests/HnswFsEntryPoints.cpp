#include "tests/HnswFsEntryPoints.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "common/json_utils.hpp"
#include "tests/HnswExperimentBatch.hpp"
#include "tests/HnswFsDataset.hpp"
#include "tests/HnswInternalMonitors.hpp"
#include "tests/HnswFsIoPolicies.hpp"
#include "tests/HnswFsRunnerShared.hpp"
#include "tests/HnswFsUnifiedRunner.hpp"
#include "tests/HnswIoCommon.hpp"

namespace annfs::tests {
namespace {

using namespace annfs::tests::fsrunner;

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

int RunFsSingleCommon(const HnswIoCliArgs& args,
                      const nlohmann::json& root,
                      IoExecutionPolicy& policy,
                      const FsUnifiedRunOptions& options,
                      const ExperimentBatchRunSpec* spec = nullptr)
{
    auto& pc = ParamConfig::instance();
    auto& cfg = GlobalConfig::instance();

    pc.load_from_json(root, args.dataset_name);
    cfg.init_logger(pc.is_debug);
    ApplyRunOverrides(args, spec, pc);
    std::cout << "[FS_RUN] dataset=" << args.dataset_name
              << " repeat_id=" << args.repeat_id
              << " config_json=" << args.config_json_path
              << " num_threads=" << pc.num_threads
              << " search_ef=" << pc.dataset.search_ef
              << " io_depths=" << GlobalConfig::instance().RING_DEPTH
              << " query_cnt=" << pc.query_cnt
              << " stable_iter=" << (args.override_stable_iter ? *args.override_stable_iter : -1)
              << " stable_hops=" << (args.override_stable_hops ? *args.override_stable_hops : -1)
              << "\n";
    std::cout << "[FS_RUN] loading/reusing index resources before query phase...\n";
    emit_memory_snapshot("[FS_MEM] before_load dataset=" + args.dataset_name +
                         " repeat_id=" + std::to_string(args.repeat_id));

    const std::string marker_path = batch_marker_path();
    const bool emit_standalone_markers =
        !marker_path.empty() && spec == nullptr &&
        !HasInternalMonitorSelectionEnv();
    const int run_index = ResolveRunIndex(args);
    std::chrono::steady_clock::time_point run_start_tp;

    auto& resources =
        LoadOrGetFsDataset(args.config_json_path,
                           root,
                           pc,
                           cfg);
    emit_memory_snapshot("[FS_MEM] after_load dataset=" + args.dataset_name +
                         " repeat_id=" + std::to_string(args.repeat_id));

    if (emit_standalone_markers) {
        run_start_tp = std::chrono::steady_clock::now();
        nlohmann::json start_marker = {
            {"event", "start"},
            {"dataset", args.dataset_name},
            {"repeat_id", args.repeat_id},
            {"num_threads", pc.num_threads},
            {"run_index", run_index},
            {"ts_ms", epoch_ms()}
        };
        emit_batch_marker(marker_path, start_marker);
    }

    int rc = RunFsUnifiedWithResources(resources,
                                       args,
                                       spec,
                                       root,
                                       policy,
                                       options);

    if (emit_standalone_markers) {
        auto run_end_tp = std::chrono::steady_clock::now();
        auto wall_time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(run_end_tp - run_start_tp).count();
        nlohmann::json end_marker = {
            {"event", "end"},
            {"dataset", args.dataset_name},
            {"repeat_id", args.repeat_id},
            {"num_threads", pc.num_threads},
            {"run_index", run_index},
            {"ts_ms", epoch_ms()},
            {"wall_time_us", wall_time_us}
        };
        emit_batch_marker(marker_path, end_marker);
    }

    return rc;
}

bool drop_page_cache_best_effort()
{
    const int sync_rc = std::system("sync");
    (void)sync_rc;
    std::ofstream out("/proc/sys/vm/drop_caches", std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out << "3\n";
    out.flush();
    return static_cast<bool>(out);
}

template <typename PolicyFactory>
int RunFsMaybeBatch(const HnswIoCliArgs& base_args,
                    const nlohmann::json& root,
                    const FsUnifiedRunOptions& options,
                    PolicyFactory&& make_policy)
{
    const char* batch_plan_path = std::getenv("ANNFS_EXPERIMENT_BATCH_PLAN");
    if (!batch_plan_path || std::string(batch_plan_path).empty()) {
        auto policy = make_policy();
        return RunFsSingleCommon(base_args, root, policy, options, nullptr);
    }

    const ExperimentBatchPlan plan = LoadExperimentBatchPlan(batch_plan_path);
    if (plan.runs.empty()) {
        std::cerr << "[FS_BATCH] empty runs in plan: " << batch_plan_path << "\n";
        return 1;
    }

    std::cout << "[FS_BATCH] plan=" << batch_plan_path
              << " persist_state=" << (plan.persist_state ? "true" : "false")
              << " reuse_dataset_index="
              << (plan.reuse_dataset_index_across_thread_runs ? "true" : "false")
              << " drop_cache_between_runs=" << (plan.drop_cache_between_runs ? "true" : "false")
              << " runs=" << plan.runs.size() << "\n";
    std::cout << "[FS_BATCH] NOTE: persist_state=true reuses in-process loaded index/resources; "
                 "this is valid for query-phase comparisons, not for cold-start index-load wall-time claims.\n";

    int first_error = 0;
    std::optional<std::string> previous_dataset_name;
    std::optional<int> previous_threads;
    const char* kMonitorsEnv = "ANNFS_INTERNAL_MONITORS";
    const char* kRunIndexEnv = "ANNFS_RUN_INDEX_HINT";
    const char* previous_monitors = std::getenv(kMonitorsEnv);
    const char* previous_run_index = std::getenv(kRunIndexEnv);
    const std::optional<std::string> previous_monitors_opt =
        previous_monitors ? std::make_optional(std::string(previous_monitors))
                          : std::nullopt;
    const std::optional<std::string> previous_run_index_opt =
        previous_run_index ? std::make_optional(std::string(previous_run_index))
                           : std::nullopt;

    for (size_t i = 0; i < plan.runs.size(); ++i) {
        const auto& spec = plan.runs[i];
        ::setenv(kRunIndexEnv, std::to_string(i).c_str(), 1);
        HnswIoCliArgs run_args = base_args;
        if (!spec.dataset_name.empty()) {
            run_args.dataset_name = spec.dataset_name;
        }
        if (spec.repeat_id > 0) {
            run_args.repeat_id = spec.repeat_id;
        }
        if (spec.num_threads) {
            run_args.override_num_threads = *spec.num_threads;
        }
        if (spec.search_ef) {
            run_args.override_search_ef = *spec.search_ef;
        }
        if (spec.stable_iter) {
            run_args.override_stable_iter = *spec.stable_iter;
        }

        bool clear_cache_before_run = false;
        if (!plan.persist_state) {
            clear_cache_before_run = true;
        } else if (!plan.reuse_dataset_index_across_thread_runs) {
            const int cur_threads =
                run_args.override_num_threads.value_or(ParamConfig::instance().num_threads);
            if (!previous_dataset_name || !previous_threads ||
                *previous_dataset_name != run_args.dataset_name ||
                *previous_threads != cur_threads) {
                clear_cache_before_run = previous_dataset_name.has_value();
            }
        } else if (previous_dataset_name &&
                   *previous_dataset_name != run_args.dataset_name) {
            ReleaseFsDataset(base_args.config_json_path, *previous_dataset_name);
        }
        if (clear_cache_before_run) {
            std::cout << "[FS_BATCH] run_index=" << i
                      << " clear_dataset_cache=1"
                      << " reason=run_unit_switch_or_persist_off\n";
            ClearFsDatasetCache();
        } else {
            std::cout << "[FS_BATCH] run_index=" << i
                      << " clear_dataset_cache=0"
                      << " reason=reuse_dataset_index\n";
        }

        const bool should_drop_cache =
            spec.drop_cache || plan.drop_cache_between_runs;
        if (should_drop_cache) {
            const bool dropped = drop_page_cache_best_effort();
            if (!dropped) {
                std::cout << "[FS_BATCH] warn: drop_caches failed (need root), continue run_index="
                          << i << "\n";
            }
            std::cout << "[FS_BATCH] run_index=" << i
                      << " drop_cache_done=" << (dropped ? "true" : "false")
                      << "\n";
        }

        if (spec.monitors) {
            if (!spec.monitors->empty()) {
                ::setenv(kMonitorsEnv, spec.monitors->c_str(), 1);
            } else {
                ::unsetenv(kMonitorsEnv);
            }
        } else if (previous_monitors_opt) {
            ::setenv(kMonitorsEnv, previous_monitors_opt->c_str(), 1);
        } else {
            ::unsetenv(kMonitorsEnv);
        }

        std::cout << "[FS_BATCH] run_index=" << i
                  << " dataset=" << run_args.dataset_name
                  << " repeat_id=" << run_args.repeat_id
                  << " monitors=" << (std::getenv(kMonitorsEnv) ? std::getenv(kMonitorsEnv) : "")
                  << " threads="
                  << (run_args.override_num_threads ? std::to_string(*run_args.override_num_threads) : "")
                  << " query_cnt="
                  << (spec.query_cnt ? std::to_string(*spec.query_cnt) : "")
                  << " io_depths="
                  << (spec.io_depths ? std::to_string(*spec.io_depths) : "")
                  << "\n";
        emit_memory_snapshot("[FS_MEM] before_run_index=" + std::to_string(i));

        auto policy = make_policy();
        const int rc = RunFsSingleCommon(run_args, root, policy, options, &spec);
        emit_memory_snapshot("[FS_MEM] after_run_index=" + std::to_string(i));
        if (rc != 0 && first_error == 0) {
            first_error = rc;
        }
        previous_dataset_name = run_args.dataset_name;
        previous_threads = run_args.override_num_threads.value_or(
            ParamConfig::instance().num_threads);
    }

    if (!plan.persist_state) {
        ClearFsDatasetCache();
    }
    if (previous_monitors_opt) {
        ::setenv(kMonitorsEnv, previous_monitors_opt->c_str(), 1);
    } else {
        ::unsetenv(kMonitorsEnv);
    }
    if (previous_run_index_opt) {
        ::setenv(kRunIndexEnv, previous_run_index_opt->c_str(), 1);
    } else {
        ::unsetenv(kRunIndexEnv);
    }
    return first_error;
}

} // namespace

int RunHnswFsIoMain(int argc, char** argv, bool enable_flops_count)
{
    auto args_opt = ParseCliArgs(argc, argv);
    if (!args_opt) {
        return 1;
    }
    HnswIoCliArgs args = *args_opt;
    auto root = annfs::common::read_json(args.config_json_path);

    FsUnifiedRunOptions options;
    options.enable_flops_count = enable_flops_count;
    return RunFsMaybeBatch(args, root, options, []() {
        return FsSyncIoExecutionPolicy {};
    });
}

int RunHnswFsOpenOnlyMain(int argc, char** argv)
{
    auto args_opt = ParseCliArgs(argc, argv);
    if (!args_opt) {
        return 1;
    }
    HnswIoCliArgs args = *args_opt;
    auto root = annfs::common::read_json(args.config_json_path);

    FsUnifiedRunOptions options;
    return RunFsMaybeBatch(args, root, options, []() {
        return FsOpenOnlyExecutionPolicy {};
    });
}

int RunHnswFsIOAsyncMain(int argc, char** argv)
{
    auto args_opt = ParseCliArgs(argc, argv);
    if (!args_opt) {
        return 1;
    }
    HnswIoCliArgs args = *args_opt;
    auto root = annfs::common::read_json(args.config_json_path);

    FsUnifiedRunOptions options;
    return RunFsMaybeBatch(args, root, options, []() {
        return FsAsyncIoExecutionPolicy {};
    });
}

} // namespace annfs::tests
