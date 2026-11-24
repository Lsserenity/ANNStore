#pragma once

#include <optional>
#include <string>
#include <vector>

#include "tests/HnswExperimentRunner.hpp"

namespace annfs::tests {

struct ExperimentBatchRunSpec {
    std::string dataset_name;
    int repeat_id = 1;
    std::optional<int> num_threads;
    std::optional<int> search_ef;
    std::optional<int> io_depths;
    std::optional<int> query_cnt;
    std::optional<std::string> iter_json_path;
    std::optional<std::string> monitors;
    std::optional<int> stable_iter;
    std::optional<int> adaptive_stop_target_phases;
    std::optional<int> adaptive_stop_on_conflict_phases;
    std::optional<double> no_prefetch_ratio_high;
    std::optional<double> io_issue_ratio_high;
    bool drop_cache = false;
};

struct ExperimentBatchPlan {
    bool persist_state = true;
    bool drop_cache_between_runs = true;
    // true: reuse loaded dataset/index resources across different thread runs
    // for the same dataset. false: use dataset+thread as run unit.
    bool reuse_dataset_index_across_thread_runs = true;
    std::vector<ExperimentBatchRunSpec> runs;
};

ExperimentBatchPlan LoadExperimentBatchPlan(const std::string& path);

int RunHnswExperimentBatch(const std::string& config_path,
                           const ExperimentBatchPlan& plan,
                           ExperimentStrategy& strategy);

} // namespace annfs::tests
