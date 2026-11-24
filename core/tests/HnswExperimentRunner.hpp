#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tests/HnswIoCommon.hpp"

namespace annfs::tests {

struct ExperimentBatchRunSpec;

struct OverlapQueryResult {
    std::unordered_set<unsigned long long> issued_label_set;
};

struct ExperimentQueryResult {
    std::vector<unsigned long long> result_ids;
    std::unique_ptr<OverlapQueryResult> overlap;
    double hnsw_time_us = 0.0;
    double io_time_us = -1.0;
    double iter_count = 0.0;
    double dist_count = 0.0;
    bool found_all_gt = true;
};

class ExperimentStrategy {
public:
    virtual ~ExperimentStrategy() = default;

    virtual std::string Name() const = 0;
    virtual std::string ResultFolderName() const = 0;
    virtual bool RequiresIterDist() const { return false; }
    virtual bool ReportsFoundGtFlag() const { return false; }
    virtual bool HandlesIO() const { return false; }
    virtual bool UsesAsyncIoSubmit() const { return false; }
    virtual bool RequiresGroundTruth() const { return false; }
    virtual bool RequiresPostQueryIoWait() const { return false; }
    virtual bool HasCustomOutputDir() const { return false; }
    virtual std::string CustomOutputDir(const HnswIoState& state,
                                        const HnswIoCliArgs& args) const;
    virtual std::string ModeSubdir(const HnswIoState& state) const { return ""; }
    virtual void OnBeforeRun(const HnswIoState& state) {}
    virtual void OnAfterRun(const HnswIoState& state,
                            const std::string& output_dir) {}
    virtual void ApplyBatchRunSpec(const struct ExperimentBatchRunSpec& spec) {}

    virtual ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) = 0;
};

int RunHnswExperiment(const HnswIoCliArgs& args,
                      ExperimentStrategy& strategy);

int ExecuteHnswExperimentWithState(HnswIoState& state,
                                   ExperimentStrategy& strategy);

} // namespace annfs::tests
