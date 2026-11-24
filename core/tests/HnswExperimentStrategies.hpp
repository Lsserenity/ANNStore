#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "tests/HnswAdaptiveOverlap.hpp"
#include "tests/HnswExperimentRunner.hpp"

namespace annfs::tests {

class OffsetIOStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetIO"; }
    std::string ResultFolderName() const override {
        return "Offset_results";
    }
    bool HandlesIO() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state,
                    const std::string& output_dir) override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    std::atomic<std::uint64_t> ann_us_total_{0};
    std::atomic<std::size_t> total_offsets_{0};
    std::atomic<std::size_t> query_total_{0};
};

class OffsetIOAsyncStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetIOAsync"; }
    std::string ResultFolderName() const override {
        return "OffsetAsync_results";
    }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
};

class EarlyStopStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetEarlyStop"; }
    std::string ResultFolderName() const override {
        return "OffsetEarlyStop_results";
    }
    bool RequiresIterDist() const override { return true; }
    bool HandlesIO() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state,
                    const std::string& output_dir) override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    std::atomic<std::uint64_t> ann_us_total_{0};
    std::atomic<std::size_t> total_offsets_{0};
    std::atomic<std::size_t> query_total_{0};
};

class EarlyStopAsyncStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetEarlyStopAsync"; }
    std::string ResultFolderName() const override {
        return "OffsetEarlyStopAsync_results";
    }
    bool RequiresIterDist() const override { return true; }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state,
                    const std::string& output_dir) override;
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
    std::atomic<std::uint64_t> ann_us_total_{0};
    std::atomic<std::size_t> total_offsets_{0};
    std::atomic<std::size_t> query_total_{0};
};

class EarlyStopOverlapAsyncStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetEarlyStopOverlapAsync"; }
    std::string ResultFolderName() const override {
        return "OffsetEarlyStopOverlapAsync_results";
    }
    bool RequiresIterDist() const override { return true; }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state,
                    const std::string& output_dir) override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
    std::atomic<std::uint64_t> ann_us_total_{0};
    std::atomic<std::size_t> issued_prefetch_labels_total_{0};
    std::atomic<std::size_t> query_total_{0};
    std::atomic<std::size_t> early_first_total_{0};
    std::atomic<std::size_t> overlap_first_total_{0};
    std::atomic<std::size_t> overlap_trigger_total_{0};
};

class IdealTimeStrategy : public ExperimentStrategy {
public:
    IdealTimeStrategy() = default;

    void SetIterJsonPath(const std::string& path) {
        iter_json_path_ = path;
    }

    std::string Name() const override { return "HnswOffsetIdealTime"; }
    std::string ResultFolderName() const override {
        return "Offset_results";
    }
    bool HasCustomOutputDir() const override { return true; }
    void ApplyBatchRunSpec(const ExperimentBatchRunSpec& spec) override;
    std::string CustomOutputDir(const HnswIoState& state,
                                const HnswIoCliArgs& args) const override;
    void OnBeforeRun(const HnswIoState& state) override;

    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    std::string iter_json_path_;
    std::vector<int> iter_counts_;
    std::vector<bool> found_gt_flags_;
};

class IdealTimeAsyncStrategy : public ExperimentStrategy {
public:
    IdealTimeAsyncStrategy() = default;

    void SetIterJsonPath(const std::string& path) {
        iter_json_path_ = path;
    }

    std::string Name() const override { return "HnswOffsetIdealTimeAsync"; }
    std::string ResultFolderName() const override {
        return "OffsetAsync_results";
    }
    bool HasCustomOutputDir() const override { return true; }
    void ApplyBatchRunSpec(const ExperimentBatchRunSpec& spec) override;
    std::string CustomOutputDir(const HnswIoState& state,
                                const HnswIoCliArgs& args) const override;
    void OnBeforeRun(const HnswIoState& state) override;
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;

    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    std::string iter_json_path_;
    std::vector<int> iter_counts_;
    std::vector<bool> found_gt_flags_;
    bool wait_per_query_ = true;
};

class IdealSpaceStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetIdealSpace"; }
    std::string ResultFolderName() const override {
        return "IdealSpace_results";
    }
    bool RequiresIterDist() const override { return true; }
    bool ReportsFoundGtFlag() const override { return true; }
    bool RequiresGroundTruth() const override { return true; }
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;
};

class IdealSpaceAsyncStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetIdealSpaceAsync"; }
    std::string ResultFolderName() const override {
        return "IdealSpaceAsync_results";
    }
    bool RequiresIterDist() const override { return true; }
    bool ReportsFoundGtFlag() const override { return true; }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    bool RequiresGroundTruth() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
};

class OverlapIOStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswOffsetOverlap"; }
    std::string ResultFolderName() const override {
        return "OffsetOverlap_results";
    }
    bool HandlesIO() const override { return true; }
    bool UsesAsyncIoSubmit() const override { return true; }
    bool RequiresPostQueryIoWait() const override { return true; }
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;
};

class MappedOffsetIOStrategy : public ExperimentStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetIO"; }
    std::string ResultFolderName() const override { return "MappedOffset_results"; }
    bool HandlesIO() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state, const std::string& output_dir) override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

protected:
    std::vector<unsigned long long> DrainLabels(
        std::priority_queue<std::pair<float, hnswlib::labeltype>>& pq) const;
    void LoadMap(const HnswIoState& state);

    std::string map_path_;
    mutable std::atomic<std::size_t> missing_total_{0};
    mutable std::atomic<std::size_t> io_offsets_total_{0};
    mutable std::atomic<std::uint64_t> ann_us_total_{0};
    mutable std::atomic<std::uint64_t> map_us_total_{0};
    mutable std::atomic<std::uint64_t> io_us_total_{0};
    mutable std::atomic<std::size_t> query_total_{0};
};

class MappedOffsetIOAsyncStrategy : public MappedOffsetIOStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetIOAsync"; }
    std::string ResultFolderName() const override { return "MappedOffsetAsync_results"; }
    bool UsesAsyncIoSubmit() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
};

class MappedOffsetIOSpanOneShotStrategy : public MappedOffsetIOStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetIOSpanOneShot"; }
    std::string ResultFolderName() const override { return "MappedOffsetSpanOneShot_results"; }
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;
};

class MappedEarlyStopStrategy : public MappedOffsetIOStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetEarlyStop"; }
    std::string ResultFolderName() const override { return "MappedOffsetEarlyStop_results"; }
    bool RequiresIterDist() const override { return true; }
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;
};

class MappedEarlyStopAsyncStrategy : public MappedEarlyStopStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetEarlyStopAsync"; }
    std::string ResultFolderName() const override { return "MappedOffsetEarlyStopAsync_results"; }
    bool UsesAsyncIoSubmit() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    bool RequiresPostQueryIoWait() const override;
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
};

class MappedEarlyStopSpanPoolSyncStrategy : public MappedOffsetIOStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetEarlyStopSpanPool"; }
    std::string ResultFolderName() const override { return "MappedOffsetEarlyStopSpanPool_results"; }
    bool RequiresIterDist() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state, const std::string& output_dir) override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;
};

class MappedEarlyStopSpanPoolAsyncStrategy : public MappedEarlyStopSpanPoolSyncStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetEarlyStopSpanPoolAsync"; }
    std::string ResultFolderName() const override {
        return "MappedOffsetEarlyStopSpanPoolAsync_results";
    }
    bool UsesAsyncIoSubmit() const override { return true; }
    void OnBeforeRun(const HnswIoState& state) override;
    void OnAfterRun(const HnswIoState& state, const std::string& output_dir) override;
    bool RequiresPostQueryIoWait() const override { return false; }
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;

private:
    bool wait_per_query_ = true;
};

class MappedOverlapIOStrategy : public MappedOffsetIOAsyncStrategy {
public:
    std::string Name() const override { return "HnswMappedOffsetOverlap"; }
    std::string ResultFolderName() const override { return "MappedOffsetOverlap_results"; }
    bool RequiresPostQueryIoWait() const override { return true; }
    std::string ModeSubdir(const HnswIoState& state) const override;
    ExperimentQueryResult ExecuteSearch(
        const HnswIoState& state,
        size_t row,
        size_t thread_id,
        int cur_k,
        const std::vector<hnswlib::labeltype>& gt_row) override;
};

} // namespace annfs::tests
