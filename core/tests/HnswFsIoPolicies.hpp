#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <json.hpp>

#include "disk_info.hpp"
#include "global_config.hpp"
#include "io/iouring_manager.hpp"
#include "param_config.hpp"

namespace annfs::io {
class QueryIoEventListener;
}

namespace annfs::tests {

class IoExecutionPolicy {
public:
    virtual ~IoExecutionPolicy() = default;

    virtual bool Init(ParamConfig& pc,
                      GlobalConfig& cfg,
                      int num_threads,
                      const nlohmann::json* root) = 0;

    virtual void SetQueryListener(annfs::io::QueryIoEventListener* listener) = 0;

    virtual void Execute(const std::vector<std::string>& path_list,
                         int row,
                         size_t thread_id) = 0;

    virtual void NotifySkipped(size_t skipped) { (void)skipped; }

    virtual void Finalize() = 0;

    virtual void EmitSummary(const ParamConfig& pc,
                             GlobalConfig& cfg) const = 0;

    virtual nlohmann::json BuildStatsMeta(const ParamConfig& pc) const = 0;

    virtual std::string ResultFolder() const = 0;
    virtual std::string ModeDir() const { return ""; }
    virtual std::string StatsFilename() const = 0;
};

class FsSyncIoExecutionPolicy final : public IoExecutionPolicy {
public:
    bool Init(ParamConfig& pc,
              GlobalConfig& cfg,
              int num_threads,
              const nlohmann::json* root) override;

    void SetQueryListener(annfs::io::QueryIoEventListener* listener) override;

    void Execute(const std::vector<std::string>& path_list,
                 int row,
                 size_t thread_id) override;

    void Finalize() override;

    void EmitSummary(const ParamConfig& pc,
                     GlobalConfig& cfg) const override;

    nlohmann::json BuildStatsMeta(const ParamConfig& pc) const override;

    std::string ResultFolder() const override { return "Fs_results"; }
    std::string StatsFilename() const override { return "FsIoIoStats.json"; }

private:
    GlobalConfig* cfg_ = nullptr;
    std::vector<std::unique_ptr<IOuringManager>> ioers_;
    std::atomic<long long> effective_io_counter_{0};
    std::atomic<long long> total_path_candidates_{0};
    std::atomic<long long> total_queries_executed_{0};
};

class FsOpenOnlyExecutionPolicy final : public IoExecutionPolicy {
public:
    bool Init(ParamConfig& pc,
              GlobalConfig& cfg,
              int num_threads,
              const nlohmann::json* root) override;

    void SetQueryListener(annfs::io::QueryIoEventListener* listener) override;

    void Execute(const std::vector<std::string>& path_list,
                 int row,
                 size_t thread_id) override;

    void NotifySkipped(size_t skipped) override;

    void Finalize() override;

    void EmitSummary(const ParamConfig& pc,
                     GlobalConfig& cfg) const override;

    nlohmann::json BuildStatsMeta(const ParamConfig& pc) const override;

    std::string ResultFolder() const override { return "FsOpenOnly_results"; }
    std::string StatsFilename() const override { return "FsOpenOnlyStats.json"; }

private:
    bool use_direct_io_ = true;
    std::atomic<long long> open_fail_counter_{0};
};

class FsAsyncIoExecutionPolicy final : public IoExecutionPolicy {
public:
    bool Init(ParamConfig& pc,
              GlobalConfig& cfg,
              int num_threads,
              const nlohmann::json* root) override;

    void SetQueryListener(annfs::io::QueryIoEventListener* listener) override;

    void Execute(const std::vector<std::string>& path_list,
                 int row,
                 size_t thread_id) override;

    void Finalize() override;

    void EmitSummary(const ParamConfig& pc,
                     GlobalConfig& cfg) const override;

    nlohmann::json BuildStatsMeta(const ParamConfig& pc) const override;

    std::string ResultFolder() const override;
    std::string ModeDir() const override;
    std::string StatsFilename() const override { return "FsAsyncStats.json"; }

private:
    GlobalConfig* cfg_ = nullptr;
    bool overlap_mode_ = false;
    bool shared_worker_ = false;
    int num_threads_ = 1;
    std::string async_mode_ = "per_query_wait";
    std::vector<std::unique_ptr<IOuringManager>> ioers_;
    std::atomic<long long> effective_io_counter_{0};
    std::atomic<long long> total_path_candidates_{0};
    std::atomic<long long> total_queries_executed_{0};
    std::atomic<std::uint64_t> execute_wall_us_total_{0};
    std::atomic<std::uint64_t> finalize_wait_us_{0};
};

} // namespace annfs::tests
