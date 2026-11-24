#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <json.hpp>
#include "global_config.hpp"

struct DatasetInfo {
    std::string dataset_name;
    std::string data_root_dir;
    std::string base_data_path;
    std::string query_data_path;
    std::string data_format;
    std::string data_type;
    std::string distance_metric;
    std::string gt_format;
    int M;
    int ef_construction;
    int search_ef;
    // earlystopping 相关参数
    int stable_hops;

    // overlap 相关参数
    int stable_iter = 30;
    // Question: ? 
    int least_else_size;
    // 自动生成的路径
    std::string data_path;
    std::string gt_directory;
    std::string index_path;
    std::string hnsw_index_path;
};

struct AdaptiveStableIterConfig {
    int adjust_every_queries = 64;
    int min_stable_iter = 2;
    int max_stable_iter = 64;
    int initial_stable_iter = 30;
    int decrease_step = 1;
    int increase_step = 1;

    double no_prefetch_ratio_high = 0.10;

    // DEEP default; load_from_json selects the other paper datasets before overrides.
    double io_issue_ratio_high = 1.085;
    double adjust_query_percent = 0.0;

    bool stop_eval_on_target = true;
    int stop_eval_target_consecutive_phases = 3;
    int stop_eval_on_conflict_phases = 3;
};

bool ValidateAdaptiveStableIterConfig(const AdaptiveStableIterConfig& cfg);

struct ParamConfig{
public:
    std::string dev_path;
    std::string fvecs_gt_root_dir;
    std::string hnsw_index_root_dir;
    std::string fs_data_dir_path;
    int max_elements;
    int topk;
    int gt_batch_size;
    int gt_gt_k;
    int query_cnt;
    // page - cache 
    bool use_direct_io = true;
    // logger info
    bool is_debug;
    int num_threads;
    int parallel_mode;
    bool skip_recall;

    // 是否在每个 query 级别等待 IO 完成（true）还是仅提交（false，等 Runner 尾部统一回收）。
    bool async_submit_and_wait;

    std::string query_or_base;
    std::string label_map_path;
    AdaptiveStableIterConfig adaptive_cfg;

    DatasetInfo dataset;
public:
    static ParamConfig& instance() {
        static ParamConfig inst;
        return inst;
    }

    ParamConfig(const ParamConfig&) = delete;
    ParamConfig& operator=(const ParamConfig&) = delete;
    void load_from_json(const nlohmann::json& js, const std::string& dataset_name_arg);
    bool UsePackedLabelMap() const { return GlobalConfig::instance().USE_LABEL_MAP; }
    bool UseLabelMap() const { return GlobalConfig::instance().USE_LABEL_MAP; }

private:
    ParamConfig() = default;
};
