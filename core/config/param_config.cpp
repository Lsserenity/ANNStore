#include "param_config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

bool is_absolute_path(const std::string& path)
{
    if (path.empty()) {
        return false;
    }
#ifdef _WIN32
    if (path.size() > 1 && path[1] == ':') {
        return true;
    }
#endif
    return path[0] == '/';
}

std::string join_paths(const std::string& root, const std::string& child)
{
    if (root.empty()) {
        return child;
    }
    if (child.empty()) {
        return root;
    }
    if (root.back() == '/') {
        return root + child;
    }
    return root + "/" + child;
}

std::string resolve_with_roots(const std::string& path,
                               std::initializer_list<std::string> roots)
{
    if (path.empty()) {
        return {};
    }
    if (is_absolute_path(path)) {
        return path;
    }
    for (const auto& root : roots) {
        if (!root.empty()) {
            return join_paths(root, path);
        }
    }
    return path;
}

template <typename T>
T require_key(const nlohmann::json& js, const char* key, const char* scope)
{
    if (!js.contains(key)) {
        std::cerr << "[WARNING] missing required key '" << key
                  << "' in " << scope << std::endl;
        throw std::runtime_error(std::string("missing required key: ") + key);
    }

    try {
        return js.at(key).get<T>();
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] invalid value for key '" << key
                  << "' in " << scope << ": " << e.what() << std::endl;
        throw;
    }
}

std::map<int, int> parse_mapped_spanpool_buckets(const nlohmann::json& js)
{
    auto& gc = GlobalConfig::instance();
    if (!js.contains("mapped_spanpool_buckets")) {
        gc.MAPPED_SPANPOOL_BUCKETS.clear();
        gc.SPANPOOL_MAX_BLOCKS = 0;
        return {};
    }
    const auto& node = js.at("mapped_spanpool_buckets");

    if (!node.is_object()) {
        std::cerr << "[WARNING] mapped_spanpool_buckets must be an object" << std::endl;
        throw std::runtime_error("mapped_spanpool_buckets must be object");
    }
    int mx = 0;
    std::map<int, int> out;
    for (auto it = node.begin(); it != node.end(); ++it) {
        try {
            const int blocks = std::stoi(it.key());
            const int cnt = it.value().get<int>();
            if (blocks == 0 || cnt <= 0) {
                std::cerr << "[WARNING] invalid mapped_spanpool_buckets entry key="
                          << it.key() << " value=" << it.value() << std::endl;
                throw std::runtime_error("mapped_spanpool_buckets entry must be positive");
            }
            out[blocks] = cnt;
            mx = std::max(mx, blocks);
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] invalid mapped_spanpool_buckets entry '" << it.key()
                      << "': " << e.what() << std::endl;
            throw;
        }
    }
    gc.MAPPED_SPANPOOL_BUCKETS = out;
    gc.SPANPOOL_MAX_BLOCKS = mx;
    return out;
}

std::string normalize_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

void load_global_root_fields(ParamConfig* pc, const nlohmann::json& js)
{
    auto& gc = GlobalConfig::instance();
    pc->dev_path = require_key<std::string>(js, "dev_path", "root config");
    gc.DEV_PATHS = {pc->dev_path};
    pc->use_direct_io = require_key<bool>(js, "use_direct_io", "root config");
    pc->label_map_path = js.value("label_map_path", std::string());
    gc.RING_DEPTH = require_key<int>(js, "io_depths", "root config");
    pc->async_submit_and_wait = require_key<bool>(js, "async_submit_and_wait", "root config");
    gc.ASYNC_POLL_US = require_key<int>(js, "async_poll_us", "root config");
    gc.ASYNC_RECLAIM_BATCH = require_key<int>(js, "async_reclaim_batch", "root config");
    gc.ASYNC_RECLAIM_THRESHOLD = require_key<int>(js, "async_reclaim_threshold", "root config");
    gc.ASYNC_RECLAIM_INTERVAL_US = require_key<int>(
        js, "async_reclaim_interval_us", "root config");
    gc.SYNC_RECLAIM_BATCH = std::max(1, js.value("sync_reclaim_batch", 32));
    gc.SYNC_SUBMIT_WAIT_NR = std::max(1, js.value("sync_submit_wait_nr", 1));
    parse_mapped_spanpool_buckets(js);

    gc.IO_WORKER_MODE = normalize_lower(js.value("io_worker_mode", std::string("per_thread")));
    if (gc.IO_WORKER_MODE != "per_thread" && gc.IO_WORKER_MODE != "shared") {
        std::cerr << "[WARNING] invalid io_worker_mode: " << gc.IO_WORKER_MODE
                  << ", fallback to per_thread" << std::endl;
        gc.IO_WORKER_MODE = "per_thread";
    }

    gc.MAPPED_SPAN_MODE = normalize_lower(js.value("mapped_span_mode", std::string("split")));
    if (gc.MAPPED_SPAN_MODE != "split" && gc.MAPPED_SPAN_MODE != "bucket_pool") {
        std::cerr << "[WARNING] invalid mapped_span_mode: " << gc.MAPPED_SPAN_MODE
                  << ", fallback to split" << std::endl;
        gc.MAPPED_SPAN_MODE = "split";
    }

    gc.USE_LABEL_MAP = js.value("use_label_map", false);
    gc.PACKED64_CROSS_BITS = js.value("packed64_cross_bits", 16);
    if (gc.PACKED64_CROSS_BITS < 1) {
        gc.PACKED64_CROSS_BITS = 1;
    }
    if (gc.PACKED64_CROSS_BITS > 16) {
        gc.PACKED64_CROSS_BITS = 16;
    }
    const std::string decode = normalize_lower(
        js.value("label_decode_mode", std::string("packed_map")));
    if (decode == "shift_offbits" || decode == "shift") {
        gc.LABEL_DECODE_MODE_CFG = GlobalConfig::LABEL_DECODE_MODE::SHIFT_OFF_BITS;
    } else {
        gc.LABEL_DECODE_MODE_CFG = GlobalConfig::LABEL_DECODE_MODE::PACKED_MAP;
    }
}

void load_adaptive_cfg_from_json(AdaptiveStableIterConfig* cfg,
                                 const nlohmann::json& node)
{
    if (!cfg || !node.is_object()) {
        return;
    }
    auto apply_int = [&](int& dst, std::initializer_list<const char*> keys) {
        for (const char* key : keys) {
            if (node.contains(key)) {
                dst = node.value(key, dst);
                return;
            }
        }
    };
    auto apply_double = [&](double& dst, std::initializer_list<const char*> keys) {
        for (const char* key : keys) {
            if (node.contains(key)) {
                dst = node.value(key, dst);
                return;
            }
        }
    };
    auto apply_bool = [&](bool& dst,
                                        std::initializer_list<const char*> keys) {
        for (const char* key : keys) {
            if (node.contains(key)) {
                dst = node.value(key, dst);
                return;
            }
        }
    };

    apply_int(cfg->adjust_every_queries, {"stableiter_phase"});
    apply_int(cfg->initial_stable_iter, {"stableiter_init"});
    apply_int(cfg->min_stable_iter, {"stableiter_min"});
    apply_int(cfg->max_stable_iter, {"stableiter_max"});
    apply_int(cfg->decrease_step, {"stableiter_dec_step"});
    apply_int(cfg->increase_step, {"stableiter_inc_step"});

    apply_double(cfg->no_prefetch_ratio_high, {"no_prefetch_ratio_high"});
    apply_double(cfg->io_issue_ratio_high, {"io_issue_ratio_high"});
    apply_double(cfg->adjust_query_percent, {"adjust_query_percent"});

    apply_bool(cfg->stop_eval_on_target,
                             {"adaptive_stop_on_target", "stop_eval_on_target"});
    apply_int(cfg->stop_eval_target_consecutive_phases,
              {"adaptive_stop_target_phases", "stop_eval_target_consecutive_phases"});
    apply_int(cfg->stop_eval_on_conflict_phases,
              {"adaptive_stop_on_conflict_phases", "stop_eval_on_conflict_phases"});
}

} // namespace

using json = nlohmann::json;

void ParamConfig::load_from_json(const json& js, const std::string& dataset_name_arg)
{
    load_global_root_fields(this, js);
    adaptive_cfg = AdaptiveStableIterConfig{};
    hnsw_index_root_dir = require_key<std::string>(js, "hnsw_index_root_dir", "root config");
    fs_data_dir_path = require_key<std::string>(js, "fs_data_dir_path", "root config");

    auto& gc = GlobalConfig::instance();
    gc.FS_MAPPER_FILE_LIMIT = require_key<size_t>(js, "fs_mapper_file_limit", "root config");

    max_elements = require_key<int>(js, "max_elements", "root config");
    topk = require_key<int>(js, "topk", "root config");
    gt_batch_size = require_key<int>(js, "gt_batch_size", "root config");
    gt_gt_k = require_key<int>(js, "gt_gt_k", "root config");
    query_cnt = require_key<int>(js, "query_cnt", "root config");

    is_debug = require_key<bool>(js, "is_debug", "root config");
    num_threads = require_key<int>(js, "num_threads", "root config");
    parallel_mode = require_key<int>(js, "parallel_mode", "root config");
    query_or_base = require_key<std::string>(js, "query_or_base", "root config");
    skip_recall = require_key<bool>(js, "skip_recall", "root config");

    std::string dataset_name = dataset_name_arg;
    if (dataset_name.empty()) {
        dataset_name = require_key<std::string>(js, "dataset_name", "root config");
    }
    if (dataset_name.empty()) {
        std::cerr << "[WARNING] dataset_name is empty" << std::endl;
        throw std::runtime_error("dataset_name is empty");
    }

    const auto& dataset_list = require_key<json>(js, "dataset_list", "root config");
    if (!dataset_list.is_object()) {
        std::cerr << "[WARNING] dataset_list must be object" << std::endl;
        throw std::runtime_error("dataset_list must be object");
    }

    auto it = dataset_list.find(dataset_name);
    if (it == dataset_list.end()) {
        std::cerr << "[WARNING] dataset_name not found in dataset_list: "
                  << dataset_name << std::endl;
        throw std::runtime_error("dataset_name not found in dataset_list");
    }
    const json& dcfg = *it;
    // Paper defaults precede explicit dataset configuration overrides.
    std::string paper_dataset = normalize_lower(dataset_name);
    paper_dataset.erase(std::remove_if(paper_dataset.begin(), paper_dataset.end(),
        [](char c) { return c == ' ' || c == '-' || c == '_'; }), paper_dataset.end());
    if (paper_dataset == "msmarco" || paper_dataset == "marco") {
        adaptive_cfg.io_issue_ratio_high = 1.035;
    } else if (paper_dataset == "laioni2i" || paper_dataset == "laion") {
        adaptive_cfg.io_issue_ratio_high = 1.025;
    } else if (paper_dataset == "sift" || paper_dataset == "sift100m" || paper_dataset == "sift10m") {
        adaptive_cfg.io_issue_ratio_high = 1.030;
    } else if (paper_dataset != "deep" && paper_dataset != "deep100m" && paper_dataset != "deep10m") {
        std::cerr << "[WARNING] no paper IRR default for dataset " << dataset_name
                  << "; using 1.085 unless adaptive_overlap.io_issue_ratio_high is set\n";
    }
    if (dcfg.contains("adaptive_overlap")) {
        load_adaptive_cfg_from_json(&adaptive_cfg, dcfg.at("adaptive_overlap"));
    }

    const std::string data_root_dir = require_key<std::string>(
        dcfg, "data_root_dir", "dataset config");
    std::cout << "data_root_dir: " << data_root_dir << std::endl;

    dataset.dataset_name = dataset_name;
    dataset.data_root_dir = data_root_dir;
    dataset.base_data_path = resolve_with_roots(
        require_key<std::string>(dcfg, "base_data_path", "dataset config"),
        { data_root_dir, fvecs_gt_root_dir });
    std::cout << "base_data_path: " << dataset.base_data_path << std::endl;

    dataset.query_data_path = resolve_with_roots(
        require_key<std::string>(dcfg, "query_data_path", "dataset config"),
        { data_root_dir, fvecs_gt_root_dir });
    std::cout << "query_data_path: " << dataset.query_data_path << std::endl;

    dataset.data_format = require_key<std::string>(dcfg, "data_format", "dataset config");
    dataset.data_type = require_key<std::string>(dcfg, "data_type", "dataset config");
    dataset.distance_metric = require_key<std::string>(dcfg, "distance_metric", "dataset config");
    dataset.gt_format = require_key<std::string>(dcfg, "gt_format", "dataset config");

    dataset.M = require_key<int>(dcfg, "M", "dataset config");
    dataset.ef_construction = require_key<int>(dcfg, "ef_construction", "dataset config");
    dataset.search_ef = require_key<int>(dcfg, "search_ef", "dataset config");
    dataset.stable_hops = require_key<int>(dcfg, "stable_hops", "dataset config");
    dataset.stable_iter = dcfg.value("stable_iter", 30);
    dataset.least_else_size = require_key<int>(dcfg, "least_else_size", "dataset config");

    dataset.data_path =
        (query_or_base == "query") ? dataset.query_data_path : dataset.base_data_path;
    std::cout << "data_path: " << dataset.data_path << std::endl;

    const std::string configured_gt_path = require_key<std::string>(
        dcfg, "gt_path", "dataset config");
    if (!configured_gt_path.empty()) {
        dataset.gt_directory = resolve_with_roots(
            configured_gt_path,
            { data_root_dir, fvecs_gt_root_dir });
    } else {
        dataset.gt_directory = resolve_with_roots(
            dataset_name + "_" + query_or_base + "_to_unique_base_gt_ip",
            { fvecs_gt_root_dir });
    }
    std::cout << "gt_directory: " << dataset.gt_directory << std::endl;

    const std::string idx_path = require_key<std::string>(
        dcfg, "hnsw_index_path", "dataset config");
    if (!idx_path.empty()) {
        dataset.index_path = resolve_with_roots(
            idx_path,
            { data_root_dir, hnsw_index_root_dir });
    } else {
        dataset.index_path =
            join_paths(hnsw_index_root_dir,
                "M" + std::to_string(dataset.M) +
                "_efc" + std::to_string(dataset.ef_construction) +
                "/" + dataset_name + ".bin");
    }
    dataset.hnsw_index_path = dataset.index_path;
    std::cout << "hnsw_index_path: " << dataset.hnsw_index_path << std::endl;

    std::string root_map_path = js.value("label_map_path", std::string());
    std::string ds_map_path = dcfg.value("label_map_path", std::string());
    std::string merged_map_path = !ds_map_path.empty() ? ds_map_path : root_map_path;
    if (merged_map_path.empty()) {
        merged_map_path = "label_map_" + dataset_name + "_l1centric_blockaligned.txt";
    }
    label_map_path = resolve_with_roots(merged_map_path, { data_root_dir, "." });

    if (!ValidateAdaptiveStableIterConfig(adaptive_cfg)) {
        std::cerr << "[WARNING] invalid adaptive_overlap config values" << std::endl;
        throw std::runtime_error("invalid adaptive_overlap config");
    }
}

bool ValidateAdaptiveStableIterConfig(const AdaptiveStableIterConfig& cfg)
{
    return cfg.adjust_every_queries > 0 &&
           cfg.min_stable_iter > 0 &&
           cfg.max_stable_iter > 0 &&
           cfg.decrease_step > 0 &&
           cfg.increase_step > 0 &&
           cfg.min_stable_iter <= cfg.max_stable_iter &&
           cfg.no_prefetch_ratio_high >= 0.0 &&
           cfg.no_prefetch_ratio_high <= 1.0 &&
           cfg.io_issue_ratio_high > 0.0 &&
           cfg.adjust_query_percent >= 0.0 &&
           cfg.adjust_query_percent <= 100.0 &&
           cfg.stop_eval_target_consecutive_phases > 0 &&
           cfg.stop_eval_on_conflict_phases >= 0;
}
