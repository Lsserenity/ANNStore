#include "tests/HnswIoCommon.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <system_error>
#include <unordered_set>

#include "data_io.hpp"
#include "disk_info.hpp"
#include "disk_info.hpp"
#include "json_utils.hpp"
#include "tests/QueryLatencyTracker.hpp"
#include <hnsw/uint8.hpp>

namespace fs = std::filesystem;

namespace annfs::tests {
namespace {

std::string SanitizePathToken(std::string value)
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

std::string EnvOrDefault(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return value;
}

std::string UtcRunStamp()
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

} // namespace

void PersistQueryTimeline(const std::string& out_dir,
                          const QueryLatencyTracker& tracker,
                          const GlobalConfig* cfg)
{
    const auto& start_us = tracker.StartTimesUs();
    const auto& end_us = tracker.EndTimesUs();
    if (start_us.size() != end_us.size()) {
        if (cfg && cfg->logger) {
            cfg->logger->error(
                "Query timeline start/end arrays mismatched ({} vs {})",
                start_us.size(),
                end_us.size());
        }
        return;
    }

    nlohmann::json entries = nlohmann::json::array();
    for (size_t i = 0; i < start_us.size(); ++i) {
        long long start_val = start_us[i];
        long long end_val = end_us[i];
        long long latency = (start_val >= 0 && end_val >= 0)
            ? (end_val - start_val)
            : -1;
        entries.push_back({
            {"query_id", static_cast<int>(i)},
            {"start_us", start_val},
            {"end_us", end_val},
            {"latency_us", latency}
        });
    }

    nlohmann::json timeline = {
        {"query_cnt", start_us.size()},
        {"entries", entries}
    };

    const std::string path = out_dir + "/HNSWIO_QueryTimeline.json";
    std::ofstream out(path);
    if (!out.is_open()) {
        if (cfg && cfg->logger) {
            cfg->logger->error(
                "Failed to open {} for writing query timeline",
                path);
        }
        return;
    }
    out << timeline.dump(4);
    if (cfg && cfg->logger) {
        cfg->logger->info("Query timeline persisted to {}", path);
    }
}

ScopedQueryListenerBinding::ScopedQueryListenerBinding(
    std::vector<std::unique_ptr<IOuringManager>>& ioers,
    annfs::io::QueryIoEventListener* listener)
    : manager_ioers_(&ioers),
      listener_(listener)
{
    for (auto& ioer : *manager_ioers_) {
        if (ioer) {
            ioer->set_query_listener(listener_);
        }
    }
}

ScopedQueryListenerBinding::~ScopedQueryListenerBinding()
{
    if (manager_ioers_) {
        for (auto& ioer : *manager_ioers_) {
            if (ioer) {
                ioer->set_query_listener(nullptr);
            }
        }
    }
}

namespace {

void print_usage(const char* binary)
{
    std::cerr << "Usage: " << binary
              << " <config_json_path> <dataset_name> [num_threads] [repeat_id]"
              << " [--stableiter <value>] [--stable_hops <value>]"
              << " [--timeline <on|off>]"
              << " [--search_ef <value>]"
              << " [--offset-decode-mode <packed_label|shift_offbits>]\n"
              << "  or:  " << binary << " warmup <config_json_path>\n";
}

std::optional<bool> parse_bool_flag(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return std::nullopt;
}

std::string infer_gt_format(const std::string& hinted_format,
                            const std::string& path)
{
    if (!hinted_format.empty()) {
        std::string fmt = hinted_format;
        std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        return fmt;
    }

    auto dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "";
    }
    std::string ext = path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (ext == "ibin" || ext == "bin") {
        return "ibin";
    }
    if (ext == "ivecs" || ext == "vecs") {
        return "ivecs";
    }
    return ext;
}

void log_dataset_info(const GlobalConfig& cfg,
                      const ParamConfig& pc,
                      const HnswIoCliArgs& args)
{
    /**
        支持打印使用的数据集信息
    */
    if (!cfg.logger) {
        return;
    }
    auto opt_int_to_string = [](const std::optional<int>& value) {
        return value ? std::to_string(*value) : std::string("null");
    };
    auto opt_double_to_string = [](const std::optional<double>& value) {
        return value ? std::to_string(*value) : std::string("null");
    };
    const auto& gc = GlobalConfig::instance();

    cfg.logger->info("Dataset loaded: {}", pc.dataset.dataset_name);
    cfg.logger->info("Data path: {}", pc.dataset.data_path);
    cfg.logger->info("GT resource: {}", pc.dataset.gt_directory);
    cfg.logger->info("Index path: {}", pc.dataset.index_path);
    cfg.logger->info("config_json_path: {}", args.config_json_path);
    cfg.logger->info("dataset_name: {} ; search_ef: {} ; num_threads: {} ; stable_iter: {} ; stable_hops: {}", 
        args.dataset_name, pc.dataset.search_ef, pc.num_threads, pc.dataset.stable_iter, pc.dataset.stable_hops);
    cfg.logger->info("io_worker_mode: {}", gc.IO_WORKER_MODE);
    cfg.logger->info("mapped_span_mode: {}", gc.MAPPED_SPAN_MODE);
    cfg.logger->info("use_label_map: {}", gc.USE_LABEL_MAP ? "true" : "false");
    cfg.logger->info("skip_recall: {}", pc.skip_recall ? "true" : "false");
    cfg.logger->info("label_decode_mode: {}",
                     GlobalConfig::LabelDecodeModeToString(gc.LABEL_DECODE_MODE_CFG));
    cfg.logger->info("repeat_id: {}", args.repeat_id);
    cfg.logger->info("offset_decode_mode: {}",
                     args.offset_decode_mode == OffsetDecodeMode::PackedLabel
                         ? "packed_label"
                         : "shift_offbits");
    cfg.logger->info("override_run_index: {}",
                     opt_int_to_string(args.override_run_index));
    cfg.logger->info("enable_query_timeline: {}",
                     args.enable_query_timeline ? "true" : "false");
}

std::unique_ptr<hnswlib::SpaceInterface<float>> create_search_space(
    int dim,
    const DatasetInfo& dataset)
{
    // 根据数据集的 data_type 和 distance_metric 来选择合适的空间类型
    if (dataset.data_type == "uint8" || dataset.data_type == "u8") {
        return std::make_unique<hnswlib::L2SpaceUint8>(dim);
    }
    if (dataset.distance_metric == "ip") {
        return std::make_unique<hnswlib::InnerProductSpace>(dim);
    }
    return std::make_unique<hnswlib::L2Space>(dim);
}

} // namespace

std::optional<HnswIoCliArgs> ParseCliArgs(int argc, char** argv)
{
    /**
     * 通用 CLI 参数
     * 位置参数：<config_json_path> <dataset_name> [num_threads] [repeat_id]
     * 可选参数：
        --stableiter / --stableiter=<v>
        --stable_hops / --stablehops / --stable_hops=<v> / --stablehops=<v>
        --timeline <on|off> / --timeline=<on|off>
        --search_ef / --search_ef=<v>
        --offset-decode-mode / --offset-decode-mode=<packed_label|shift_offbits>
     */
    HnswIoCliArgs args;
    args.repeat_id = 1;
    int positional_index = 0;
    auto parse_int = [](const std::string& value) -> std::optional<int> {
        try {
            size_t parsed = 0;
            int result = std::stoi(value, &parsed);
            if (parsed != value.size()) {
                return std::nullopt;
            }
            return result;
        } catch (...) {
            return std::nullopt;
        }
    };
    for (int i = 1; i < argc; ++i) {
        std::string token(argv[i]);
        auto require_value = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };
        if (token == "--stableiter") {
            auto value = require_value();
            if (!value) {
                return std::nullopt;
            }
            auto parsed = parse_int(*value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_stable_iter = *parsed;
            continue;
        }
        if (token.rfind("--stableiter=", 0) == 0) {
            auto value = token.substr(std::string("--stableiter=").size());
            if (value.empty()) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            auto parsed = parse_int(value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_stable_iter = *parsed;
            continue;
        }
        if (token == "--stable_hops" || token == "--stablehops") {
            auto value = require_value();
            if (!value) {
                return std::nullopt;
            }
            auto parsed = parse_int(*value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_stable_hops = *parsed;
            continue;
        }
        if (token.rfind("--stable_hops=", 0) == 0 ||
            token.rfind("--stablehops=", 0) == 0) {
            auto pos = token.find('=');
            auto value = token.substr(pos + 1);
            if (value.empty()) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            auto parsed = parse_int(value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_stable_hops = *parsed;
            continue;
        }
        if (token == "--timeline") {
            auto value = require_value();
            if (!value) {
                return std::nullopt;
            }
            auto parsed = parse_bool_flag(*value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.enable_query_timeline = *parsed;
            continue;
        }
        if (token.rfind("--timeline=", 0) == 0) {
            auto value = token.substr(std::string("--timeline=").size());
            if (value.empty()) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            auto parsed = parse_bool_flag(value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.enable_query_timeline = *parsed;
            continue;
        }
        if (token == "--search_ef") {
            auto value = require_value();
            if (!value) {
                return std::nullopt;
            }
            auto parsed = parse_int(*value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_search_ef = *parsed;
            continue;
        }
        if (token.rfind("--search_ef=", 0) == 0) {
            auto value = token.substr(std::string("--search_ef=").size());
            if (value.empty()) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            auto parsed = parse_int(value);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_search_ef = *parsed;
            continue;
        }
        if (token == "--offset-decode-mode" || token == "--offset_decode_mode") {
            auto value = require_value();
            if (!value) {
                return std::nullopt;
            }
            if (*value == "packed_label" || *value == "packed" || *value == "map") {
                args.offset_decode_mode = OffsetDecodeMode::PackedLabel;
            } else if (*value == "shift_offbits" || *value == "shift") {
                args.offset_decode_mode = OffsetDecodeMode::ShiftOffbits;
            } else {
                print_usage(argv[0]);
                return std::nullopt;
            }
            continue;
        }
        if (token.rfind("--offset-decode-mode=", 0) == 0 ||
            token.rfind("--offset_decode_mode=", 0) == 0) {
            auto pos = token.find('=');
            auto value = token.substr(pos + 1);
            if (value == "packed_label" || value == "packed" || value == "map") {
                args.offset_decode_mode = OffsetDecodeMode::PackedLabel;
            } else if (value == "shift_offbits" || value == "shift") {
                args.offset_decode_mode = OffsetDecodeMode::ShiftOffbits;
            } else {
                print_usage(argv[0]);
                return std::nullopt;
            }
            continue;
        }
        if (token.rfind("--", 0) == 0) {
            print_usage(argv[0]);
            return std::nullopt;
        }
        if (positional_index == 0) {
            args.config_json_path = token;
        } else if (positional_index == 1) {
            args.dataset_name = token;
        } else if (positional_index == 2) {
            auto parsed = parse_int(token);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.override_num_threads = *parsed;
        } else if (positional_index == 3) {
            auto parsed = parse_int(token);
            if (!parsed) {
                print_usage(argv[0]);
                return std::nullopt;
            }
            args.repeat_id = *parsed;
        } else {
            print_usage(argv[0]);
            return std::nullopt;
        }
        ++positional_index;
    }

    if (positional_index < 2) {
        print_usage(argv[0]);
        return std::nullopt;
    }
    return args;
}

HnswIoState PrepareState(const HnswIoCliArgs& args)
{
    /**
        核心：状态初始化（配置/索引/io worker）
            - 根据CLI参数加载配置文件，并覆盖相关参数到全局配置和参数配置单例中；
            - 加载数据集，包括向量数据和地面真实值（GT），并构建HNSW索引对象；
            - 根据配置初始化IO管理器（IOuringManager）实例，支持共享或独立的IO工作者模式；
            - 返回包含所有实验所需状态信息的HnswIoState对象；
        接受CLI命令行参数，对global config以及param config进行覆盖更新
    */
    auto& cfg = GlobalConfig::instance();
    auto& pc = ParamConfig::instance();

    auto js = annfs::common::read_json(args.config_json_path);
    pc.load_from_json(js, args.dataset_name);
    cfg.init_logger(pc.is_debug);

    if (args.override_stable_iter) {
        pc.dataset.stable_iter = *args.override_stable_iter;
    }

    if (args.override_num_threads) {
        pc.num_threads = *args.override_num_threads;
    }
    if (args.override_stable_hops) {
        pc.dataset.stable_hops = *args.override_stable_hops;
    }
    if (args.override_search_ef) {
        pc.dataset.search_ef = *args.override_search_ef;
    }
    if (args.offset_decode_mode == OffsetDecodeMode::ShiftOffbits) {
        cfg.LABEL_DECODE_MODE_CFG = GlobalConfig::LABEL_DECODE_MODE::SHIFT_OFF_BITS;
    } else {
        cfg.LABEL_DECODE_MODE_CFG = GlobalConfig::LABEL_DECODE_MODE::PACKED_MAP;
    }

    log_dataset_info(cfg, pc, args);

    HnswIoState state;
    state.args = args;
    state.cfg = &cfg;
    state.pc = &pc;
    // 传递参数到 state 中，供后续实验使用
    state.num_threads = pc.num_threads;
    state.base_stable_iter = pc.dataset.stable_iter;

    unsigned N = 0;
    unsigned Dim = 0;
    std::unique_ptr<float[]> raw_data_float;
    std::unique_ptr<uint8_t[]> raw_data_u8;
    const std::string data_format =
        pc.dataset.data_format.empty()
            ? "bin"
            : pc.dataset.data_format;
    const bool use_uint8_data =
        pc.dataset.data_type == "uint8" || pc.dataset.data_type == "u8";
    state.vec_data_type = use_uint8_data
        ? VectorDataType::Uint8
        : VectorDataType::Float32;
    try {
        if (use_uint8_data) {
            if (data_format == "bin" || data_format == "fbin" ||
                data_format == "bbin" || data_format == "ibin" ||
                data_format == "u8bin") {
                raw_data_u8.reset(
                    annfs::common::read_data_bin<uint8_t>(
                        pc.dataset.data_path,
                        N,
                        Dim));
            } else if (data_format == "vecs" || data_format == "fvecs" ||
                       data_format == "bvecs" || data_format == "u8vecs") {
                raw_data_u8.reset(
                    annfs::common::read_data_vecs<uint8_t>(
                        pc.dataset.data_path,
                        N,
                        Dim));
            } else {
                throw std::runtime_error(
                    "Unsupported data_format for uint8 dataset: " +
                    data_format);
            }
        } else {
            raw_data_float.reset(
                annfs::common::read_data(pc.dataset.data_path,
                                         data_format,
                                         N,
                                         Dim));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read dataset: " +
                                 std::string(e.what()));
    }

    if (N == 0 || Dim == 0) {
        throw std::runtime_error("No vectors read from " +
                                 pc.dataset.data_path);
    }

    if (use_uint8_data) {
        state.vecs_uint8.reserve(N);
        for (unsigned i = 0; i < N; ++i) {
            state.vecs_uint8.emplace_back(
                raw_data_u8.get() + static_cast<uint64_t>(i) * Dim,
                raw_data_u8.get() + static_cast<uint64_t>(i + 1) * Dim);
        }
    } else {
        state.vecs_float.reserve(N);
        for (unsigned i = 0; i < N; ++i) {
            state.vecs_float.emplace_back(
                raw_data_float.get() + static_cast<uint64_t>(i) * Dim,
                raw_data_float.get() + static_cast<uint64_t>(i + 1) * Dim);
        }
        annfs::common::normalize_rows(state.vecs_float);
    }

    int dim = static_cast<int>(Dim);
    size_t max_elements = state.VectorCount();
    if (cfg.logger) {
        cfg.logger->info("vecs.size(): {}, dim: {}", max_elements, Dim);
        cfg.logger->info("max_elements: {}", max_elements);
    }

    state.query_cnt =
        std::min(pc.query_cnt, static_cast<int>(max_elements));
    if (state.query_cnt != pc.query_cnt && cfg.logger) {
        cfg.logger->info("query_cnt clipped to {}", state.query_cnt);
    }
    pc.query_cnt = state.query_cnt;
    // 加载 index 并 setEf 搜索参数
    state.space = create_search_space(dim, pc.dataset);
    state.hnsw_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        state.space.get(),
        pc.dataset.index_path);
    state.hnsw_index->setEf(pc.dataset.search_ef);

    std::error_code gt_ec;
    bool gt_is_dir = false;
    if (!pc.dataset.gt_directory.empty()) {
        gt_is_dir = fs::is_directory(pc.dataset.gt_directory, gt_ec);
    }

    if (gt_is_dir) {
        state.gt_reader = std::make_unique<annfs::common::GroundTruthReader>(
            pc.dataset.gt_directory,
            pc.gt_batch_size,
            max_elements,
            pc.gt_gt_k);
    } else if (!pc.dataset.gt_directory.empty()) {
        unsigned gtN = 0;
        unsigned gtDim = 0;
        const std::string gt_format =
            infer_gt_format(pc.dataset.gt_format, pc.dataset.gt_directory);
        auto gt_unsigned = annfs::common::read_gt(
            pc.dataset.gt_directory,
            gt_format.empty() ? "ibin" : gt_format,
            gtN,
            gtDim);

        state.gt_preloaded.reserve(gt_unsigned.size());
        for (const auto& row : gt_unsigned) {
            std::vector<hnswlib::labeltype> converted;
            converted.reserve(row.size());
            for (auto id : row) {
                converted.push_back(
                    static_cast<hnswlib::labeltype>(id));
            }
            state.gt_preloaded.push_back(std::move(converted));
        }

        if (gtN == 0 || state.gt_preloaded.empty()) {
            throw std::runtime_error(
                "No ground truth entries loaded from " +
                pc.dataset.gt_directory);
        }

        if (gtN < static_cast<unsigned>(state.query_cnt)) {
            state.query_cnt = static_cast<int>(gtN);
            pc.query_cnt = state.query_cnt;
            if (cfg.logger) {
                cfg.logger->warn(
                    "Clipping query_cnt to GT size: {}",
                    state.query_cnt);
            }
        }
    } else {
        throw std::runtime_error("Ground truth path is empty.");
    }

    if (max_elements < static_cast<size_t>(pc.query_cnt)) {
        if (cfg.logger) {
            cfg.logger->warn(
                "max_elements < query_cnt, reset query_cnt to {}",
                max_elements);
        }
        pc.query_cnt = static_cast<int>(max_elements);
        state.query_cnt = pc.query_cnt;
    }
    // 初始化 io_uring workers
    const auto& gc = GlobalConfig::instance();
    get_block_size(pc.dev_path);
    state.use_shared_io_worker = (gc.IO_WORKER_MODE == "shared");
    int ioer_count = state.use_shared_io_worker ? 1 : state.num_threads;
    if (state.use_shared_io_worker && cfg.logger) {
        cfg.logger->info("IO worker mode: shared ({} threads -> 1 ring)",
                         state.num_threads);
    }
    // 根据 ioer-count数量正确初始化对应数量的io工作者
    for (int i = 0; i < ioer_count; i++) {
        state.ioers.emplace_back(std::make_unique<IOuringManager>());
    }
    return state;
}

const std::vector<hnswlib::labeltype>* FetchGroundTruth(
    const HnswIoState& state,
    size_t row,
    std::vector<hnswlib::labeltype>& scratch)
{
    if (state.gt_reader) {
        auto gt_res = state.gt_reader->getTopKResults(row);
        scratch.clear();
        scratch.reserve(gt_res.size());
        for (auto id : gt_res) {
            scratch.push_back(static_cast<hnswlib::labeltype>(id));
        }
        return &scratch;
    }
    if (row < state.gt_preloaded.size()) {
        return &state.gt_preloaded[row];
    }
    return nullptr;
}

std::string BuildOutputDir(const HnswIoState& state,
                           const std::string& result_folder_name)
{
    const auto& pc = *state.pc;
    const std::string root = EnvOrDefault("ANNFS_END_RESULTS_ROOT", "./json_results");
    const std::string run_id = EnvOrDefault("ANNFS_END_RESULTS_RUN_ID", UtcRunStamp());
    const std::string exec_name = EnvOrDefault("ANNFS_EXEC_NAME", result_folder_name);
    const std::string variant = EnvOrDefault("ANNFS_END_RESULTS_VARIANT", "");
    std::string exec_token = exec_name;
    if (!variant.empty()) {
        exec_token += "_" + variant;
    }

    fs::path out_dir = fs::path(root) /
                       SanitizePathToken(run_id) /
                       SanitizePathToken(exec_token);
    out_dir /= SanitizePathToken(state.args.dataset_name);
    out_dir /= "t" + std::to_string(state.num_threads);
    out_dir /= "efS" + std::to_string(pc.dataset.search_ef) +
               "_qcnt" + std::to_string(state.query_cnt) +
               "_rpt_" + std::to_string(state.args.repeat_id);
    fs::create_directories(out_dir);
    return out_dir.string();
}

IOuringManager& IoWorkerForThread(const HnswIoState& state,
                                  size_t thread_id)
{
    if (state.ioers.empty()) {
        throw std::out_of_range("No IO managers available");
    }
    if (state.use_shared_io_worker) {
        return *(state.ioers[0]);
    }
    if (thread_id >= state.ioers.size()) {
        throw std::out_of_range("Invalid IO manager index");
    }
    return *(state.ioers[thread_id]);
}

IOuringManager& IoManagerForThread(const HnswIoState& state, size_t thread_id)
{
    return IoWorkerForThread(state, thread_id);
}

std::vector<off_t> BuildIoOffsetsFromLabels(
    const std::vector<unsigned long long>& labels,
    const HnswIoState& state)
{
    std::vector<off_t> offsets;
    offsets.reserve(labels.size());

    const std::uint32_t block_size = state.cfg->LOGIC_BLOCK_SIZE;
    if (block_size == 0) {
        throw std::runtime_error("LOGIC_BLOCK_SIZE is zero");
    }

    std::unordered_set<std::uint64_t> dedup_offsets;
    dedup_offsets.reserve(labels.size() * 2 + 1);
    for (auto id : labels) {
        auto dec = annfs::io::DecodePacked64(
            static_cast<std::uint64_t>(id), block_size);
        for (std::uint32_t i = 0; i < dec.blocks_to_read; ++i) {
            const std::uint64_t off = dec.block_base + static_cast<std::uint64_t>(i) * block_size;
            if (dedup_offsets.insert(off).second) {
                offsets.emplace_back(static_cast<off_t>(off));
            }
        }
    }
    return offsets;
}

int ResolveRunIndex(const HnswIoCliArgs& args)
{
    /**
        根据当前 数据集的运行实验的参数信息 组合 此次 run的 index label
    */
    if (args.override_run_index) {
        return *args.override_run_index;
    }
    const char* hint = std::getenv("ANNFS_RUN_INDEX_HINT");
    if (hint && *hint) {
        char* endptr = nullptr;
        long value = std::strtol(hint, &endptr, 10);
        if (endptr != hint) {
            return static_cast<int>(value);
        }
    }
    return 0;
}

} // namespace annfs::tests
