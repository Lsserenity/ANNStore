#include "tests/HnswFsDataset.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <unordered_map>

#include "common/data_io.hpp"
#include "common/gt_reader.hpp"
#include "common/json_utils.hpp"
#include "tests/HnswIoCommon.hpp"

namespace fs = std::filesystem;

namespace annfs::tests {
namespace {

std::mutex g_cache_mutex;
std::unordered_map<std::string, std::unique_ptr<FsDatasetResources>> g_dataset_cache;

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

std::unique_ptr<hnswlib::SpaceInterface<float>> create_space(
    int dim,
    const DatasetInfo& dataset)
{
    if (dataset.data_type == "uint8" || dataset.data_type == "u8") {
        return std::make_unique<hnswlib::L2SpaceUint8>(dim);
    }
    if (dataset.distance_metric == "ip") {
        return std::make_unique<hnswlib::InnerProductSpace>(dim);
    }
    return std::make_unique<hnswlib::L2Space>(dim);
}

GroundTruthAccessor PrepareGroundTruthAccessor(
    const ParamConfig& pc,
    size_t dataset_size,
    const std::shared_ptr<spdlog::logger>& logger)
{
    GroundTruthAccessor accessor;
    const std::string& gt_path = pc.dataset.gt_directory;
    if (gt_path.empty()) {
        throw std::runtime_error("GT path is empty in config.");
    }

    std::error_code ec;
    bool is_file = std::filesystem::is_regular_file(gt_path, ec);
    ec.clear();
    bool is_dir = !is_file && std::filesystem::is_directory(gt_path, ec);
    if (is_file) {
        unsigned gtN = 0;
        unsigned gtDim = 0;
        const auto format = infer_gt_format(pc.dataset.gt_format, gt_path);
        auto gt_unsigned = annfs::common::read_gt(
            gt_path,
            format.empty() ? "ibin" : format,
            gtN,
            gtDim);
        if (gt_unsigned.empty()) {
            throw std::runtime_error(
                "Ground truth file is empty: " + gt_path);
        }
        accessor.cached_gt.reserve(gt_unsigned.size());
        for (const auto& row : gt_unsigned) {
            accessor.cached_gt.emplace_back(row.begin(), row.end());
        }
        accessor.available_rows = accessor.cached_gt.size();
        if (logger) {
            logger->info("Loaded {} GT rows from {}", accessor.available_rows, gt_path);
        }
    } else if (is_dir) {
        accessor.streaming_reader =
            std::make_unique<annfs::common::GroundTruthReader>(
                gt_path,
                pc.gt_batch_size,
                dataset_size,
                pc.gt_gt_k);
        accessor.available_rows = dataset_size;
        if (logger) {
            logger->info("Using GT directory {} with batch size {}", gt_path, pc.gt_batch_size);
        }
    } else {
        throw std::runtime_error(
            "Ground truth path does not exist: " + gt_path);
    }

    return accessor;
}

FsDatasetResources BuildDatasetResources(const std::string& config_path,
                                         const nlohmann::json& root,
                                         ParamConfig& pc,
                                         GlobalConfig& cfg)
{
    if (pc.fs_data_dir_path.empty()) {
        throw std::runtime_error(
            "fs_data_dir_path is empty in config for dataset " +
            pc.dataset.dataset_name);
    }

    const bool use_uint8 =
        pc.dataset.data_type == "uint8" || pc.dataset.data_type == "u8";
    const std::string data_format =
        pc.dataset.data_format.empty() ? "bin" : pc.dataset.data_format;

    unsigned N = 0;
    unsigned Dim = 0;
    std::unique_ptr<float[]> raw_float;
    std::unique_ptr<uint8_t[]> raw_u8;

    auto is_bin_format = [](const std::string& fmt) {
        return fmt == "bin" || fmt == "fbin" || fmt == "bbin" ||
               fmt == "ibin" || fmt == "u8bin";
    };
    auto is_vecs_format = [](const std::string& fmt) {
        return fmt == "vecs" || fmt == "fvecs" || fmt == "bvecs" ||
               fmt == "u8vecs";
    };

    try {
        if (use_uint8) {
            if (is_bin_format(data_format)) {
                raw_u8.reset(
                    annfs::common::read_data_bin<uint8_t>(
                        pc.dataset.data_path,
                        N,
                        Dim));
            } else if (is_vecs_format(data_format)) {
                raw_u8.reset(
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
            raw_float.reset(
                annfs::common::read_data(
                    pc.dataset.data_path,
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

    FsDatasetResources resources;
    resources.use_uint8_data = use_uint8;
    resources.dim = static_cast<int>(Dim);
    if (use_uint8) {
        resources.vecs_uint8.reserve(N);
        for (unsigned i = 0; i < N; ++i) {
            resources.vecs_uint8.emplace_back(
                raw_u8.get() + static_cast<uint64_t>(i) * Dim,
                raw_u8.get() + static_cast<uint64_t>(i + 1) * Dim);
        }
    } else {
        resources.vecs_float.reserve(N);
        for (unsigned i = 0; i < N; ++i) {
            resources.vecs_float.emplace_back(
                raw_float.get() + static_cast<uint64_t>(i) * Dim,
                raw_float.get() + static_cast<uint64_t>(i + 1) * Dim);
        }
        annfs::common::normalize_rows(resources.vecs_float);
    }

    resources.vector_count = use_uint8
        ? resources.vecs_uint8.size()
        : resources.vecs_float.size();

    resources.space_holder = create_space(resources.dim, pc.dataset);
    resources.hnsw_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        resources.space_holder.get(),
        pc.dataset.index_path);
    resources.hnsw_index->setEf(pc.dataset.search_ef);

    resources.fs_fanout = root.value("fs_fanout", 100);
    size_t max_elements = resources.hnsw_index->getMaxElements();
    resources.fs_mapper_max_elements = max_elements;
    resources.fs_mapper_file_limit = GlobalConfig::instance().FS_MAPPER_FILE_LIMIT;
    resources.fs_round_size = GlobalConfig::instance().FS_ROUND_SIZE;
    if (cfg.logger) {
        cfg.logger->info(
            "[FsCache] dataset={} vectors={} dim={} hnsw_max={} fs_limit={} fs_round_size={}",
            pc.dataset.dataset_name,
            resources.vector_count,
            resources.dim,
            max_elements,
            resources.fs_mapper_file_limit,
            resources.fs_round_size);
    }

    resources.path_mapper = std::make_unique<annfs::common::IndexToPathMapper>(
        pc.fs_data_dir_path,
        resources.fs_mapper_max_elements,
        resources.fs_fanout,
        GlobalConfig::instance().FS_MAPPER_FILE_LIMIT,
        GlobalConfig::instance().FS_ROUND_SIZE);

    // NOTE:
    // Sample path existence validation is intentionally disabled for
    // open-vs-offset IO analysis to avoid injecting extra metadata lookups.
    if (cfg.logger) {
        cfg.logger->info(
            "[FsCache] Sample path validation disabled for this build");
    }

    resources.gt_accessor = std::make_shared<GroundTruthAccessor>(
        PrepareGroundTruthAccessor(pc, resources.vector_count, cfg.logger));

    return resources;
}

std::string BuildCacheKey(const std::string& config_path,
                          const std::string& dataset_name)
{
    return config_path + "::" + dataset_name;
}

} // namespace

std::vector<int64_t> GroundTruthAccessor::fetch(size_t row) const
{
    if (streaming_reader) {
        return streaming_reader->getTopKResults(row);
    }
    if (row >= cached_gt.size()) {
        throw std::out_of_range("GT row out of range");
    }
    return cached_gt[row];
}

const void* FsDatasetResources::QueryPtr(size_t row) const
{
    if (use_uint8_data) {
        return static_cast<const void*>(vecs_uint8[row].data());
    }
    return static_cast<const void*>(vecs_float[row].data());
}

FsDatasetResources& LoadOrGetFsDataset(const std::string& config_path,
                                       const nlohmann::json& root,
                                       ParamConfig& pc,
                                       GlobalConfig& cfg)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    const std::string key = BuildCacheKey(config_path, pc.dataset.dataset_name);
    auto it = g_dataset_cache.find(key);
    if (it != g_dataset_cache.end()) {
        return *(it->second);
    }

    auto entry = std::make_unique<FsDatasetResources>(
        BuildDatasetResources(config_path, root, pc, cfg));
    auto* ptr = entry.get();
    g_dataset_cache.emplace(key, std::move(entry));
    return *ptr;
}

void ReleaseFsDataset(const std::string& config_path,
                      const std::string& dataset_name)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    const std::string key = BuildCacheKey(config_path, dataset_name);
    g_dataset_cache.erase(key);
}

void ClearFsDatasetCache()
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_dataset_cache.clear();
}

} // namespace annfs::tests
