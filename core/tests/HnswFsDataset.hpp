#pragma once

#include <memory>
#include <string>
#include <vector>

#include <hnswlib/hnswlib.h>
#include <hnsw/uint8.hpp>
#include <json.hpp>

#include "common/fs_path_mapper.hpp"
#include "global_config.hpp"
#include "param_config.hpp"

namespace annfs::common {
class GroundTruthReader;
}

namespace annfs::tests {

struct GroundTruthAccessor {
    std::unique_ptr<annfs::common::GroundTruthReader> streaming_reader;
    std::vector<std::vector<int64_t>> cached_gt;
    size_t available_rows = 0;

    std::vector<int64_t> fetch(size_t row) const;
};

struct FsDatasetResources {
    bool use_uint8_data = false;
    int dim = 0;
    size_t vector_count = 0;
    std::vector<std::vector<float>> vecs_float;
    std::vector<std::vector<uint8_t>> vecs_uint8;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_holder;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index;
    std::shared_ptr<GroundTruthAccessor> gt_accessor;
    std::unique_ptr<annfs::common::IndexToPathMapper> path_mapper;
    size_t fs_mapper_max_elements = 0;
    size_t fs_mapper_file_limit = 0;
    int fs_fanout = 100;
    size_t fs_round_size = 1000000;

    const void* QueryPtr(size_t row) const;
};

FsDatasetResources& LoadOrGetFsDataset(const std::string& config_path,
                                       const nlohmann::json& root,
                                       ParamConfig& pc,
                                       GlobalConfig& cfg);

void ReleaseFsDataset(const std::string& config_path,
                      const std::string& dataset_name);

void ClearFsDatasetCache();

} // namespace annfs::tests
