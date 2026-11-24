#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hnswlib/hnswlib.h>

#include "param_config.hpp"
#include "global_config.hpp"
#include "gt_reader.hpp"
#include "iouring_manager.hpp"
#include "common/parallel.hpp"

class IOuringManager;

namespace annfs::tests {

class QueryLatencyTracker;

enum class OffsetDecodeMode {
    PackedLabel,
    ShiftOffbits,
};

enum class VectorDataType {
    Float32,
    Uint8,
};

struct HnswIoCliArgs {
    // 实验运行时的 CLI 参数
    std::string config_json_path;
    std::string dataset_name;
    int repeat_id;
    std::optional<int> override_num_threads;
    std::optional<int> override_run_index;
    std::optional<int> override_stable_iter;
    std::optional<int> override_stable_hops;
    std::optional<int> override_search_ef;
    bool enable_query_timeline = true;
    // offset decode mode for OffsetIO-family strategies.
    // packed_label: decode high-3/low-29 from returned label;
    // shift_offbits: legacy (label << OFF_BITS_LEN).
    OffsetDecodeMode offset_decode_mode = OffsetDecodeMode::PackedLabel;
};

struct HnswIoState {
    /**
        专门用于封装 HNSWIO实验所需的配置信息
        1. cli参数
        2. global config
        3. param config
        4. hnsw space index
        
    */
    HnswIoCliArgs args;
    GlobalConfig* cfg = nullptr;
    ParamConfig* pc = nullptr;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index;
    std::unique_ptr<annfs::common::GroundTruthReader> gt_reader;
    std::vector<std::vector<hnswlib::labeltype>> gt_preloaded;
    VectorDataType vec_data_type = VectorDataType::Float32;
    std::vector<std::vector<float>> vecs_float;
    std::vector<std::vector<uint8_t>> vecs_uint8;
    std::vector<std::unique_ptr<IOuringManager>> ioers;
    bool use_shared_io_worker = false;
    int query_cnt = 0;
    int num_threads = 0;
    int base_stable_iter = 0;

    const void* QueryVector(size_t row) const
    {
        if (vec_data_type == VectorDataType::Uint8) {
            return static_cast<const void*>(vecs_uint8[row].data());
        }
        return static_cast<const void*>(vecs_float[row].data());
    }

    size_t VectorCount() const
    {
        return vec_data_type == VectorDataType::Uint8
            ? vecs_uint8.size()
            : vecs_float.size();
    }
};

std::optional<HnswIoCliArgs> ParseCliArgs(int argc, char** argv);

HnswIoState PrepareState(const HnswIoCliArgs& args);

const std::vector<hnswlib::labeltype>* FetchGroundTruth(
    const HnswIoState& state,
    size_t row,
    std::vector<hnswlib::labeltype>& scratch);

template <typename Fn>
void ParallelForQueries(const HnswIoState& state, Fn&& fn);

std::string BuildOutputDir(const HnswIoState& state,
                           const std::string& result_folder_name);

IOuringManager& IoWorkerForThread(const HnswIoState& state, size_t thread_id);
IOuringManager& IoManagerForThread(const HnswIoState& state, size_t thread_id);
std::vector<off_t> BuildIoOffsetsFromLabels(
    const std::vector<unsigned long long>& labels,
    const HnswIoState& state);

int ResolveRunIndex(const HnswIoCliArgs& args);

void PersistQueryTimeline(const std::string& out_dir,
                          const QueryLatencyTracker& tracker,
                          const GlobalConfig* cfg);

class ScopedQueryListenerBinding {
public:
    ScopedQueryListenerBinding(
        std::vector<std::unique_ptr<IOuringManager>>& ioers,
        annfs::io::QueryIoEventListener* listener);
    ~ScopedQueryListenerBinding();

    ScopedQueryListenerBinding(const ScopedQueryListenerBinding&) = delete;
    ScopedQueryListenerBinding& operator=(const ScopedQueryListenerBinding&) = delete;

private:
    std::vector<std::unique_ptr<IOuringManager>>* manager_ioers_ = nullptr;
    annfs::io::QueryIoEventListener* listener_;
};

template <typename Fn>
void ParallelForQueries(const HnswIoState& state, Fn&& fn)
{
    if (!state.pc) {
        throw std::runtime_error("HnswIoState not initialized");
    }
    size_t query_cnt = static_cast<size_t>(state.query_cnt);
    switch (state.pc->parallel_mode) {
        case 0:
            OptimizedParallelFor(0, query_cnt, state.num_threads,
                                 std::forward<Fn>(fn));
            break;
        case 1:
            WorkStealingParallelFor(0, query_cnt, state.num_threads,
                                    std::forward<Fn>(fn));
            break;
        case 2:
        default:
            StaticPartitionParallelFor(0, query_cnt, state.num_threads,
                                       std::forward<Fn>(fn));
            break;
    }
}

} // namespace annfs::tests
