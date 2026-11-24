#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <json.hpp>

#include "tests/HnswExperimentBatch.hpp"
#include "tests/HnswIoCommon.hpp"

namespace annfs::tests::fsrunner {

inline long long epoch_ms()
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
}

inline std::string batch_marker_path()
{
    const char* path = std::getenv("ANNFS_BATCH_MARKER_PATH");
    return path ? std::string(path) : std::string();
}

inline void append_batch_marker(const std::string& path,
                                const nlohmann::json& entry)
{
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << entry.dump() << "\n";
    }
}

inline void emit_batch_marker(const std::string& path,
                              const nlohmann::json& entry)
{
    std::cout << "[BATCH_MARKER] " << entry.dump() << std::endl;
    append_batch_marker(path, entry);
}

inline void ApplyRunOverrides(const annfs::tests::HnswIoCliArgs& args,
                              const annfs::tests::ExperimentBatchRunSpec* spec,
                              ParamConfig& pc)
{
    if (spec) {
        if (spec->num_threads) {
            pc.num_threads = *spec->num_threads;
        }
        if (spec->search_ef) {
            pc.dataset.search_ef = *spec->search_ef;
        }
        if (spec->io_depths) {
            GlobalConfig::instance().RING_DEPTH = *spec->io_depths;
        }
        if (spec->query_cnt) {
            pc.query_cnt = *spec->query_cnt;
        }
    } else if (args.override_num_threads) {
        pc.num_threads = *args.override_num_threads;
    }
}

} // namespace annfs::tests::fsrunner
