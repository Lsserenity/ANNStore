#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../config/param_config.hpp"
#include "hnswlib/hnswalg.h"
#include "hnswlib/space_l2.h"
#include "hnswlib/space_ip.h"
#include "json.hpp"
#include "hnsw/uint8.hpp"

namespace {

int ReadVectorDimFromDataFile(const std::string& data_path, const std::string& data_format) {
    std::ifstream in(data_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open data file: " + data_path);
    }

    std::uint32_t first = 0;
    std::uint32_t second = 0;

    if (data_format == "bin" || data_format == "fbin" || data_format == "bbin") {
        in.read(reinterpret_cast<char*>(&first), sizeof(first));
        in.read(reinterpret_cast<char*>(&second), sizeof(second));
        if (!in.good() || second == 0) {
            throw std::runtime_error("invalid bin header for data file: " + data_path);
        }
        return static_cast<int>(second);
    }

    if (data_format == "vecs" || data_format == "fvecs" || data_format == "bvecs") {
        in.read(reinterpret_cast<char*>(&first), sizeof(first));
        if (!in.good() || first == 0) {
            throw std::runtime_error("invalid vecs header for data file: " + data_path);
        }
        return static_cast<int>(first);
    }

    throw std::runtime_error("unsupported data_format: " + data_format);
}

std::unique_ptr<hnswlib::SpaceInterface<float>> CreateSpace(const DatasetInfo& ds, int dim) {
    if (ds.data_type == "uint8" || ds.data_type == "u8") {
        return std::make_unique<hnswlib::L2SpaceUint8>(dim);
    }
    if (ds.distance_metric == "ip") {
        return std::make_unique<hnswlib::InnerProductSpace>(dim);
    }
    return std::make_unique<hnswlib::L2Space>(dim);
}

ParamConfig& LoadConfigAndDataset(const std::string& config_path, const std::string& dataset_name) {
    std::ifstream in(config_path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open config json: " + config_path);
    }

    nlohmann::json js;
    in >> js;

    ParamConfig& pc = ParamConfig::instance();
    pc.load_from_json(js, dataset_name);
    return pc;
}

void DumpInternalIdLabels(
    hnswlib::HierarchicalNSW<float>& index,
    const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open dump file: " + path);
    }
    out << "# internal_id label\n";
    for (hnswlib::tableint i = 0; i < index.cur_element_count; ++i) {
        out << i << ' ' << index.getExternalLabel(i) << "\n";
    }
}

std::vector<std::pair<hnswlib::tableint, hnswlib::labeltype>>
LoadInternalRelabelMap(const std::string& mapping_path) {
    std::ifstream in(mapping_path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open mapping file: " + mapping_path);
    }

    std::vector<std::pair<hnswlib::tableint, hnswlib::labeltype>> updates;
    std::unordered_set<hnswlib::tableint> seen_internal;
    std::string line;
    size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        unsigned long long internal = 0;
        unsigned long long label = 0;
        if (!(iss >> internal >> label)) {
            throw std::runtime_error(
                "invalid mapping format at line " + std::to_string(lineno) +
                ", expected: <internal_id> <new_label>");
        }
        hnswlib::tableint iid = static_cast<hnswlib::tableint>(internal);
        if (!seen_internal.insert(iid).second) {
            throw std::runtime_error(
                "duplicate internal_id in mapping at line " + std::to_string(lineno));
        }
        updates.emplace_back(iid, static_cast<hnswlib::labeltype>(label));
    }
    return updates;
}

void ApplyRelabelByInternalId(
    hnswlib::HierarchicalNSW<float>& index,
    const std::vector<std::pair<hnswlib::tableint, hnswlib::labeltype>>& updates) {
    for (const auto& kv : updates) {
        if (kv.first >= index.cur_element_count) {
            throw std::runtime_error(
                "internal_id out of range: " + std::to_string(kv.first));
        }
    }

    for (const auto& kv : updates) {
        index.setExternalLabel(kv.first, kv.second);
    }

    std::unordered_map<hnswlib::labeltype, hnswlib::tableint> rebuilt;
    rebuilt.reserve(index.cur_element_count * 2 + 1);
    for (hnswlib::tableint i = 0; i < index.cur_element_count; ++i) {
        hnswlib::labeltype lbl = index.getExternalLabel(i);
        auto ok = rebuilt.emplace(lbl, i);
        if (!ok.second) {
            throw std::runtime_error(
                "duplicate label detected after relabel: " + std::to_string(lbl) +
                " (internal_id=" + std::to_string(i) +
                ", existing_internal_id=" + std::to_string(ok.first->second) + ")");
        }
    }

    {
        std::lock_guard<std::mutex> lk(index.label_lookup_lock);
        index.label_lookup_.swap(rebuilt);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <config_json> <dataset_name> <mapping_txt|-> <output_index> [dump_before] [dump_after]\n";
        std::cerr << "mapping line format: <internal_id> <new_label>\n";
        std::cerr << "pass mapping_txt='-' to skip relabel and only dump mapping files\n";
        return 1;
    }

    try {
        const std::string config_path = argv[1];
        const std::string dataset_name = argv[2];
        const std::string mapping_path = argv[3];
        const std::string output_index = argv[4];
        const std::string dump_before =
            (argc >= 6) ? std::string(argv[5])
                        : ("./internal_label_before_" + dataset_name + ".txt");
        const std::string dump_after =
            (argc >= 7) ? std::string(argv[6])
                        : ("./internal_label_after_" + dataset_name + ".txt");

        ParamConfig& pc = LoadConfigAndDataset(config_path, dataset_name);
        const DatasetInfo& ds = pc.dataset;

        const int dim = ReadVectorDimFromDataFile(ds.data_path, ds.data_format);
        auto space = CreateSpace(ds, dim);

        hnswlib::HierarchicalNSW<float> index(space.get(), ds.hnsw_index_path);
        DumpInternalIdLabels(index, dump_before);

        if (mapping_path != "-") {
            const auto updates = LoadInternalRelabelMap(mapping_path);
            ApplyRelabelByInternalId(index, updates);
        }

        DumpInternalIdLabels(index, dump_after);
        index.saveIndex(output_index);

        std::cout << "dataset        : " << ds.dataset_name << "\n";
        std::cout << "source_index   : " << ds.hnsw_index_path << "\n";
        std::cout << "output_index   : " << output_index << "\n";
        std::cout << "dump_before    : " << dump_before << "\n";
        std::cout << "dump_after     : " << dump_after << "\n";
        std::cout << "mapping_file   : " << mapping_path << "\n";
        std::cout << "done.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 3;
    }
}
