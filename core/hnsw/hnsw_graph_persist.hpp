#pragma once

#include <hnswlib/hnswalg.h>

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hnswlib {

template <typename dist_t>
struct HNSWNodeRelation {
    tableint internal_id;
    labeltype label;
    int level;
    std::vector<tableint> neighbor_ids;
    std::vector<labeltype> neighbor_labels;
};

template <typename dist_t>
inline std::vector<HNSWNodeRelation<dist_t>> collectNodeRelationsAtLevel(
    HierarchicalNSW<dist_t>& index,
    int requested_level) {
    if (index.cur_element_count == 0) {
        return {};
    }

    const int level = (requested_level < 0) ? index.maxlevel_ : requested_level;
    if (level < 0 || level > index.maxlevel_) {
        throw std::invalid_argument("requested_level is out of range");
    }

    const size_t count = index.cur_element_count;
    std::vector<HNSWNodeRelation<dist_t>> relations;
    relations.reserve(count);

    for (tableint node = 0; node < count; ++node) {
        if (level > index.element_levels_[node]) {
            continue;
        }

        HNSWNodeRelation<dist_t> rel;
        rel.internal_id = node;
        rel.label = index.getExternalLabel(node);
        rel.level = level;

        std::unique_lock<std::mutex> lock(index.link_list_locks_[node]);
        linklistsizeint* ll = index.get_linklist_at_level(node, level);
        const size_t degree = index.getListCount(ll);
        tableint* neighbors = reinterpret_cast<tableint*>(ll + 1);

        rel.neighbor_ids.reserve(degree);
        rel.neighbor_labels.reserve(degree);
        for (size_t i = 0; i < degree; ++i) {
            const tableint nb = neighbors[i];
            rel.neighbor_ids.push_back(nb);
            rel.neighbor_labels.push_back(index.getExternalLabel(nb));
        }

        relations.push_back(std::move(rel));
    }

    return relations;
}

template <typename dist_t>
inline bool dumpLevelRelationsToFile(
    HierarchicalNSW<dist_t>& index,
    int requested_level,
    const std::string& file_path) {
    const auto relations = collectNodeRelationsAtLevel(index, requested_level);

    const int level = (index.cur_element_count == 0)
        ? 0
        : ((requested_level < 0) ? index.maxlevel_ : requested_level);

    std::ofstream out(file_path);
    if (!out.is_open()) {
        return false;
    }

    out << "# hnsw_node_relations\n";
    out << "# level=" << level << " node_count=" << relations.size() << "\n";
    out << "# format: NODE <id> <label> <level> | NEIGHBORS <n> <nid:label>...\n";

    for (const auto& rel : relations) {
        out << "NODE " << rel.internal_id << ' ' << rel.label << ' ' << rel.level << "\n";
        out << "NEIGHBORS " << rel.neighbor_ids.size();
        for (size_t i = 0; i < rel.neighbor_ids.size(); ++i) {
            out << ' ' << rel.neighbor_ids[i] << ':' << rel.neighbor_labels[i];
        }
        out << "\n";
    }

    return true;
}

}  // namespace hnswlib

