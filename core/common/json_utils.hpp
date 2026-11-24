#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <json.hpp>
#include "../../core/config/global_config.hpp"

namespace annfs::common {

using json = nlohmann::json;

/**
 * @brief 读取 JSON 文件
 * @param path - JSON 文件路径
 * @return JSON 对象
 */
inline json read_json(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("cannot open: " + path);

    json j;
    file >> j;
    return j;
}

/**
 * @brief 保存 hop_id_lists 到 JSON 文件
 * @param hop_id_lists - 每个查询的 hop_id 列表（变长数组）
 * @param output_path - 输出文件路径
 * @param query_cnt - 查询数量（可选，如果不提供则使用 hop_id_lists.size()）
 */
inline void save_hop_id_lists(const std::vector<std::vector<int>>& hop_id_lists, 
                               const std::string& output_path, 
                               size_t query_cnt = 0)
{
    size_t actual_query_cnt = (query_cnt > 0) ? query_cnt : hop_id_lists.size();

    json entries = json::array();
    size_t total_hops = 0;
    size_t max_hops = 0;
    size_t min_hops = (hop_id_lists.empty() || hop_id_lists[0].empty()) ? 0 : hop_id_lists[0].size();

    for (size_t i = 0; i < actual_query_cnt && i < hop_id_lists.size(); ++i) {
        json entry;
        entry["query_id"] = i;
        entry["hop_ids"] = hop_id_lists[i];
        entry["hop_count"] = hop_id_lists[i].size();
        entries.push_back(entry);

        total_hops += hop_id_lists[i].size();
        if (hop_id_lists[i].size() > max_hops) {
            max_hops = hop_id_lists[i].size();
        }
        if (hop_id_lists[i].size() < min_hops) {
            min_hops = hop_id_lists[i].size();
        }
    }

    json summary;
    summary["total_queries"] = actual_query_cnt;
    summary["total_hops"] = total_hops;
    summary["avg_hops_per_query"] = (actual_query_cnt > 0) ? (double)total_hops / actual_query_cnt : 0.0;
    summary["max_hops"] = max_hops;
    summary["min_hops"] = min_hops;

    json data = {
        {"summary", summary},
        {"entries", entries}
    };

    std::ofstream fout(output_path);
    if (!fout.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + output_path);
    }
    fout << data.dump(4);
    fout.close();
}

/**
 * @brief 生成多项指标的 JSON 文件
 * @param costs - 每个查询的代价数据
 * @param name_labels - 代价名称标签
 * @param searchcnt - 查询数量
 * @param output_path - 输出文件路径
 */
template <typename T>
nlohmann::json generate_json_multi_T(const std::vector<std::vector<T>>& costs,
                           const std::vector<std::string>& name_labels,
                           const int searchcnt, 
                           const std::string& output_path)
{
    int comp_size = costs.size();
    assert(comp_size == name_labels.size());
    std::vector<double> avgcosts(comp_size, 0);
    nlohmann::json entries = nlohmann::json::array();
    
    for (int i = 0; i < searchcnt; i++) {
        nlohmann::json entry;
        for (int j = 0; j < comp_size; j++) {
            entry[name_labels[j]] = costs[j][i];
            avgcosts[j] += costs[j][i];
        }
        entries.push_back(entry);
    }
    
    nlohmann::json avgcostjson;
    for (int j = 0; j < comp_size; j++) {
        avgcostjson["avg_" + name_labels[j]] = avgcosts[j] / searchcnt;
    }
    
    nlohmann::json data = {
        {"avgcost", avgcostjson},
        {"entries", entries}
    };
    // 如果存在logger则会打印信息 === hnswoffset实验都会正确打印
    if (GlobalConfig::instance().logger) {
        GlobalConfig::instance().logger->info("avgcost: {}", avgcostjson.dump());
    }
    
    std::ofstream fout(output_path);
    fout << data.dump(4);
    fout.close();
    return avgcostjson;
}

} // namespace annfs::common

