#pragma once

#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace annfs::common {

/**
 * @brief 计算两个结果集的 recall
 * @param r1 - 第一个结果集（ground truth 或搜索结果）
 * @param r2 - 第二个结果集（搜索结果 或 ground truth）
 * @param K - Top-K 数量
 * @return recall 值 (0.0 ~ 1.0)
 */
template <typename T, typename F>
float get_recall(const std::vector<T>& r1, const std::vector<F>& r2, int K)
{
    if (r1.size() < static_cast<size_t>(K) || r2.size() < static_cast<size_t>(K)) {
        return 0.0f;
    }
    
    std::set<T> a(r1.begin(), r1.begin() + K);
    std::set<T> b(r2.begin(), r2.begin() + K);
    std::set<T> result;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), 
                         std::inserter(result, result.begin()));
    return static_cast<float>(result.size()) / static_cast<float>(a.size());
}

template <typename IssuedT, typename GtT>
inline double overlap_prefetch_recall_at_k(
    const std::unordered_set<IssuedT>& issued_set,
    const std::vector<GtT>& gt_row,
    int k)
{
    if (k <= 0) {
        return 0.0;
    }
    int hit = 0;
    const int limit = std::min<int>(k, static_cast<int>(gt_row.size()));
    for (int i = 0; i < limit; ++i) {
        if (issued_set.find(static_cast<IssuedT>(gt_row[i])) != issued_set.end()) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

/**
 * @brief 计算两个结果集的 recall（旧版接口，兼容性）
 * @param r1 - 第一个结果集
 * @param r2 - 第二个结果集
 * @param K - Top-K 数量
 * @return recall 值 (0.0 ~ 1.0)
 */
template <typename T>
inline float get_recall(const std::vector<T>& r1, const std::vector<T>& r2, unsigned K)
{
    if (r1.size() < K || r2.size() < K)
        throw std::runtime_error("error@get_recall: groundtruth or result is not enough.");
    std::set<T> a(r1.begin(), r1.begin() + K);
    std::set<T> b(r2.begin(), r2.begin() + K);
    std::set<T> result;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), 
                         std::inserter(result, result.begin()));
    return static_cast<float>(result.size()) / K;
}

/**
 * @brief 批量计算 recall（返回百分比）
 * @param query_num - 查询数量
 * @param K - Top-K 数量
 * @param gt - Ground truth 结果
 * @param res - 搜索结果
 * @return recall 百分比 (0.0 ~ 100.0)
 */
template <typename T>
inline float compute_recall(unsigned query_num, unsigned K, 
                            const std::vector<std::vector<T>> &gt, 
                            const std::vector<std::vector<T>> &res)
{
    if (query_num > gt.size() || query_num > res.size())
        throw std::runtime_error("error@compute_recall");
    float recall = 0.0;
    for (size_t i = 0; i < (size_t)query_num; i++)
    {
        recall += get_recall(gt[i], res[i], K);
    }
    return recall * 100.0f / query_num;
}

} // namespace annfs::common
