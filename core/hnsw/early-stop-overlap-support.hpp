#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "param_config.hpp"
#include "hnswlib/hnswalg.h"
#include "iouring_manager.hpp"

namespace annfs::hnsw {

struct EarlyStopOverlapSignals {
    bool early_stop_triggered = false;
    bool overlap_prefetch_triggered = false;
    int pending_prefetch_ios = 0;
    std::size_t issued_prefetch_labels = 0;
    std::size_t topk_labels = 0;
    std::unordered_set<unsigned long long> issued_label_set;
};

template <typename dist_t>
using HnswInternalQueue = std::priority_queue<
    std::pair<dist_t, hnswlib::tableint>,
    std::vector<std::pair<dist_t, hnswlib::tableint>>,
    typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst>;

template <typename dist_t>
struct CompareEarlyOverlapTupleByFirst {
    bool operator()(const std::tuple<dist_t, hnswlib::tableint, int>& a,
                    const std::tuple<dist_t, hnswlib::tableint, int>& b) const
    {
        return std::get<0>(a) < std::get<0>(b);
    }
};

template <typename dist_t, bool bare_bone_search = true, bool collect_metrics = false>
HnswInternalQueue<dist_t> SearchBaseLayerSTEarlyStopOverlapMapped(
    const hnswlib::HierarchicalNSW<dist_t>& index,
    hnswlib::tableint ep_id,
    const void* data_point,
    std::size_t ef,
    std::size_t k,
    int stable_hops,
    int stableiter,
    IOuringManager* ioer,
    int least_else_size,
    int query_id,
    EarlyStopOverlapSignals* signals,
    bool collect_issued_labels = false,
    hnswlib::BaseFilterFunctor* isIdAllowed = nullptr,
    hnswlib::BaseSearchStopCondition<dist_t>* stop_condition = nullptr)
{
    auto* vl = index.visited_list_pool_->getFreeVisitedList();
    auto* visited_array = vl->mass;
    auto visited_array_tag = vl->curV;

    HnswInternalQueue<dist_t> only_top_candidates;
    HnswInternalQueue<dist_t> else_only_top_candidates;
    HnswInternalQueue<dist_t> top_candidates;
    std::priority_queue<
        std::tuple<dist_t, hnswlib::tableint, int>,
        std::vector<std::tuple<dist_t, hnswlib::tableint, int>>,
        CompareEarlyOverlapTupleByFirst<dist_t>>
        candidate_set;

    int pending_prefetch = 0;
    std::size_t issued_prefetch_labels = 0;
    std::size_t mapped_missing = 0;
    std::unordered_set<unsigned long long> issued_label_set;

    auto issue_prefetch = [&](const std::vector<unsigned long long>& labels) {
        if (!ioer || labels.empty()) {
            return std::size_t{0};
        }
        std::size_t missing = 0;
        std::size_t issued = 0;
        const int pending = ioer->mapped_prefetch(
            labels,
            ParamConfig::instance(),
            query_id,
            false,
            &missing,
            &issued);
        pending_prefetch += std::max(0, pending);
        mapped_missing += missing;
        (void)issued;
        issued_prefetch_labels += labels.size();
        return labels.size();
    };

    dist_t lowerBound;
    if (bare_bone_search ||
        (!index.isMarkedDeleted(ep_id) &&
         ((!isIdAllowed) || (*isIdAllowed)(index.getExternalLabel(ep_id))))) {
        char* ep_data = index.getDataByInternalId(ep_id);
        dist_t dist = index.fstdistfunc_(data_point, ep_data, index.dist_func_param_);
        lowerBound = dist;
        only_top_candidates.emplace(dist, ep_id);
        top_candidates.emplace(dist, ep_id);
        if (!bare_bone_search && stop_condition) {
            stop_condition->add_point_to_result(index.getExternalLabel(ep_id), ep_data, dist);
        }
        candidate_set.emplace(-dist, ep_id, 0);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(-lowerBound, ep_id, 0);
    }

    visited_array[ep_id] = visited_array_tag;

    dist_t prev_top_bound = 0;
    int top_unchanged_iter = 0;
    bool overlap_prefetched = false;
    bool early_stop_triggered = false;
    int pre_visit_node_hop = -1;
    int change_hop_count = 0;
    int iter_count = 0;

    const int max_iter_count = static_cast<int>(ef);
    stable_hops = std::max(0, stable_hops);
    stableiter = std::max(1, stableiter);
    least_else_size = std::max(1, least_else_size);

    auto maybe_record_label = [&](unsigned long long label) {
        if (collect_issued_labels) {
            issued_label_set.insert(label);
        }
    };

    auto issue_current_top = [&]() {
        auto copy_top = only_top_candidates;
        std::vector<unsigned long long> labels;
        labels.reserve(copy_top.size());
        while (!copy_top.empty()) {
            const auto label = static_cast<unsigned long long>(
                index.getExternalLabel(copy_top.top().second));
            labels.push_back(label);
            maybe_record_label(label);
            copy_top.pop();
        }
        issue_prefetch(labels);
    };

    while (!candidate_set.empty()) {
        auto current_node_tuple = candidate_set.top();
        dist_t candidate_dist = -std::get<0>(current_node_tuple);
        hnswlib::tableint current_node_id = std::get<1>(current_node_tuple);
        int cur_visit_node_hop = std::get<2>(current_node_tuple);

        bool flag_stop_search;
        if (bare_bone_search) {
            flag_stop_search = candidate_dist > lowerBound;
        } else if (stop_condition) {
            flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
        } else {
            flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
        }
        if (flag_stop_search) {
            break;
        }

        candidate_set.pop();
        iter_count++;

        if (pre_visit_node_hop != -1) {
            if (cur_visit_node_hop - pre_visit_node_hop == 1) {
                change_hop_count = 0;
            } else {
                change_hop_count++;
            }
            if (top_candidates.size() >= k &&
                iter_count < max_iter_count &&
                change_hop_count >= stable_hops) {
                early_stop_triggered = true;
                if (signals) {
                    signals->early_stop_triggered = true;
                }
                break;
            }
        }
        pre_visit_node_hop = cur_visit_node_hop;

        int* data = reinterpret_cast<int*>(index.get_linklist0(current_node_id));
        const std::size_t size = index.getListCount(
            reinterpret_cast<hnswlib::linklistsizeint*>(data));
        if (collect_metrics) {
            index.metric_hops++;
            index.metric_distance_computations += size;
        }

        for (std::size_t j = 1; j <= size; j++) {
            const int candidate_raw = *(data + j);
            if (candidate_raw < 0 ||
                static_cast<std::size_t>(candidate_raw) >= index.max_elements_) {
                index.visited_list_pool_->releaseVisitedList(vl);
                throw std::runtime_error("candidate id out of range");
            }
            const auto candidate_id = static_cast<hnswlib::tableint>(candidate_raw);
            if (visited_array[candidate_id] == visited_array_tag) {
                continue;
            }
            visited_array[candidate_id] = visited_array_tag;

            char* candidate_data = index.getDataByInternalId(candidate_id);
            dist_t dist = index.fstdistfunc_(data_point, candidate_data, index.dist_func_param_);

            bool flag_consider_candidate;
            if (!bare_bone_search && stop_condition) {
                flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
            } else {
                flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
            }
            if (!flag_consider_candidate) {
                continue;
            }

            candidate_set.emplace(-dist, candidate_id, cur_visit_node_hop + 1);

            if (bare_bone_search ||
                (!index.isMarkedDeleted(candidate_id) &&
                 ((!isIdAllowed) || (*isIdAllowed)(index.getExternalLabel(candidate_id))))) {
                top_candidates.emplace(dist, candidate_id);
                if (only_top_candidates.size() < k) {
                    only_top_candidates.emplace(dist, candidate_id);
                } else if (dist < only_top_candidates.top().first) {
                    if (overlap_prefetched) {
                        else_only_top_candidates.emplace(dist, candidate_id);
                    }
                    only_top_candidates.emplace(dist, candidate_id);
                    only_top_candidates.pop();
                }
                if (!bare_bone_search && stop_condition) {
                    stop_condition->add_point_to_result(
                        index.getExternalLabel(candidate_id), candidate_data, dist);
                }
            }

            bool flag_remove_extra;
            if (!bare_bone_search && stop_condition) {
                flag_remove_extra = stop_condition->should_remove_extra();
            } else {
                flag_remove_extra = top_candidates.size() > ef;
            }
            while (flag_remove_extra) {
                hnswlib::tableint id = top_candidates.top().second;
                top_candidates.pop();
                if (!bare_bone_search && stop_condition) {
                    stop_condition->remove_point_from_result(
                        index.getExternalLabel(id),
                        index.getDataByInternalId(id),
                        dist);
                    flag_remove_extra = stop_condition->should_remove_extra();
                } else {
                    flag_remove_extra = top_candidates.size() > ef;
                }
            }
            if (!top_candidates.empty()) {
                lowerBound = top_candidates.top().first;
            }
        }

        if (!overlap_prefetched && only_top_candidates.size() >= k) {
            const dist_t current_top_bound = only_top_candidates.top().first;
            if (prev_top_bound == 0) {
                prev_top_bound = current_top_bound;
            } else if (prev_top_bound == current_top_bound) {
                top_unchanged_iter++;
                if (top_unchanged_iter >= stableiter) {
                    overlap_prefetched = true;
                    if (signals) {
                        signals->overlap_prefetch_triggered = true;
                    }
                    issue_current_top();
                }
            } else {
                top_unchanged_iter = 0;
                prev_top_bound = current_top_bound;
            }
        } else if (overlap_prefetched &&
                   else_only_top_candidates.size() >= static_cast<std::size_t>(least_else_size)) {
            std::vector<unsigned long long> labels;
            labels.reserve(else_only_top_candidates.size());
            while (!else_only_top_candidates.empty()) {
                const auto label = static_cast<unsigned long long>(
                    index.getExternalLabel(else_only_top_candidates.top().second));
                labels.push_back(label);
                maybe_record_label(label);
                else_only_top_candidates.pop();
            }
            issue_prefetch(labels);
        }
    }

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }

    if (!overlap_prefetched) {
        issue_current_top();
    } else if (!else_only_top_candidates.empty()) {
        std::vector<unsigned long long> labels;
        labels.reserve(else_only_top_candidates.size());
        while (!else_only_top_candidates.empty()) {
            const auto label = static_cast<unsigned long long>(
                index.getExternalLabel(else_only_top_candidates.top().second));
            labels.push_back(label);
            maybe_record_label(label);
            else_only_top_candidates.pop();
        }
        issue_prefetch(labels);
    }

    if (signals) {
        signals->pending_prefetch_ios = pending_prefetch;
        signals->issued_prefetch_labels = issued_prefetch_labels;
        signals->topk_labels = k;
        signals->issued_label_set = std::move(issued_label_set);
    }
    (void)mapped_missing;
    index.visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}

template <typename dist_t>
std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
SearchKnnEarlyStopOverlapMapped(
    const hnswlib::HierarchicalNSW<dist_t>& index,
    const void* query_data,
    std::size_t k,
    int stable_hops,
    int stableiter,
    IOuringManager* ioer,
    int least_else_size,
    int query_id,
    EarlyStopOverlapSignals* signals,
    bool collect_issued_labels = false,
    hnswlib::BaseFilterFunctor* isIdAllowed = nullptr)
{
    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result;
    if (signals) {
        signals->topk_labels = k;
    }
    if (index.cur_element_count == 0 || k == 0) {
        return result;
    }

    hnswlib::tableint currObj = index.enterpoint_node_;
    dist_t curdist = index.fstdistfunc_(
        query_data,
        index.getDataByInternalId(index.enterpoint_node_),
        index.dist_func_param_);

    for (int level = index.maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            auto* data = reinterpret_cast<unsigned int*>(index.get_linklist(currObj, level));
            const int size = index.getListCount(data);
            index.metric_hops++;
            index.metric_distance_computations += static_cast<std::size_t>(std::max(0, size));

            auto* datal = reinterpret_cast<hnswlib::tableint*>(data + 1);
            for (int i = 0; i < size; i++) {
                hnswlib::tableint cand = datal[i];
                if (cand < 0 ||
                    static_cast<std::size_t>(cand) >= index.max_elements_) {
                    throw std::runtime_error("candidate id out of range");
                }
                dist_t d = index.fstdistfunc_(
                    query_data,
                    index.getDataByInternalId(cand),
                    index.dist_func_param_);
                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    const std::size_t ef = std::max(index.ef_, k);
    HnswInternalQueue<dist_t> queue;
    const bool bare_bone_search = !index.num_deleted_ && !isIdAllowed;
    if (bare_bone_search) {
        queue = SearchBaseLayerSTEarlyStopOverlapMapped<dist_t, true>(
            index,
            currObj,
            query_data,
            ef,
            k,
            stable_hops,
            stableiter,
            ioer,
            least_else_size,
            query_id,
            signals,
            collect_issued_labels,
            isIdAllowed);
    } else {
        queue = SearchBaseLayerSTEarlyStopOverlapMapped<dist_t, false>(
            index,
            currObj,
            query_data,
            ef,
            k,
            stable_hops,
            stableiter,
            ioer,
            least_else_size,
            query_id,
            signals,
            collect_issued_labels,
            isIdAllowed);
    }

    while (!queue.empty()) {
        result.emplace(queue.top().first, index.getExternalLabel(queue.top().second));
        queue.pop();
    }
    return result;
}

} // namespace annfs::hnsw
