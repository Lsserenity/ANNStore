// ANNFS-specific mapped-overlap support for IO prefetch during search.
// This header is included inside the class definition in hnswalg.h.

#include <unordered_set>
#include <cstdlib>
#include <iostream>

#ifndef ANNFS_HNSW_MEMBER_CONTEXT
#include <hnswlib/hnswalg.h>
#else

    using CandidateQueue = std::priority_queue<std::pair<dist_t, tableint>,
                                               std::vector<std::pair<dist_t, tableint>>,
                                               CompareByFirst>;

    inline void issue_prefetch_mapped_by_worker(
        IOuringManager *io_worker,
        const std::vector<unsigned long long>& labels,
        int query_id,
        int& pending_prefetch,
        std::size_t& issued_label_total,
        std::size_t& mapped_missing_total) const
    {
        if (!io_worker || labels.empty()) {
            return;
        }
        std::size_t missing = 0;
        std::size_t issued = 0;
        const auto& pc = ParamConfig::instance();
        constexpr bool kWaitPerQuery = false;
        const int pending = io_worker->mapped_prefetch(
            labels,
            pc,
            query_id,
            kWaitPerQuery,
            &missing,
            &issued);
        pending_prefetch += pending;
        issued_label_total += labels.size();
        mapped_missing_total += missing;
    }

    std::unordered_set<unsigned long long>
    searchKnnPrefetchMappedBaseline(
        const void *query_data,
        size_t k,
        int stableiter,
        IOuringManager *io_worker,
        int least_else_size,
        BaseFilterFunctor *isIdAllowed = nullptr,
        int *pending_prefetch_ios_out = nullptr,
        int query_id = -1,
        std::size_t *issued_prefetch_labels_out = nullptr,
        std::size_t *mapped_missing_total_out = nullptr,
        bool *prefetch_triggered_out = nullptr,
        std::size_t *topk_labels_out = nullptr,
        bool collect_issued_labels = false) const
    {
        if (cur_element_count == 0)
            return {};

        int pending_prefetch_count = 0;
        std::size_t issued_prefetch_labels = 0;
        std::size_t mapped_missing_total = 0;
        std::size_t topk_labels = k;
        std::unordered_set<unsigned long long> sent_prefetch_labels;
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                unsigned int *data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++)
                {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist)
                    {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        bool bare_bone_search = !num_deleted_;
        if (bare_bone_search)
        {
            sent_prefetch_labels = searchBaseLayerSTPrefetchMappedBaseline<true>(
                currObj,
                query_data,
                std::max(ef_, k),
                k,
                stableiter,
                io_worker,
                least_else_size,
                nullptr,
                nullptr,
                &pending_prefetch_count,
                query_id,
                &issued_prefetch_labels,
                &mapped_missing_total,
                prefetch_triggered_out,
                &topk_labels,
                collect_issued_labels);
        }
        else
        {
            sent_prefetch_labels = searchBaseLayerSTPrefetchMappedBaseline<false>(
                currObj,
                query_data,
                std::max(ef_, k),
                k,
                stableiter,
                io_worker,
                least_else_size,
                nullptr,
                nullptr,
                &pending_prefetch_count,
                query_id,
                &issued_prefetch_labels,
                &mapped_missing_total,
                prefetch_triggered_out,
                &topk_labels,
                collect_issued_labels);
        }
        if (pending_prefetch_ios_out) {
            *pending_prefetch_ios_out = pending_prefetch_count;
        }
        if (issued_prefetch_labels_out) {
            *issued_prefetch_labels_out = issued_prefetch_labels;
        }
        if (mapped_missing_total_out) {
            *mapped_missing_total_out = mapped_missing_total;
        }
        if (topk_labels_out) {
            *topk_labels_out = topk_labels;
        }
        if (collect_issued_labels && sent_prefetch_labels.size() < k) {
            std::cerr << "[ERROR] mapped overlap issued_labels("
                      << sent_prefetch_labels.size() << ") < topk(" << k << ")\n";
            std::exit(1);
        }
        return sent_prefetch_labels;
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::unordered_set<unsigned long long> searchBaseLayerSTPrefetchMappedBaseline(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        size_t k,
        int stableiter,
        IOuringManager *io_worker,
        int least_else_size,
        BaseFilterFunctor *isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t> *stop_condition = nullptr,
        int *pending_prefetch_ios_out = nullptr,
        int query_id = -1,
        std::size_t *issued_prefetch_labels_out = nullptr,
        std::size_t *mapped_missing_total_out = nullptr,
        bool *prefetch_triggered_out = nullptr,
        std::size_t *topk_labels_out = nullptr,
        bool collect_issued_labels = false) const
    {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        dist_t lK = 0;
        dist_t prelK = 0;
        CandidateQueue only_top_candidates;
        CandidateQueue else_only_top_candidates;
        CandidateQueue top_candidates;
        CandidateQueue candidate_set;
        std::unordered_set<unsigned long long> send_prefetch_labels;
        int topunchangediter = 0;
        bool prefetch = false;
        int pending_prefetch = 0;
        std::size_t issued_prefetch_labels = 0;
        std::size_t mapped_missing_total = 0;
        std::size_t topk_labels = k;
        auto issue_prefetch = [&](const std::vector<unsigned long long>& labels) {
            const std::size_t before = issued_prefetch_labels;
            issue_prefetch_mapped_by_worker(
                io_worker, labels, query_id,
                pending_prefetch, issued_prefetch_labels, mapped_missing_total);
            return issued_prefetch_labels - before;
        };
        auto maybe_record_label = [&](unsigned long long label) {
            if (collect_issued_labels) {
                send_prefetch_labels.insert(label);
            }
        };

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))))
        {
            char *ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            only_top_candidates.emplace(dist, ep_id);
            lK = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition)
            {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        }
        else
        {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;
        while (!candidate_set.empty())
        {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search)
            {
                flag_stop_search = candidate_dist > lowerBound;
            }
            else
            {
                if (stop_condition)
                {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                }
                else
                {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search)
            {
                break;
            }
            candidate_set.pop();
            tableint current_node_id = current_node_pair.second;
            int *data = (int *)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *)data);
            if (collect_metrics)
            {
                metric_hops++;
                metric_distance_computations += size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif
            for (size_t j = 1; j <= size; j++)
            {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);
#endif
                if (!(visited_array[candidate_id] == visited_array_tag))
                {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition)
                    {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    }
                    else
                    {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate)
                    {
                        candidate_set.emplace(-dist, candidate_id);

#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                            offsetLevel0_,
                                        _MM_HINT_T0);
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id)))))
                        {
                            top_candidates.emplace(dist, candidate_id);
                            if (only_top_candidates.size() < k)
                            {
                                only_top_candidates.emplace(dist, candidate_id);
                            }
                            else if (dist < lK)
                            {
                                if (prefetch)
                                {
                                    else_only_top_candidates.emplace(dist, candidate_id);
                                }
                                only_top_candidates.emplace(dist, candidate_id);
                                only_top_candidates.pop();
                            }
                            lK = only_top_candidates.top().first;
                            if (!bare_bone_search && stop_condition)
                            {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition)
                        {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        }
                        else
                        {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra)
                        {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition)
                            {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            }
                            else
                            {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                        {
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
            if (!prefetch && only_top_candidates.size() >= k)
            {
                if (prelK == 0)
                {
                    prelK = lK;
                }
                else
                {
                    if (prelK == lK)
                    {
                        topunchangediter++;
                        if (topunchangediter == stableiter)
                        {
                            prefetch = true;
                            auto only_top_candidates_copy = only_top_candidates;
                            std::vector<unsigned long long> cur_labels;
                            while (!only_top_candidates_copy.empty())
                            {
                                auto temp_label = static_cast<unsigned long long>(getExternalLabel(only_top_candidates_copy.top().second));
                                cur_labels.emplace_back(temp_label);
                                maybe_record_label(temp_label);
                                only_top_candidates_copy.pop();
                            }
                            issue_prefetch(cur_labels);
                        }
                    }
                    else
                    {
                        topunchangediter = 0;
                        prelK = lK;
                    }
                }
            }
            else if (prefetch)
            {
                if (else_only_top_candidates.size() >= static_cast<size_t>(least_else_size))
                {
                    std::vector<unsigned long long> elseprefetchindexs;
                    while (!else_only_top_candidates.empty())
                    {
                        auto temp_label = static_cast<unsigned long long>(getExternalLabel(else_only_top_candidates.top().second));
                        elseprefetchindexs.emplace_back(temp_label);
                        maybe_record_label(temp_label);
                        else_only_top_candidates.pop();
                    }
                    issue_prefetch(elseprefetchindexs);
                }
            }
        }
        if (!prefetch)
        {
            std::vector<unsigned long long> cur_labels;
            auto only_top_candidates_copy = only_top_candidates;
            cur_labels.reserve(only_top_candidates_copy.size());
            while (!only_top_candidates_copy.empty()) {
                auto temp_label = static_cast<unsigned long long>(getExternalLabel(only_top_candidates_copy.top().second));
                cur_labels.emplace_back(temp_label);
                maybe_record_label(temp_label);
                only_top_candidates_copy.pop();
            }
            issue_prefetch(cur_labels);
        }
        else if (!else_only_top_candidates.empty())
        {
            std::vector<unsigned long long> elseprefetchindexs;
            elseprefetchindexs.reserve(else_only_top_candidates.size());
            while (!else_only_top_candidates.empty())
            {
                auto temp_label = static_cast<unsigned long long>(getExternalLabel(else_only_top_candidates.top().second));
                elseprefetchindexs.emplace_back(temp_label);
                maybe_record_label(temp_label);
                else_only_top_candidates.pop();
            }
            issue_prefetch(elseprefetchindexs);
        }
        visited_list_pool_->releaseVisitedList(vl);
        if (pending_prefetch_ios_out) {
            *pending_prefetch_ios_out = pending_prefetch;
        }
        if (issued_prefetch_labels_out) {
            *issued_prefetch_labels_out = issued_prefetch_labels;
        }
        if (mapped_missing_total_out) {
            *mapped_missing_total_out = mapped_missing_total;
        }
        if (topk_labels_out) {
            *topk_labels_out = topk_labels;
        }
        if (prefetch_triggered_out) {
            *prefetch_triggered_out = prefetch;
        }
        return send_prefetch_labels;
    }

#endif  // ANNFS_HNSW_MEMBER_CONTEXT
