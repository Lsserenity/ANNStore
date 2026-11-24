#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../config/global_config.hpp"
#include "../config/param_config.hpp"
#include "mapped_label_offset_resolver.hpp"

namespace annfs::io {

// 统一请求规划结果。
// by_blocks: key=连续块数，value=该块数对应的起始 block offset 列表（已去重）。
// missing: map 解码未命中数（shift_offbits 模式恒为 0）。
// total_reqs: 去重后请求总数（所有 bucket 的 value.size() 之和）。
struct BlockReqPlan {
    std::unordered_map<std::uint32_t, std::vector<BlockReq>> by_blocks;
    std::size_t missing = 0;
    std::size_t total_reqs = 0;
    std::size_t input_labels = 0;
    std::size_t raw_reqs = 0;
    std::size_t normalized_reqs = 0;
    std::size_t dropped_unsupported = 0;
    std::uint32_t max_blocks_raw = 0;
    std::uint32_t max_blocks_normalized = 0;
    bool used_label_map = false;
    bool label_map_fallback = false;
};

namespace detail {

inline std::vector<BlockReq> BuildNoMapReqs(
    const std::vector<unsigned long long>& labels,
    bool split,
    std::uint64_t block_size,
    GlobalConfig::LABEL_DECODE_MODE decode_mode)
{
    std::vector<BlockReq> reqs;
    if (labels.empty()) {
        return reqs;
    }

    if (decode_mode == GlobalConfig::LABEL_DECODE_MODE::SHIFT_OFF_BITS) {
        reqs.reserve(labels.size());
        for (auto id : labels) {
            reqs.push_back(BlockReq{
                static_cast<off_t>(id << GlobalConfig::instance().OFF_BITS_LEN),
                1u,
                {BlockReq::ParseHint{0u, 0u, static_cast<std::uint64_t>(id)}}});
        }
        return reqs;
    }

    if (block_size == 0) {
        return reqs;
    }

    // no-map + packed_map: label 直接承载 packed64
    // packed64 = [3 bits crossed_blocks][61 bits start_byte]
    reqs.reserve(labels.size());
    std::unordered_map<std::uint64_t, std::size_t> idx_block1;
    std::unordered_map<std::uint64_t, std::size_t> idx_span;

    for (auto raw : labels) {
        const std::uint64_t packed64 = static_cast<std::uint64_t>(raw);
        const auto dec = DecodePacked64(
            packed64, static_cast<std::uint32_t>(block_size));
        const std::uint64_t block_base = dec.block_base;
        const std::uint32_t blocks = dec.blocks_to_read;
        const std::uint32_t in_block_offset =
            static_cast<std::uint32_t>(dec.in_block_offset);

        if (split) {
            for (std::uint32_t i = 0; i < blocks; ++i) {
                const std::uint64_t off =
                    block_base + static_cast<std::uint64_t>(i) * block_size;
                auto it = idx_block1.find(off);
                if (it == idx_block1.end()) {
                    BlockReq req;
                    req.base = static_cast<off_t>(off);
                    req.blocks = 1u;
                    idx_block1.emplace(off, reqs.size());
                    reqs.push_back(std::move(req));
                    it = idx_block1.find(off);
                }
                if (i == 0) {
                    reqs[it->second].parse_hints.push_back(BlockReq::ParseHint{
                        in_block_offset,
                        in_block_offset,
                        packed64});
                }
            }
            continue;
        }

        if (blocks <= 1u) {
            auto it = idx_block1.find(block_base);
            if (it == idx_block1.end()) {
                BlockReq req;
                req.base = static_cast<off_t>(block_base);
                req.blocks = 1u;
                idx_block1.emplace(block_base, reqs.size());
                reqs.push_back(std::move(req));
                it = idx_block1.find(block_base);
            }
            reqs[it->second].parse_hints.push_back(BlockReq::ParseHint{
                in_block_offset,
                in_block_offset,
                packed64});
            continue;
        }

        const std::uint64_t span_key = (block_base << 16) ^ static_cast<std::uint64_t>(blocks & 0xFFFFu);
        auto sit = idx_span.find(span_key);
        if (sit == idx_span.end()) {
            BlockReq req;
            req.base = static_cast<off_t>(block_base);
            req.blocks = blocks;
            idx_span.emplace(span_key, reqs.size());
            reqs.push_back(std::move(req));
            sit = idx_span.find(span_key);
        }
        reqs[sit->second].parse_hints.push_back(BlockReq::ParseHint{
            in_block_offset,
            in_block_offset,
            packed64});
    }
    return reqs;
}

inline std::vector<BlockReq> MergeContiguousReqs(
    const std::vector<BlockReq>& reqs,
    std::uint64_t block_size)
{
    if (reqs.empty() || block_size == 0) {
        return {};
    }
    std::vector<BlockReq> sorted;
    sorted.reserve(reqs.size());
    for (const auto& r : reqs) {
        sorted.push_back(r);
        sorted.back().blocks = std::max<std::uint32_t>(1u, sorted.back().blocks);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const BlockReq& a, const BlockReq& b) {
                  return static_cast<std::uint64_t>(a.base) <
                         static_cast<std::uint64_t>(b.base);
              });

    std::vector<BlockReq> out;
    out.reserve(sorted.size());
    BlockReq cur = sorted[0];
    cur.base = static_cast<off_t>(static_cast<std::uint64_t>(cur.base));
    std::uint64_t cur_base = static_cast<std::uint64_t>(cur.base);
    std::uint64_t cur_end = cur_base +
                            static_cast<std::uint64_t>(cur.blocks) * block_size;

    for (std::size_t i = 1; i < sorted.size(); ++i) {
        const auto b = static_cast<std::uint64_t>(sorted[i].base);
        const auto e = b + static_cast<std::uint64_t>(sorted[i].blocks) * block_size;
        if (b <= cur_end) {
            cur_end = std::max(cur_end, e);
            // 合并时把后续 req 的解析 hint 重映射到合并后 req 的相对偏移。
            for (const auto& h : sorted[i].parse_hints) {
                const std::uint64_t abs_start =
                    b + static_cast<std::uint64_t>(h.start_in_req);
                if (abs_start < cur_base || abs_start >= cur_end) {
                    continue;
                }
                BlockReq::ParseHint nh = h;
                nh.start_in_req = static_cast<std::uint32_t>(abs_start - cur_base);
                cur.parse_hints.push_back(nh);
            }
            continue;
        }
        cur.base = static_cast<off_t>(cur_base);
        cur.blocks = static_cast<std::uint32_t>((cur_end - cur_base) / block_size);
        out.push_back(std::move(cur));
        cur = sorted[i];
        cur_base = b;
        cur_end = e;
    }
    cur.base = static_cast<off_t>(cur_base);
    cur.blocks = static_cast<std::uint32_t>((cur_end - cur_base) / block_size);
    out.push_back(std::move(cur));
    return out;
}

inline void AppendReqChunkWithHints(std::vector<BlockReq>& out,
                                    const BlockReq& src,
                                    std::uint64_t src_base,
                                    std::uint64_t block_size,
                                    std::uint64_t chunk_base,
                                    std::uint32_t chunk_blocks)
{
    BlockReq part = src;
    part.base = static_cast<off_t>(chunk_base);
    part.blocks = chunk_blocks;
    part.parse_hints.clear();
    const std::uint64_t chunk_end =
        chunk_base + static_cast<std::uint64_t>(chunk_blocks) * block_size;
    for (const auto& h : src.parse_hints) {
        const std::uint64_t abs_start =
            src_base + static_cast<std::uint64_t>(h.start_in_req);
        if (abs_start < chunk_base || abs_start >= chunk_end) {
            continue;
        }
        auto nh = h;
        nh.start_in_req = static_cast<std::uint32_t>(abs_start - chunk_base);
        part.parse_hints.push_back(nh);
    }
    out.push_back(std::move(part));
}

inline std::vector<BlockReq> MergeContiguousReqsWithMaxBlocks(
    const std::vector<BlockReq>& reqs,
    std::uint64_t block_size,
    std::uint32_t max_blocks)
{
    if (reqs.empty() || block_size == 0) {
        return {};
    }
    if (max_blocks == 0) {
        max_blocks = std::numeric_limits<std::uint32_t>::max();
    }

    std::vector<BlockReq> merged = MergeContiguousReqs(reqs, block_size);
    std::vector<BlockReq> out;
    out.reserve(merged.size());
    for (const auto& r : merged) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, r.blocks);
        if (blocks <= max_blocks) {
            out.push_back(r);
            continue;
        }
        std::uint32_t remain = blocks;
        std::uint64_t chunk_base = static_cast<std::uint64_t>(r.base);
        const std::uint64_t req_base = static_cast<std::uint64_t>(r.base);
        while (remain > 0) {
            const std::uint32_t chunk = std::min(max_blocks, remain);
            AppendReqChunkWithHints(
                out, r, req_base, block_size, chunk_base, chunk);
            chunk_base += static_cast<std::uint64_t>(chunk) * block_size;
            remain -= chunk;
        }
    }
    return out;
}

inline std::vector<BlockReq> SplitReqsByMaxBlocks(
    const std::vector<BlockReq>& reqs,
    std::uint64_t block_size,
    std::uint32_t max_blocks)
{
    if (reqs.empty() || block_size == 0 || max_blocks == 0) {
        return reqs;
    }
    std::vector<BlockReq> out;
    out.reserve(reqs.size());
    for (const auto& r : reqs) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, r.blocks);
        if (blocks <= max_blocks) {
            out.push_back(r);
            continue;
        }
        std::uint32_t remain = blocks;
        std::uint64_t cur_base = static_cast<std::uint64_t>(r.base);
        const std::uint64_t req_base = static_cast<std::uint64_t>(r.base);
        while (remain > 0) {
            const std::uint32_t chunk = std::min(max_blocks, remain);
            BlockReq part = r;
            part.base = static_cast<off_t>(cur_base);
            part.blocks = chunk;
            part.parse_hints.clear();
            const std::uint64_t chunk_end =
                cur_base + static_cast<std::uint64_t>(chunk) * block_size;
            for (const auto& h : r.parse_hints) {
                const std::uint64_t abs_start =
                    req_base + static_cast<std::uint64_t>(h.start_in_req);
                if (abs_start < cur_base || abs_start >= chunk_end) {
                    continue;
                }
                auto nh = h;
                nh.start_in_req =
                    static_cast<std::uint32_t>(abs_start - cur_base);
                part.parse_hints.push_back(nh);
            }
            out.push_back(std::move(part));
            cur_base += static_cast<std::uint64_t>(chunk) * block_size;
            remain -= chunk;
        }
    }
    return out;
}

inline std::vector<std::uint32_t> BuildAllowedBucketBlocks(
    const std::map<int, int>& buckets)
{
    std::vector<std::uint32_t> allowed;
    allowed.reserve(buckets.size());
    for (const auto& kv : buckets) {
        if (kv.first > 0 && kv.second > 0) {
            allowed.push_back(static_cast<std::uint32_t>(kv.first));
        }
    }
    std::sort(allowed.begin(), allowed.end());
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return allowed;
}

inline std::uint32_t MaxReqBlocks(const std::vector<BlockReq>& reqs)
{
    std::uint32_t max_blocks = 0;
    for (const auto& r : reqs) {
        max_blocks = std::max(max_blocks, std::max<std::uint32_t>(1u, r.blocks));
    }
    return max_blocks;
}

inline std::vector<BlockReq> NormalizeReqsByBucketBlocks(
    const std::vector<BlockReq>& reqs,
    std::uint64_t block_size,
    const std::vector<std::uint32_t>& allowed_bucket_blocks)
{
    if (reqs.empty() || block_size == 0 || allowed_bucket_blocks.empty()) {
        return reqs;
    }

    std::unordered_set<std::uint32_t> allowed_set;
    allowed_set.reserve(allowed_bucket_blocks.size() * 2 + 1);
    for (auto b : allowed_bucket_blocks) {
        allowed_set.insert(std::max<std::uint32_t>(1u, b));
    }

    std::vector<std::uint32_t> allowed_asc = allowed_bucket_blocks;
    std::sort(allowed_asc.begin(), allowed_asc.end());
    const std::uint32_t max_bucket = allowed_asc.empty() ? 0u : allowed_asc.back();

    std::vector<BlockReq> out;
    out.reserve(reqs.size());

    for (const auto& r : reqs) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, r.blocks);
        if (allowed_set.find(blocks) != allowed_set.end()) {
            out.push_back(r);
            continue;
        }

        std::uint32_t remain = blocks;
        std::uint64_t chunk_base = static_cast<std::uint64_t>(r.base);
        const std::uint64_t req_base = static_cast<std::uint64_t>(r.base);
        bool split_ok = true;

        while (remain > 0) {
            std::uint32_t pick = 0;
            if (max_bucket > 0 && remain > max_bucket) {
                pick = max_bucket;
            } else {
                for (auto b : allowed_asc) {
                    if (b >= remain) {
                        pick = b;
                        break;
                    }
                }
            }
            if (pick == 0) {
                split_ok = false;
                break;
            }
            AppendReqChunkWithHints(
                out, r, req_base, block_size, chunk_base, pick);
            chunk_base += static_cast<std::uint64_t>(pick) * block_size;
            remain -= std::min(remain, pick);
        }

        if (!split_ok) {
            out.push_back(r);
        }
    }
    return out;
}

inline std::vector<BlockReq> KeepOnlyAllowedBucketReqs(
    const std::vector<BlockReq>& reqs,
    const std::vector<std::uint32_t>& allowed_bucket_blocks,
    std::size_t* dropped)
{
    if (dropped) {
        *dropped = 0;
    }
    if (reqs.empty() || allowed_bucket_blocks.empty()) {
        return reqs;
    }

    std::unordered_set<std::uint32_t> allowed_set;
    allowed_set.reserve(allowed_bucket_blocks.size() * 2 + 1);
    for (auto b : allowed_bucket_blocks) {
        allowed_set.insert(std::max<std::uint32_t>(1u, b));
    }

    std::vector<BlockReq> out;
    out.reserve(reqs.size());
    for (const auto& r : reqs) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, r.blocks);
        if (allowed_set.find(blocks) != allowed_set.end()) {
            out.push_back(r);
        } else if (dropped) {
            (*dropped)++;
        }
    }
    return out;
}

inline std::vector<BlockReq> SplitReqsToBlock1(
    const std::vector<BlockReq>& reqs,
    std::uint64_t block_size)
{
    std::vector<BlockReq> out;
    if (reqs.empty() || block_size == 0) {
        return out;
    }
    out.reserve(reqs.size());
    for (const auto& r : reqs) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, r.blocks);
        for (std::uint32_t i = 0; i < blocks; ++i) {
            out.push_back(BlockReq{
                static_cast<off_t>(
                    static_cast<std::uint64_t>(r.base) +
                    static_cast<std::uint64_t>(i) * block_size),
                1u,
                {}});
            // 只有命中该 block 的 hint 才复制到这个 split 后的 req。
            const std::uint64_t block_base =
                static_cast<std::uint64_t>(r.base) +
                static_cast<std::uint64_t>(i) * block_size;
            const std::uint64_t block_end = block_base + block_size;
            for (const auto& h : r.parse_hints) {
                const std::uint64_t abs_start =
                    static_cast<std::uint64_t>(r.base) +
                    static_cast<std::uint64_t>(h.start_in_req);
                if (abs_start < block_base || abs_start >= block_end) {
                    continue;
                }
                auto nh = h;
                nh.start_in_req = static_cast<std::uint32_t>(abs_start - block_base);
                out.back().parse_hints.push_back(nh);
            }
        }
    }
    return out;
}

inline BlockReqPlan BucketizeReqs(const std::vector<BlockReq>& reqs,
                                  std::size_t missing)
{
    BlockReqPlan plan;
    plan.missing = missing;
    std::unordered_map<std::uint32_t, std::unordered_map<off_t, std::size_t>> idx;
    idx.reserve(8);

    for (const auto& r : reqs) {
        const std::uint32_t blocks = std::max<std::uint32_t>(1u, r.blocks);
        auto& bm = idx[blocks];
        auto it = bm.find(r.base);
        if (it == bm.end()) {
            bm.emplace(r.base, plan.by_blocks[blocks].size());
            plan.by_blocks[blocks].push_back(r);
            ++plan.total_reqs;
        } else {
            auto& dst = plan.by_blocks[blocks][it->second];
            dst.parse_hints.insert(
                dst.parse_hints.end(), r.parse_hints.begin(), r.parse_hints.end());
        }
    }
    return plan;
}

} // namespace detail

// 统一唯一规划入口：
// 1) use_label_map + label_decode_mode 决定 map/no-map 与解码方式。
// 2) mapped_span_mode 决定 split / bucket_pool。
// 3) 返回按 blocks 分桶后的请求 map（key=blocks，value=base offsets）。
inline BlockReqPlan BuildReqPlanU64(
    const std::vector<unsigned long long>& labels,
    const ParamConfig& pc)
{
    const auto& gc = GlobalConfig::instance();
    const bool split = (gc.MAPPED_SPAN_MODE == "split");
    const auto block_size =
        static_cast<std::uint64_t>(GlobalConfig::instance().LOGIC_BLOCK_SIZE);

    BlockReqPlan plan;
    plan.input_labels = labels.size();
    std::vector<BlockReq> reqs;
    std::size_t missing = 0;

    if (pc.UseLabelMap()) {
        plan.used_label_map = true;
        try {
            reqs = MappedLabelOffsetResolver::Instance().MapLabelsToBlockReqsLiteU64(
                labels,
                false,
                &missing);
        } catch (...) {
            plan.label_map_fallback = true;
            missing = 0;
            reqs = detail::BuildNoMapReqs(labels, split, block_size, gc.LABEL_DECODE_MODE_CFG);
        }
    } else {
        reqs = detail::BuildNoMapReqs(labels, split, block_size, gc.LABEL_DECODE_MODE_CFG);
    }
    plan.raw_reqs = reqs.size();
    plan.max_blocks_raw = detail::MaxReqBlocks(reqs);

    const auto allowed_bucket_blocks =
        detail::BuildAllowedBucketBlocks(gc.MAPPED_SPANPOOL_BUCKETS);

    if (split) {
        reqs = detail::SplitReqsToBlock1(reqs, block_size);
    } else if (!allowed_bucket_blocks.empty()) {
        // bucket_pool/no-split: use bucket sizes as the only source of truth.
        reqs = detail::NormalizeReqsByBucketBlocks(
            reqs, block_size, allowed_bucket_blocks);
        std::size_t dropped_unsupported = 0;
        reqs = detail::KeepOnlyAllowedBucketReqs(
            reqs, allowed_bucket_blocks, &dropped_unsupported);
        plan.dropped_unsupported = dropped_unsupported;
        if (dropped_unsupported > 0) {
            std::cerr << "[WARNING] dropped unsupported bucket_pool reqs: "
                      << dropped_unsupported << std::endl;
        }
    }
    plan.normalized_reqs = reqs.size();
    plan.max_blocks_normalized = detail::MaxReqBlocks(reqs);

    auto bucketized = detail::BucketizeReqs(reqs, missing);
    bucketized.input_labels = plan.input_labels;
    bucketized.raw_reqs = plan.raw_reqs;
    bucketized.normalized_reqs = plan.normalized_reqs;
    bucketized.dropped_unsupported = plan.dropped_unsupported;
    bucketized.max_blocks_raw = plan.max_blocks_raw;
    bucketized.max_blocks_normalized = plan.max_blocks_normalized;
    bucketized.used_label_map = plan.used_label_map;
    bucketized.label_map_fallback = plan.label_map_fallback;
    return bucketized;
}

} // namespace annfs::io
