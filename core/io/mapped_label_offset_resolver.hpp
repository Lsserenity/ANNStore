#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "disk_info.hpp"

namespace annfs::io {

struct BlockReq {
    // base: 起始块偏移；blocks: 连续块数。
    off_t base = 0;
    std::uint32_t blocks = 1;
    struct ParseHint {
        // 相对 req.base 的起始字节偏移（用于后续解析字符串边界）。
        std::uint32_t start_in_req = 0;
        // 原始块内偏移（便于调试/回溯）。
        std::uint32_t in_block_offset = 0;
        // 对应 label。
        std::uint64_t label = 0;
    };
    // 同一 req 可对应多个 label 起始点。
    std::vector<ParseHint> parse_hints;
};

class MappedLabelOffsetResolver {
public:
    // 单例入口。
    // 返回值: 全局 resolver 实例。
    static MappedLabelOffsetResolver& Instance()
    {
        static MappedLabelOffsetResolver inst;
        return inst;
    }

    // 加载 label map 文件。
    // map_path: map 文件路径；block_size: 解码时使用的逻辑块大小。
    // 异常: block_size=0 或文件不可读时抛异常。
    void LoadFromFile(const std::string& map_path, std::uint32_t block_size)
    {
        if (block_size == 0) {
            throw std::runtime_error("MappedLabelOffsetResolver: block_size is zero");
        }
        auto snap = std::make_shared<Snapshot>();
        snap->label_map = LoadLabelMapFile(map_path);
        snap->map_path = map_path;
        snap->block_size = block_size;
        snap->block_align_mask = static_cast<std::uint64_t>(block_size) - 1ULL;
        std::atomic_store_explicit(&snapshot_, std::shared_ptr<const Snapshot>(snap), std::memory_order_release);
    }

    // label -> offsets（去重后）。
    // labels: 输入 label 列表；missing: 输出未命中的 label 数（可空）。
    // 返回值: 需要读取的 offset 列表（多块 label 会展开为多个块偏移）。
    std::vector<off_t> MapLabelsToOffsetsU64(
        const std::vector<unsigned long long>& labels,
        std::size_t* missing = nullptr) const
    {
        std::vector<off_t> offsets;
        if (missing) {
            *missing = 0;
        }
        if (labels.empty()) {
            return offsets;
        }

        auto snap = SnapshotOrThrow();
        offsets.reserve(labels.size());
        std::unordered_set<std::uint64_t> seen_offsets;

        for_each_decoded(labels, snap, missing, [&](std::uint64_t, const DecodedPacked64& d) {
            for (std::uint32_t i = 0; i < d.blocks_to_read; ++i) {
                const std::uint64_t off = d.block_base + static_cast<std::uint64_t>(i) * snap->block_size;
                if (seen_offsets.insert(off).second) {
                    offsets.push_back(static_cast<off_t>(off));
                }
            }
        });
        return offsets;
    }

    // label -> block request 列表（供 span-pool async 批量提交）。
    // force_split_to_block1=true: 输出全是 blocks=1 请求。
    // missing: 输出 map miss 数（可空）。
    std::vector<BlockReq> MapLabelsToBlockReqsLiteU64(
        const std::vector<unsigned long long>& labels,
        bool force_split_to_block1,
        std::size_t* missing = nullptr) const
    {
        std::vector<BlockReq> reqs;
        if (missing) {
            *missing = 0;
        }
        if (labels.empty()) {
            return reqs;
        }

        auto snap = SnapshotOrThrow();
        reqs.reserve(labels.size());
        std::unordered_map<std::uint64_t, std::size_t> idx_block1;
        std::unordered_map<std::uint64_t, std::size_t> idx_span;

        for_each_decoded(labels, snap, missing, [&](std::uint64_t lbl, const DecodedPacked64& d) {
            if (force_split_to_block1) {
                for (std::uint32_t i = 0; i < d.blocks_to_read; ++i) {
                    const std::uint64_t off = d.block_base + static_cast<std::uint64_t>(i) * snap->block_size;
                    auto it = idx_block1.find(off);
                    if (it == idx_block1.end()) {
                        BlockReq req;
                        req.base = static_cast<off_t>(off);
                        req.blocks = 1;
                        idx_block1.emplace(off, reqs.size());
                        reqs.push_back(std::move(req));
                        it = idx_block1.find(off);
                    }
                    if (i == 0) {
                        reqs[it->second].parse_hints.push_back(BlockReq::ParseHint{
                            static_cast<std::uint32_t>(d.in_block_offset),
                            static_cast<std::uint32_t>(d.in_block_offset),
                            lbl});
                    }
                }
                return;
            }

            if (d.blocks_to_read <= 1u) {
                auto it = idx_block1.find(d.block_base);
                if (it == idx_block1.end()) {
                    BlockReq req;
                    req.base = static_cast<off_t>(d.block_base);
                    req.blocks = 1;
                    idx_block1.emplace(d.block_base, reqs.size());
                    reqs.push_back(std::move(req));
                    it = idx_block1.find(d.block_base);
                }
                reqs[it->second].parse_hints.push_back(BlockReq::ParseHint{
                    static_cast<std::uint32_t>(d.in_block_offset),
                    static_cast<std::uint32_t>(d.in_block_offset),
                    lbl});
                return;
            }

            const std::uint64_t span_key = MakeSpanKey(d.block_base, d.blocks_to_read);
            auto sit = idx_span.find(span_key);
            if (sit == idx_span.end()) {
                BlockReq req;
                req.base = static_cast<off_t>(d.block_base);
                req.blocks = d.blocks_to_read;
                idx_span.emplace(span_key, reqs.size());
                reqs.push_back(std::move(req));
                sit = idx_span.find(span_key);
            }
            reqs[sit->second].parse_hints.push_back(BlockReq::ParseHint{
                static_cast<std::uint32_t>(d.in_block_offset),
                static_cast<std::uint32_t>(d.in_block_offset),
                lbl});
        });

        return reqs;
    }

    // 返回当前 map 中 label 总数。
    std::size_t Size() const
    {
        auto snap = SnapshotOrThrow();
        return snap->label_map.size();
    }

    // 返回当前 map 文件路径。
    std::string Path() const
    {
        auto snap = SnapshotOrThrow();
        return snap->map_path;
    }

    // 返回当前解码使用的 block_size。
    std::uint32_t BlockSize() const
    {
        auto snap = SnapshotOrThrow();
        return snap->block_size;
    }

private:
    struct Snapshot {
        std::unordered_map<std::uint64_t, std::uint64_t> label_map;
        std::string map_path;
        std::uint32_t block_size = 0;
        std::uint64_t block_align_mask = 0;
    };

    // 为 span 请求去重构造 key（同 base+blocks 视为同请求）。
    static std::uint64_t MakeSpanKey(std::uint64_t base, std::uint32_t blocks)
    {
        return (base << 16) ^ static_cast<std::uint64_t>(blocks & 0xFFFFu);
    }

    // 通用迭代器：完成 map 查找+packed 解码，再回调 fn。
    // missing: 未命中计数输出（可空）。
    template <typename Fn>
    void for_each_decoded(const std::vector<unsigned long long>& labels,
                          const std::shared_ptr<const Snapshot>& snap,
                          std::size_t* missing,
                          Fn&& fn) const
    {
        const auto& label_map = snap->label_map;
        for (auto lbl : labels) {
            auto it = label_map.find(static_cast<std::uint64_t>(lbl));
            if (it == label_map.end()) {
                if (missing) {
                    ++(*missing);
                }
                continue;
            }
            const auto d = DecodePacked64(it->second, snap->block_size);
            fn(static_cast<std::uint64_t>(lbl), d);
        }
    }

    // 获取当前快照，未加载时抛异常。
    std::shared_ptr<const Snapshot> SnapshotOrThrow() const
    {
        auto snap = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
        if (!snap) {
            throw std::runtime_error("MappedLabelOffsetResolver: map not loaded");
        }
        return snap;
    }

    mutable std::shared_ptr<const Snapshot> snapshot_;
};

} // namespace annfs::io
