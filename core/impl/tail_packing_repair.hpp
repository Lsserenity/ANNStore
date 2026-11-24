#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace annfs_repair {
// Input order is the existing (dist, vertex ID) layout order.
// Share cursor across clusters, starting at the physical writer's offset.
template<class Node>
std::vector<Node> pack_cluster(const std::vector<Node>& nodes,
                              std::uint32_t block, std::uint64_t& cursor) {
    if (!block) throw std::invalid_argument("zero block size");
    struct State {
        bool reachable = false;
        std::uint64_t hops = 0;
        std::vector<std::size_t> ranks;
    };
    std::vector<bool> used(nodes.size(), false);
    std::vector<Node> result;
    result.reserve(nodes.size());
    for (std::size_t primary = 0; primary < nodes.size(); ++primary) {
        if (used[primary]) continue;
        const auto size = nodes[primary].total_size;
        if (!size) throw std::invalid_argument("zero record size");
        const auto rem = cursor % block;
        if (rem && (size > block || size > block - rem)) cursor += block - rem;
        result.push_back(nodes[primary]);
        used[primary] = true;
        cursor += size;
        const std::size_t cap = (block - cursor % block) % block;
        if (!cap) continue;
        std::vector<State> dp(cap + 1);
        dp[0].reachable = true;
        for (std::size_t rank = primary + 1; rank < nodes.size(); ++rank) {
            if (used[rank] || nodes[rank].total_size > cap) continue;
            const std::size_t bytes = nodes[rank].total_size;
            if (!bytes) throw std::invalid_argument("zero record size");
            for (std::size_t b = cap; b >= bytes; --b) {
                if (!dp[b - bytes].reachable) continue;
                // Value snapshots avoid mutable-predecessor reconstruction bugs.
                State candidate = dp[b - bytes];
                candidate.hops += nodes[rank].hop_distance;
                candidate.ranks.push_back(rank);
                if (!dp[b].reachable || candidate.hops < dp[b].hops ||
                    (candidate.hops == dp[b].hops && candidate.ranks < dp[b].ranks))
                    dp[b] = std::move(candidate);
            }
        }
        std::size_t best = cap;
        while (best && !dp[best].reachable) --best;
        for (const auto rank : dp[best].ranks) {
            used[rank] = true;
            result.push_back(nodes[rank]);
            cursor += nodes[rank].total_size;
        }
    }
    return result;
}
} // namespace annfs_repair
