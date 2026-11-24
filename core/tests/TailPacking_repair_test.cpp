#include "../impl/tail_packing_repair.hpp"
#include <iostream>
#include <random>

struct Node { unsigned total_size, hop_distance, id; };
static void require(bool ok) { if (!ok) throw std::runtime_error("repair test failed"); }

// Exhaustive subset oracle, independent of the DP recurrence.
static std::vector<Node> oracle(const std::vector<Node>& nodes, unsigned block,
                                std::uint64_t& cursor) {
    std::vector<bool> used(nodes.size());
    std::vector<Node> out;
    for (std::size_t p = 0; p < nodes.size(); ++p) {
        if (used[p]) continue;
        auto rem = cursor % block;
        if (rem && rem + nodes[p].total_size > block) cursor += block - rem;
        out.push_back(nodes[p]); used[p] = true; cursor += nodes[p].total_size;
        const auto cap = (block - cursor % block) % block;
        std::vector<std::size_t> best;
        unsigned best_bytes = 0, best_hops = 0;
        for (unsigned mask = 0; mask < (1u << nodes.size()); ++mask) {
            unsigned bytes = 0, hops = 0; bool valid = true;
            std::vector<std::size_t> ranks;
            for (std::size_t v = 0; v < nodes.size(); ++v) if (mask & (1u << v)) {
                if (used[v]) { valid = false; break; }
                bytes += nodes[v].total_size; hops += nodes[v].hop_distance;
                ranks.push_back(v);
            }
            if (!valid || bytes > cap) continue;
            if (bytes > best_bytes || (bytes == best_bytes &&
                (hops < best_hops || (hops == best_hops && ranks < best)))) {
                best_bytes = bytes; best_hops = hops; best = ranks;
            }
        }
        for (auto v : best) { used[v] = true; out.push_back(nodes[v]); cursor += nodes[v].total_size; }
    }
    return out;
}

static void check(const std::vector<Node>& nodes, unsigned block, std::uint64_t start) {
    auto actual_cursor = start, expected_cursor = start;
    auto actual = annfs_repair::pack_cluster(nodes, block, actual_cursor);
    auto expected = oracle(nodes, block, expected_cursor);
    require(actual.size() == expected.size() && actual_cursor == expected_cursor);
    std::vector<bool> seen(nodes.size());
    auto writer_cursor = start;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].id == expected[i].id && !seen[actual[i].id]);
        seen[actual[i].id] = true;
        auto rem = writer_cursor % block;
        if (rem && rem + actual[i].total_size > block) writer_cursor += block - rem;
        writer_cursor += actual[i].total_size;
    }
    require(writer_cursor == actual_cursor);
}

int main() {
    check({}, 16, 0);
    check({{20,0,0},{12,1,1},{6,2,2},{6,2,3}}, 16, 3); // multi-block tail
    check({{8,0,0},{4,1,1},{4,1,2},{8,2,3}}, 16, 0); // equal depth sum, rank tie
    check({{16,0,0},{3,1,1},{3,1,2}}, 16, 0); // exact boundary
    std::mt19937 rng(20261118);
    for (int trial = 0; trial < 1000; ++trial) {
        std::vector<Node> nodes;
        for (unsigned i = 0, n = rng()%8+1; i < n; ++i)
            nodes.push_back({unsigned(rng()%40+3), i/2, i});
        check(nodes, 16, rng()%32);
    }
    std::uint64_t cursor = 0;
    annfs_repair::pack_cluster(std::vector<Node>{{19,0,0}}, 16, cursor);
    require(cursor == 19);
    annfs_repair::pack_cluster(std::vector<Node>{{3,0,0},{10,1,1}}, 16, cursor);
    require(cursor == 32); // shared cursor across cluster boundary
    std::cout << "repair DP: exhaustive oracle and writer checks passed\n";
}
