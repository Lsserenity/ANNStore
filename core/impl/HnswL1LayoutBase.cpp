#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <liburing.h>
#include <memory>
#include <mutex>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../config/param_config.hpp"
#include "../io/disk_info.hpp"
#include "../io/raw_trace_block_writer.hpp"
#include "trace_npy_view.hpp"
#include "hnswlib/hnswalg.h"
#include "hnswlib/space_ip.h"
#include "hnswlib/space_l2.h"
#include "hnsw/uint8.hpp"
#include "json.hpp"

namespace {

using tableint = hnswlib::tableint;

struct CliArgs {
    std::string config_path;
    std::string dataset_name;
    std::string trace_path = "DownLoadDataSets/workload_sizes_100M.npy";
    std::string layout_json_path;
    std::string manifest_path;
    // Deprecated legacy arg slot; kept only for CLI compatibility.
    std::string label_map_path;
    std::uint64_t start_offset = 0;
    std::string graph_prefix;
    std::string graph_mode = "auto";  // auto | rebuild | reuse
    std::uint32_t logic_block_bytes = 4096;
    std::uint32_t packed64_cross_bits = 16;
};

struct GraphPersistPaths {
    std::string offsets_path;
    std::string neighbors_path;
    std::string labels_path;
    std::string l1_seeds_path;
    std::string meta_path;
};

struct MMapRO {
    int fd = -1;
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;

    ~MMapRO() {
        if (ptr != nullptr && len > 0) {
            ::munmap(const_cast<std::uint8_t*>(ptr), len);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    MMapRO() = default;
    MMapRO(const MMapRO&) = delete;
    MMapRO& operator=(const MMapRO&) = delete;

    MMapRO(MMapRO&& other) noexcept {
        fd = other.fd;
        ptr = other.ptr;
        len = other.len;
        other.fd = -1;
        other.ptr = nullptr;
        other.len = 0;
    }

    MMapRO& operator=(MMapRO&& other) noexcept {
        if (this == &other) return *this;
        if (ptr != nullptr && len > 0) {
            ::munmap(const_cast<std::uint8_t*>(ptr), len);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        fd = other.fd;
        ptr = other.ptr;
        len = other.len;
        other.fd = -1;
        other.ptr = nullptr;
        other.len = 0;
        return *this;
    }
};

CliArgs parse_args(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: <config_json> <dataset_name> [trace_npy] [layout_json] [manifest] [label_map] [start_offset] [graph_prefix] [graph_mode:auto|rebuild|reuse] [logic_block_bytes=4096] [packed64_cross_bits=16]");
    }
    CliArgs a;
    a.config_path = argv[1];
    a.dataset_name = argv[2];
    if (argc >= 4) a.trace_path = argv[3];
    if (argc >= 5) a.layout_json_path = argv[4];
    if (argc >= 6) a.manifest_path = argv[5];
    if (argc >= 7) a.label_map_path = argv[6];
    if (argc >= 8) a.start_offset = std::stoull(argv[7]);
    if (argc >= 9) a.graph_prefix = argv[8];
    if (argc >= 10) a.graph_mode = argv[9];
    if (argc >= 11) a.logic_block_bytes = static_cast<std::uint32_t>(std::stoul(argv[10]));
    if (argc >= 12) a.packed64_cross_bits = static_cast<std::uint32_t>(std::stoul(argv[11]));
    if (a.layout_json_path.empty()) {
        a.layout_json_path = "./l1_centric_layout_" + a.dataset_name + "_lowmem.json";
    }
    if (a.manifest_path.empty()) {
        a.manifest_path = "./trace_write_manifest_" + a.dataset_name + "_l1centric_blockaligned_lowmem.txt";
    }
    if (a.label_map_path.empty()) {
        a.label_map_path = "./label_map_" + a.dataset_name + "_l1centric_blockaligned_lowmem.txt";
    }
    if (a.graph_prefix.empty()) {
        a.graph_prefix = "./tmp/hnsw_l0_graph_" + a.dataset_name + "_lowmem";
    }
    if (a.graph_mode != "auto" && a.graph_mode != "rebuild" && a.graph_mode != "reuse") {
        throw std::runtime_error("invalid graph_mode: " + a.graph_mode + " (expect auto|rebuild|reuse)");
    }
    if (a.logic_block_bytes == 0) {
        throw std::runtime_error("logic_block_bytes must be > 0");
    }
    if (a.packed64_cross_bits < 1 || a.packed64_cross_bits > 16) {
        throw std::runtime_error("packed64_cross_bits must be in [1,16]");
    }
    return a;
}

struct PersistMeta {
    std::uint64_t node_count = 0;
    std::uint64_t edge_count_directed = 0;
    std::uint64_t l1_seed_count = 0;
    std::int64_t enterpoint = 0;
};

int read_dim_from_data(const std::string& data_path, const std::string& data_format) {
    std::ifstream in(data_path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("failed to open data file: " + data_path);
    std::uint32_t a = 0, b = 0;
    if (data_format == "bin" || data_format == "fbin" || data_format == "bbin") {
        in.read(reinterpret_cast<char*>(&a), sizeof(a));
        in.read(reinterpret_cast<char*>(&b), sizeof(b));
        if (!in.good() || b == 0) throw std::runtime_error("invalid bin header");
        return static_cast<int>(b);
    }
    if (data_format == "vecs" || data_format == "fvecs" || data_format == "bvecs") {
        in.read(reinterpret_cast<char*>(&a), sizeof(a));
        if (!in.good() || a == 0) throw std::runtime_error("invalid vecs header");
        return static_cast<int>(a);
    }
    throw std::runtime_error("unsupported data format: " + data_format);
}

std::unique_ptr<hnswlib::SpaceInterface<float>> create_space(const DatasetInfo& ds, int dim) {
    if (ds.data_type == "uint8" || ds.data_type == "u8") return std::make_unique<hnswlib::L2SpaceUint8>(dim);
    if (ds.distance_metric == "ip") return std::make_unique<hnswlib::InnerProductSpace>(dim);
    return std::make_unique<hnswlib::L2Space>(dim);
}
// 获取节点的l0层的邻居
std::vector<tableint> collect_neighbors_l0(hnswlib::HierarchicalNSW<float>& index, tableint node) {
    std::vector<tableint> out;
    std::unique_lock<std::mutex> lock(index.link_list_locks_[node]);
    auto* ll = index.get_linklist_at_level(node, 0);
    const size_t degree = index.getListCount(ll);
    auto* data = reinterpret_cast<tableint*>(ll + 1);
    out.reserve(degree);
    for (size_t i = 0; i < degree; ++i) out.push_back(data[i]);
    return out;
}

inline bool bit_get(const std::vector<std::uint8_t>& bits, std::uint64_t idx) {
    return (bits[idx >> 3] >> (idx & 7U)) & 1U;
}

inline void bit_set(std::vector<std::uint8_t>& bits, std::uint64_t idx) {
    bits[idx >> 3] = static_cast<std::uint8_t>(bits[idx >> 3] | (1U << (idx & 7U)));
}

MMapRO mmap_ro_file(const std::string& path) {
    MMapRO mm;
    mm.fd = ::open(path.c_str(), O_RDONLY);
    if (mm.fd < 0) throw std::runtime_error("open failed: " + path);

    struct stat st {};
    if (::fstat(mm.fd, &st) != 0) {
        throw std::runtime_error("fstat failed: " + path);
    }
    if (st.st_size < 0) {
        throw std::runtime_error("invalid file size: " + path);
    }

    mm.len = static_cast<std::size_t>(st.st_size);
    if (mm.len == 0) {
        return mm;
    }

    mm.ptr = static_cast<const std::uint8_t*>(::mmap(nullptr, mm.len, PROT_READ, MAP_PRIVATE, mm.fd, 0));
    if (mm.ptr == MAP_FAILED) {
        mm.ptr = nullptr;
        throw std::runtime_error("mmap failed: " + path);
    }
    return mm;
}

bool file_exists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec) && !ec;
}

PersistMeta load_meta(const std::string& meta_path) {
    std::ifstream in(meta_path);
    if (!in.is_open()) throw std::runtime_error("cannot open graph meta: " + meta_path);
    nlohmann::json js;
    in >> js;
    PersistMeta m;
    m.node_count = js.value("node_count", 0ULL);
    m.edge_count_directed = js.value("edge_count_directed", 0ULL);
    m.l1_seed_count = js.value("l1_seed_count", 0ULL);
    m.enterpoint = js.value("enterpoint", 0LL);
    if (m.node_count == 0) {
        throw std::runtime_error("invalid node_count in graph meta: " + meta_path);
    }
    return m;
}

void save_meta(const std::string& meta_path, const PersistMeta& m) {
    nlohmann::json js;
    js["node_count"] = m.node_count;
    js["edge_count_directed"] = m.edge_count_directed;
    js["l1_seed_count"] = m.l1_seed_count;
    js["enterpoint"] = m.enterpoint;
    std::ofstream out(meta_path);
    if (!out.is_open()) throw std::runtime_error("cannot write graph meta: " + meta_path);
    out << js.dump(2);
}

std::vector<std::uint64_t> load_offsets(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open offsets file: " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff len = in.tellg();
    if (len < 0 || (len % static_cast<std::streamoff>(sizeof(std::uint64_t))) != 0) {
        throw std::runtime_error("invalid offsets file size: " + path);
    }
    const std::size_t n64 = static_cast<std::size_t>(len / static_cast<std::streamoff>(sizeof(std::uint64_t)));
    if (n64 < 2) throw std::runtime_error("offsets file too small: " + path);
    in.seekg(0, std::ios::beg);
    std::vector<std::uint64_t> offsets(n64, 0);
    in.read(reinterpret_cast<char*>(offsets.data()), static_cast<std::streamsize>(len));
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("failed reading offsets file: " + path);
    }
    return offsets;
}

template <typename T>
void save_vector_binary(const std::string& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("cannot write binary file: " + path);
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
}

template <typename T>
std::vector<T> load_vector_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open binary file: " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff len = in.tellg();
    if (len < 0 || (len % static_cast<std::streamoff>(sizeof(T))) != 0) {
        throw std::runtime_error("invalid binary file size: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<T> data(static_cast<std::size_t>(len / static_cast<std::streamoff>(sizeof(T))));
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), len);
    }
    return data;
}

void persist_graph_and_labels(
    hnswlib::HierarchicalNSW<float>& index,
    const GraphPersistPaths& p,
    std::vector<tableint>& l1_seeds,
    std::uint64_t& total_edges,
    std::vector<std::uint64_t>& offsets) {
    const tableint n = static_cast<tableint>(index.cur_element_count.load());
    if (n <= 0) throw std::runtime_error("empty index");

    std::vector<std::uint32_t> deg(static_cast<std::size_t>(n), 0);
    l1_seeds.clear();
    l1_seeds.reserve(static_cast<std::size_t>(n / 10 + 1));
    // 获取l1节点
    for (tableint u = 0; u < n; ++u) {
        if (index.element_levels_[u] >= 1) {
            l1_seeds.push_back(u);
        }
        const auto nbs = collect_neighbors_l0(index, u);
        std::uint32_t d = 0;
        for (auto v : nbs) {
            if (v < n) ++d;
        }
        deg[static_cast<std::size_t>(u)] = d;
    }

    if (l1_seeds.empty()) {
        l1_seeds.push_back(static_cast<tableint>(index.enterpoint_node_));
    }

    offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    for (tableint u = 0; u < n; ++u) {
        offsets[static_cast<std::size_t>(u) + 1] =
            offsets[static_cast<std::size_t>(u)] + static_cast<std::uint64_t>(deg[static_cast<std::size_t>(u)]);
    }
    total_edges = offsets.back();

    {
        std::ofstream out(p.offsets_path, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("cannot open offsets file: " + p.offsets_path);
        out.write(reinterpret_cast<const char*>(offsets.data()),
                  static_cast<std::streamsize>(offsets.size() * sizeof(std::uint64_t)));
    }

    {
        std::ofstream out(p.neighbors_path, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("cannot open neighbors file: " + p.neighbors_path);
        std::vector<tableint> row;
        for (tableint u = 0; u < n; ++u) {
            row.clear();
            const auto nbs = collect_neighbors_l0(index, u);
            row.reserve(nbs.size());
            for (auto v : nbs) {
                if (v < n) row.push_back(v);
            }
            if (!row.empty()) {
                out.write(reinterpret_cast<const char*>(row.data()),
                          static_cast<std::streamsize>(row.size() * sizeof(tableint)));
            }
        }
    }

    {
        std::ofstream out(p.labels_path, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("cannot open labels file: " + p.labels_path);
        for (tableint u = 0; u < n; ++u) {
            const std::uint64_t label = static_cast<std::uint64_t>(index.getExternalLabel(u));
            out.write(reinterpret_cast<const char*>(&label), sizeof(label));
        }
    }

    save_vector_binary<tableint>(p.l1_seeds_path, l1_seeds);
}

struct MultiSourcePartitionResult {
    std::vector<tableint> owner;
    std::vector<int> dist;
    std::vector<tableint> seeds;
    std::vector<tableint> pseudo_seeds;
};

// 1. l1 分簇
MultiSourcePartitionResult partition_by_l1_multisource_directed(
    const std::vector<std::uint64_t>& offsets,
    const tableint* neighbors,
    tableint n,
    std::vector<tableint> l1_seeds) {
    MultiSourcePartitionResult r;
    r.owner.assign(static_cast<std::size_t>(n), static_cast<tableint>(-1));
    r.dist.assign(static_cast<std::size_t>(n), -1);
    r.seeds = std::move(l1_seeds);

    {
        std::unordered_set<tableint> seen;
        seen.reserve(r.seeds.size() * 2 + 1);
        std::vector<tableint> deduped;
        deduped.reserve(r.seeds.size());
        for (auto s : r.seeds) {
            if (s < 0 || s >= n) continue;
            if (seen.insert(s).second) deduped.push_back(s);
        }
        r.seeds.swap(deduped);
    }

    std::queue<tableint> q;
    for (auto s : r.seeds) {
        r.owner[static_cast<std::size_t>(s)] = s;
        r.dist[static_cast<std::size_t>(s)] = 0;
        q.push(s);
    }
    while (!q.empty()) {
        const tableint u = q.front();
        q.pop();
        const std::uint64_t beg = offsets[static_cast<std::size_t>(u)];
        const std::uint64_t end = offsets[static_cast<std::size_t>(u) + 1];
        for (std::uint64_t i = beg; i < end; ++i) {
            const tableint v = neighbors[i];
            if (v < 0 || v >= n) continue;
            if (r.owner[static_cast<std::size_t>(v)] != static_cast<tableint>(-1)) continue;
            r.owner[static_cast<std::size_t>(v)] = r.owner[static_cast<std::size_t>(u)];
            r.dist[static_cast<std::size_t>(v)] = r.dist[static_cast<std::size_t>(u)] + 1;
            q.push(v);
        }
    }
    // 2. pseudo-seed --- 按 internal id 扫描补全
    // 按 u=0..n-1 扫描，先遇到的未分配节点先当 pseudo-seed，再 BFS 吃掉该分量
    // cover disconnected components: bootstrap each unassigned L0 node as pseudo seed.
    for (tableint u = 0; u < n; ++u) {
        if (r.owner[static_cast<std::size_t>(u)] != static_cast<tableint>(-1)) continue;
        const tableint ps = u;
        r.pseudo_seeds.push_back(ps);
        r.seeds.push_back(ps);
        r.owner[static_cast<std::size_t>(u)] = ps;
        r.dist[static_cast<std::size_t>(u)] = 0;
        q.push(u);
        while (!q.empty()) {
            const tableint x = q.front();
            q.pop();
            const std::uint64_t beg = offsets[static_cast<std::size_t>(x)];
            const std::uint64_t end = offsets[static_cast<std::size_t>(x) + 1];
            for (std::uint64_t i = beg; i < end; ++i) {
                const tableint y = neighbors[i];
                if (y < 0 || y >= n) continue;
                if (r.owner[static_cast<std::size_t>(y)] != static_cast<tableint>(-1)) continue;
                r.owner[static_cast<std::size_t>(y)] = ps;
                r.dist[static_cast<std::size_t>(y)] = r.dist[static_cast<std::size_t>(x)] + 1;
                q.push(y);
            }
        }
    }

    return r;
}

// 2. 构建簇之间的顺序
std::vector<tableint> build_cluster_order_from_owner(
    const std::vector<std::uint64_t>& offsets,
    const tableint* neighbors,
    tableint n,
    const std::vector<tableint>& owner,
    tableint enter_seed) {
    // seeds: 去重后的“簇代表 seed 列表”（值就是 owner 的取值，通常是 L1 seed 或 pseudo-seed）
    // id2idx: seed值 -> 簇编号[0..k-1]，用于把“稀疏seed值空间”压缩成连续下标，方便后续建图。
    std::vector<tableint> seeds;
    seeds.reserve(static_cast<std::size_t>(n / 8 + 1));
    std::unordered_map<tableint, int> id2idx;
    for (tableint u = 0; u < n; ++u) {
        const tableint s = owner[static_cast<std::size_t>(u)];
        if (s == static_cast<tableint>(-1)) continue;
        auto [it, inserted] = id2idx.emplace(s, static_cast<int>(seeds.size()));
        if (inserted) seeds.push_back(s);
    }
    const int k = static_cast<int>(seeds.size());
    if (k == 0) return {};

    // cg = cluster graph（簇图）邻接表：
    // cg[i] 存“与簇 i 相邻的簇编号集合”。
    // 如果原图边 u->v 跨簇（owner[u] != owner[v]），就在簇图上连边 cu<->cv。
    std::vector<std::unordered_set<int>> cg(static_cast<std::size_t>(k));
    for (tableint u = 0; u < n; ++u) {
        const int cu = id2idx[owner[static_cast<std::size_t>(u)]];
        const std::uint64_t beg = offsets[static_cast<std::size_t>(u)];
        const std::uint64_t end = offsets[static_cast<std::size_t>(u) + 1];
        for (std::uint64_t i = beg; i < end; ++i) {
            const tableint v = neighbors[i];
            if (v < 0 || v >= n) continue;
            const int cv = id2idx[owner[static_cast<std::size_t>(v)]];
            if (cu != cv) {
                cg[static_cast<std::size_t>(cu)].insert(cv);
                cg[static_cast<std::size_t>(cv)].insert(cu);
            }
        }
    }

    // 对“簇图”做 BFS 得到簇访问顺序：
    // out: 最终簇顺序（存的是seed值，不是簇编号）
    // vis: 标记簇图节点是否已访问
    // q: 簇图 BFS 队列（队列元素是簇编号）
    std::vector<tableint> out;
    out.reserve(seeds.size());
    std::vector<unsigned char> vis(static_cast<std::size_t>(k), 0);
    std::queue<int> q;
    auto it = id2idx.find(enter_seed);
    const int start = (it == id2idx.end()) ? 0 : it->second;
    vis[static_cast<std::size_t>(start)] = 1;
    q.push(start);
    while (!q.empty()) {
        const int c = q.front();
        q.pop();
        out.push_back(seeds[static_cast<std::size_t>(c)]);
        // 这里先把邻接簇收集到 next，再按 seed值升序，保证结果稳定可复现
        // （否则 unordered_set 的迭代顺序不稳定，不同进程/编译器下可能不同）。
        std::vector<int> next(cg[static_cast<std::size_t>(c)].begin(), cg[static_cast<std::size_t>(c)].end());
        std::sort(next.begin(), next.end(), [&](int a, int b) {
            return seeds[static_cast<std::size_t>(a)] < seeds[static_cast<std::size_t>(b)];
        });
        for (int t : next) {
            if (!vis[static_cast<std::size_t>(t)]) {
                vis[static_cast<std::size_t>(t)] = 1;
                q.push(t);
            }
        }
    }
    // 对簇图中“与起点不连通”的簇做兜底追加，确保全覆盖。
    // 追加顺序按簇编号线性扫描（对应 seeds 向量顺序）。
    for (int i = 0; i < k; ++i) {
        if (!vis[static_cast<std::size_t>(i)]) out.push_back(seeds[static_cast<std::size_t>(i)]);
    }
    return out;
}
// 簇内顺序代码
std::vector<tableint> build_clustered_write_order(
    tableint n,
    const std::vector<tableint>& owner,
    const std::vector<int>& dist,
    const std::vector<std::uint64_t>& offsets,
    const std::vector<tableint>& cluster_order) {
    std::unordered_map<tableint, std::vector<tableint>> cluster_nodes;
    cluster_nodes.reserve(cluster_order.size() * 2 + 1);
    for (tableint u = 0; u < n; ++u) {
        const auto s = owner[static_cast<std::size_t>(u)];
        if (s != static_cast<tableint>(-1)) {
            cluster_nodes[s].push_back(u);
        }
    }

    std::vector<tableint> write_order;
    write_order.reserve(static_cast<std::size_t>(n));
    for (const auto seed : cluster_order) {
        auto it = cluster_nodes.find(seed);
        if (it == cluster_nodes.end()) continue;
        auto& nodes = it->second;
        // base 模式簇内顺序（轻量拓扑顺序）：
        // 1) dist 升序：越靠近 seed 的节点越先写；
        // 2) internal id 升序：同层节点用稳定 tie-break，保证可复现。
        //
        // 这里不引入 payload 尺寸，也不做装箱优化；
        // base 仅负责“分簇 + 簇序 + 轻量簇内顺序”。
        std::sort(nodes.begin(), nodes.end(), [&](tableint a, tableint b) {
            if (dist[static_cast<std::size_t>(a)] != dist[static_cast<std::size_t>(b)]) {
                return dist[static_cast<std::size_t>(a)] < dist[static_cast<std::size_t>(b)];
            }
            return a < b;
        });
        write_order.insert(write_order.end(), nodes.begin(), nodes.end());
    }
    return write_order;
}

std::uint32_t resolve_size(
    tableint node_id,
    std::uint64_t label,
    const annfs::io::TraceView& trace,
    std::uint64_t safe_upper,
    std::size_t& fallback_count,
    std::size_t& invalid_count) {
    std::uint64_t idx = label;
    if (idx >= safe_upper) {
        idx = static_cast<std::uint64_t>(node_id);
        ++fallback_count;
    }
    if (idx >= safe_upper) {
        ++invalid_count;
        return 1;
    }
    const std::int32_t v = trace.data[idx];
    if (v <= 0) {
        ++invalid_count;
        return 1;
    }
    return static_cast<std::uint32_t>(v);
}

struct StreamWriteStats {
    std::size_t written_records = 0;
    std::uint64_t total_payload_bytes = 0;
    std::uint64_t total_record_bytes = 0;
};

inline std::uint64_t pack_label_64(std::uint64_t crossed_blocks,
                                   std::uint64_t start_byte,
                                   std::uint32_t cross_bits) {
    // 64位标签编码：高 cross_bits 位保存 crossed_blocks，低位保存 start_byte。
    // 默认 cross_bits=16 => [16 bits crossed][48 bits start_byte]。
    const std::uint32_t shift = 64U - cross_bits;
    const std::uint64_t cross_limit = (cross_bits >= 64U)
        ? std::numeric_limits<std::uint64_t>::max()
        : ((1ULL << cross_bits) - 1ULL);
    const std::uint64_t start_limit = (shift >= 64U)
        ? std::numeric_limits<std::uint64_t>::max()
        : (1ULL << shift);
    if (crossed_blocks > cross_limit) {
        throw std::runtime_error("crossed_blocks overflow for packed64");
    }
    if (start_byte >= start_limit) {
        throw std::runtime_error("start_byte overflow for packed64");
    }
    return (crossed_blocks << shift) | start_byte;
}

void write_internal_to_packed64_map(
    const std::string& out_path,
    const std::uint64_t* labels,
    tableint n,
    const std::unordered_map<std::uint64_t, std::uint64_t>& label_to_packed64) {
    // 直接输出 internal_id -> packed64_label，供 HnswRelabelIndex 直接使用，
    // 避免脚本再从 manifest 二次计算。
    std::ofstream out(out_path);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open label_map output: " + out_path);
    }
    out << "# internal_id packed64_label\n";
    std::unordered_map<std::uint64_t, tableint> seen_packed;
    seen_packed.reserve(static_cast<std::size_t>(n));
    std::size_t missing = 0;
    for (tableint i = 0; i < n; ++i) {
        const std::uint64_t lbl = labels[static_cast<std::size_t>(i)];
        auto it = label_to_packed64.find(lbl);
        if (it == label_to_packed64.end()) {
            ++missing;
            continue;
        }
        const std::uint64_t packed64 = it->second;
        auto ok = seen_packed.emplace(packed64, i);
        if (!ok.second) {
            throw std::runtime_error(
                "duplicate packed64 label detected: " + std::to_string(packed64) +
                " (internal_id=" + std::to_string(i) +
                ", existing_internal_id=" + std::to_string(ok.first->second) + ")");
        }
        out << i << ' ' << packed64 << '\n';
    }
    if (missing > 0) {
        throw std::runtime_error("missing packed64 labels for " + std::to_string(missing) + " internal ids");
    }
}

StreamWriteStats stream_write_block_aligned(
    const std::string& dev_path,
    std::uint32_t logic_block_size,
    std::uint64_t start_offset_bytes,
    const std::vector<tableint>& write_order,
    const std::uint64_t* labels,
    const annfs::io::TraceView& trace,
    std::uint64_t safe_upper,
    std::ofstream& manifest_out,
    std::uint32_t packed64_cross_bits,
    std::unordered_map<std::uint64_t, std::uint64_t>& label_to_packed64,
    std::size_t& fallback_count,
    std::size_t& invalid_count,
    unsigned queue_depth) {
    if (logic_block_size == 0) throw std::invalid_argument("logic_block_size must be > 0");
    if (queue_depth == 0) throw std::invalid_argument("queue_depth must be > 0");

    auto env_true = [](const char* v) {
        if (!v) return false;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(s.empty() || s == "0" || s == "false" || s == "off" || s == "no");
    };
    const bool skip_device_write = env_true(std::getenv("ANNFS_WRITER_SKIP_DEVICE_WRITE"));

    StreamWriteStats stats;
    std::uint64_t cur = start_offset_bytes;
    if (skip_device_write) {
        for (auto node_id : write_order) {
            const std::uint64_t label = labels[static_cast<std::size_t>(node_id)];
            const std::uint32_t payload_size = resolve_size(
                node_id, label, trace, safe_upper, fallback_count, invalid_count);
            const std::uint32_t total_size = payload_size + 2u;

            const std::uint64_t rem = cur % logic_block_size;
            if (rem != 0 && rem + total_size > logic_block_size) {
                cur += (logic_block_size - rem);
            }

            const std::uint16_t end_off =
                static_cast<std::uint16_t>((cur + total_size - 1) % logic_block_size);
            // 2字节头部写的是 end_off：
            // “当前 record(2B头+payload) 最后一个字节”在逻辑块内的 zero-based 偏移。
            const std::uint64_t block_span = static_cast<std::uint64_t>(
                ((cur % logic_block_size) + total_size - 1) / logic_block_size);
            // block_span 是“跨越的额外块数”：
            // 0=不跨块，1=跨1个额外块，...，后续编码到 packed64 高位。
            const std::uint64_t packed64 = pack_label_64(block_span, cur, packed64_cross_bits);
            auto old = label_to_packed64.find(label);
            if (old != label_to_packed64.end() && old->second != packed64) {
                throw std::runtime_error("inconsistent packed64 for same label");
            }
            label_to_packed64[label] = packed64;

            manifest_out << label << ' '
                         << cur << ' '
                         << payload_size << ' '
                         << total_size << ' '
                         << end_off << ' '
                         << block_span << ' '
                         << packed64 << "\n";

            ++stats.written_records;
            stats.total_payload_bytes += payload_size;
            stats.total_record_bytes += total_size;
            cur += total_size;
        }
        return stats;
    }

    int fd = ::open(dev_path.c_str(), O_WRONLY);
    if (fd < 0) throw std::runtime_error("open device failed: " + dev_path);

    io_uring ring {};
    if (io_uring_queue_init(queue_depth, &ring, 0) != 0) {
        ::close(fd);
        throw std::runtime_error("io_uring_queue_init failed");
    }

    struct InFlightSlot {
        bool used = false;
        std::vector<char> buf;
        std::size_t len = 0;
    };
    std::vector<InFlightSlot> slots(queue_depth);
    int inflight = 0;

    auto reap_one = [&]() {
        io_uring_cqe* cqe = nullptr;
        const int rc = io_uring_wait_cqe(&ring, &cqe);
        if (rc < 0 || cqe == nullptr) {
            throw std::runtime_error("io_uring_wait_cqe failed");
        }
        const std::uint64_t tag = io_uring_cqe_get_data64(cqe);
        const std::size_t slot_id = static_cast<std::size_t>(tag);
        if (slot_id >= slots.size() || !slots[slot_id].used) {
            io_uring_cqe_seen(&ring, cqe);
            throw std::runtime_error("invalid cqe slot id");
        }
        if (cqe->res < 0) {
            const int err = -cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            throw std::runtime_error("io_uring write failed: errno=" + std::to_string(err));
        }
        if (static_cast<std::size_t>(cqe->res) != slots[slot_id].len) {
            io_uring_cqe_seen(&ring, cqe);
            throw std::runtime_error("short write from io_uring");
        }
        slots[slot_id].buf.clear();
        slots[slot_id].len = 0;
        slots[slot_id].used = false;
        --inflight;
        io_uring_cqe_seen(&ring, cqe);
    };

    auto acquire_slot = [&]() -> std::size_t {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (!slots[i].used) return i;
        }
        return static_cast<std::size_t>(-1);
    };

    try {
        for (auto node_id : write_order) {
            while (inflight >= static_cast<int>(queue_depth)) {
                reap_one();
            }

            const std::uint64_t label = labels[static_cast<std::size_t>(node_id)];
            const std::uint32_t payload_size = resolve_size(
                node_id, label, trace, safe_upper, fallback_count, invalid_count);
            const std::uint32_t total_size = payload_size + 2u;

            const std::uint64_t rem = cur % logic_block_size;
            if (rem != 0 && rem + total_size > logic_block_size) {
                cur += (logic_block_size - rem);
            }

            std::vector<char> buf(total_size, 'z');
            const std::uint16_t end_off =
                static_cast<std::uint16_t>((cur + total_size - 1) % logic_block_size);
            std::memcpy(buf.data(), &end_off, sizeof(end_off));

            const std::uint64_t block_span = static_cast<std::uint64_t>(
                ((cur % logic_block_size) + total_size - 1) / logic_block_size);
            const std::uint64_t packed64 = pack_label_64(block_span, cur, packed64_cross_bits);
            auto old = label_to_packed64.find(label);
            if (old != label_to_packed64.end() && old->second != packed64) {
                throw std::runtime_error("inconsistent packed64 for same label");
            }
            label_to_packed64[label] = packed64;

            std::size_t slot_id = acquire_slot();
            if (slot_id == static_cast<std::size_t>(-1)) {
                reap_one();
                slot_id = acquire_slot();
                if (slot_id == static_cast<std::size_t>(-1)) {
                    throw std::runtime_error("no free io_uring slot");
                }
            }

            slots[slot_id].used = true;
            slots[slot_id].len = buf.size();
            slots[slot_id].buf = std::move(buf);

            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (sqe == nullptr) {
                io_uring_submit(&ring);
                sqe = io_uring_get_sqe(&ring);
            }
            if (sqe == nullptr) {
                throw std::runtime_error("io_uring_get_sqe failed");
            }

            io_uring_prep_write(
                sqe,
                fd,
                slots[slot_id].buf.data(),
                static_cast<unsigned>(slots[slot_id].len),
                static_cast<off_t>(cur));
            io_uring_sqe_set_data64(sqe, static_cast<std::uint64_t>(slot_id));
            ++inflight;

            manifest_out << label << ' '
                         << cur << ' '
                         << payload_size << ' '
                         << total_size << ' '
                         << end_off << ' '
                         << block_span << ' '
                         << packed64 << "\n";

            ++stats.written_records;
            stats.total_payload_bytes += payload_size;
            stats.total_record_bytes += total_size;

            cur += total_size;

            if (io_uring_sq_space_left(&ring) == 0) {
                io_uring_submit(&ring);
            }
        }

        io_uring_submit(&ring);
        while (inflight > 0) {
            reap_one();
        }
    } catch (...) {
        io_uring_queue_exit(&ring);
        ::close(fd);
        throw;
    }

    io_uring_queue_exit(&ring);
    ::close(fd);
    return stats;
}

void save_layout_json(const std::string& path,
                      const std::string& graph_offsets_path,
                      const std::string& graph_neighbors_path,
                      const std::string& graph_l1_seeds_path,
                      std::size_t n,
                      std::size_t l1_seed_count,
                      std::size_t pseudo_seed_count,
                      std::size_t cluster_count,
                      std::size_t write_order_size,
                      std::uint64_t total_edges,
                      bool complete_coverage) {
    nlohmann::json js;
    js["method"] = {
        {"name", "persist_l0_relations_then_release_index_then_l1_multisource_partition_order"},
        {"steps", {
            "stream L0 neighbors from index and persist to binary graph files",
            "persist internal_id->original_label to binary file",
            "release index memory",
            "load persisted graph and run L1 multi-source BFS partition",
            "bootstrap unassigned components with L0 pseudo-seeds",
            "build cluster graph order from enter_seed",
            "stream write trace blocks + manifest"
        }}
    };
    js["graph_offsets_path"] = graph_offsets_path;
    js["graph_neighbors_path"] = graph_neighbors_path;
    js["graph_l1_seeds_path"] = graph_l1_seeds_path;
    js["node_count"] = n;
    js["edge_count_directed"] = total_edges;
    js["l1_seed_count"] = l1_seed_count;
    js["pseudo_seed_count"] = pseudo_seed_count;
    js["cluster_count"] = cluster_count;
    js["write_order_size"] = write_order_size;
    js["complete_coverage"] = complete_coverage;

    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("cannot open layout json: " + path);
    out << js.dump(2);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args = parse_args(argc, argv);

        std::ifstream in(args.config_path);
        if (!in.is_open()) throw std::runtime_error("cannot open config json");
        nlohmann::json js;
        in >> js;

        ParamConfig& pc = ParamConfig::instance();
        pc.load_from_json(js, args.dataset_name);

        const std::string graph_prefix = args.graph_prefix;
        const GraphPersistPaths gp {
            graph_prefix + ".offsets.bin",
            graph_prefix + ".neighbors.bin",
            graph_prefix + ".labels.bin",
            graph_prefix + ".l1_seeds.bin",
            graph_prefix + ".meta.json"
        };

        tableint n = 0;
        tableint enterpoint = 0;
        std::size_t l1_seed_count = 0;
        std::vector<std::uint64_t> offsets;
        std::uint64_t total_edges = 0;
        std::string graph_action = "rebuild";

        const bool graph_files_exist =
            file_exists(gp.offsets_path) &&
            file_exists(gp.neighbors_path) &&
            file_exists(gp.labels_path) &&
            file_exists(gp.l1_seeds_path) &&
            file_exists(gp.meta_path);

        bool should_rebuild = false;
        if (args.graph_mode == "rebuild") {
            should_rebuild = true;
            graph_action = "rebuild";
        } else if (args.graph_mode == "reuse") {
            if (!graph_files_exist) {
                throw std::runtime_error("graph_mode=reuse but graph files are missing under prefix: " + graph_prefix);
            }
            should_rebuild = false;
            graph_action = "reuse";
        } else {
            should_rebuild = !graph_files_exist;
            graph_action = should_rebuild ? "rebuild(auto)" : "reuse(auto)";
        }

        if (should_rebuild) {
            std::vector<tableint> l1_seeds;
            {
                const int dim = read_dim_from_data(pc.dataset.data_path, pc.dataset.data_format);
                auto space = create_space(pc.dataset, dim);
                hnswlib::HierarchicalNSW<float> index(space.get(), pc.dataset.hnsw_index_path);
                n = static_cast<tableint>(index.cur_element_count.load());
                if (n == 0) throw std::runtime_error("empty index");
                enterpoint = static_cast<tableint>(index.enterpoint_node_);

                persist_graph_and_labels(index, gp, l1_seeds, total_edges, offsets);
            }
            l1_seed_count = l1_seeds.size();
            PersistMeta meta;
            meta.node_count = static_cast<std::uint64_t>(n);
            meta.edge_count_directed = total_edges;
            meta.l1_seed_count = static_cast<std::uint64_t>(l1_seed_count);
            meta.enterpoint = static_cast<std::int64_t>(enterpoint);
            save_meta(gp.meta_path, meta);
        } else {
            const PersistMeta meta = load_meta(gp.meta_path);
            n = static_cast<tableint>(meta.node_count);
            if (n <= 0) throw std::runtime_error("invalid node_count in graph meta");
            enterpoint = static_cast<tableint>(meta.enterpoint);
            l1_seed_count = static_cast<std::size_t>(meta.l1_seed_count);
            offsets = load_offsets(gp.offsets_path);
            if (offsets.size() != static_cast<std::size_t>(n) + 1U) {
                throw std::runtime_error("offsets count mismatch with meta node_count");
            }
            total_edges = offsets.back();
        }

        const MMapRO mm_neighbors = mmap_ro_file(gp.neighbors_path);
        const MMapRO mm_labels = mmap_ro_file(gp.labels_path);

        if (mm_neighbors.len != static_cast<std::size_t>(total_edges) * sizeof(tableint)) {
            throw std::runtime_error("neighbors file size mismatch");
        }
        if (mm_labels.len != static_cast<std::size_t>(n) * sizeof(std::uint64_t)) {
            throw std::runtime_error("labels file size mismatch");
        }

        const auto* neighbors = reinterpret_cast<const tableint*>(mm_neighbors.ptr);
        const auto* labels = reinterpret_cast<const std::uint64_t*>(mm_labels.ptr);
        const auto l1_seeds = load_vector_binary<tableint>(gp.l1_seeds_path);
        if (l1_seeds.empty()) {
            throw std::runtime_error("l1 seeds file is empty: " + gp.l1_seeds_path);
        }

        const auto part = partition_by_l1_multisource_directed(
            offsets, neighbors, n, l1_seeds);
        tableint enter_seed = (enterpoint >= 0 && enterpoint < n)
            ? part.owner[static_cast<std::size_t>(enterpoint)]
            : part.seeds.front();
        if (enter_seed == static_cast<tableint>(-1)) {
            enter_seed = part.seeds.front();
        }
        const auto cluster_order = build_cluster_order_from_owner(
            offsets, neighbors, n, part.owner, enter_seed);
        const auto write_order = build_clustered_write_order(
            n, part.owner, part.dist, offsets, cluster_order);
        if (write_order.size() != static_cast<std::size_t>(n)) {
            throw std::runtime_error("write_order not full coverage");
        }

        annfs::io::TraceView trace = annfs::io::load_trace_npy_int32(args.trace_path);
        const std::uint64_t safe_upper = std::min<std::uint64_t>(
            static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(trace.count));

        const std::string dev_path = pc.dev_path;
        if (dev_path.empty()) throw std::runtime_error("dev_path is empty in config");
        const std::uint32_t logic_block_size = args.logic_block_bytes;
        const unsigned queue_depth = (GlobalConfig::instance().RING_DEPTH > 0)
            ? static_cast<unsigned>(GlobalConfig::instance().RING_DEPTH)
            : 256u;

        std::ofstream manifest_out(args.manifest_path);
        if (!manifest_out.is_open()) {
            throw std::runtime_error("cannot open manifest output: " + args.manifest_path);
        }
        manifest_out << "# dev_path " << dev_path << "\n";
        manifest_out << "# logic_block_size " << logic_block_size << "\n";
        manifest_out << "# end_offset is ZERO-BASED within logic block: ((start_byte % block_size) + (2 + payload_size) - 1) % block_size\n";
        manifest_out << "# packed64 format: [cross_bits][start_byte_bits], cross_bits=" << args.packed64_cross_bits << "\n";
        manifest_out << "# format: source_label start_byte payload_size total_size end_offset_zero_based crossed_blocks packed64_label\n";

        std::size_t fallback_count = 0;
        std::size_t invalid_count = 0;
        std::unordered_map<std::uint64_t, std::uint64_t> label_to_packed64;
        label_to_packed64.reserve(static_cast<std::size_t>(n));
        const auto stats = stream_write_block_aligned(
            dev_path,
            logic_block_size,
            args.start_offset,
            write_order,
            labels,
            trace,
            safe_upper,
            manifest_out,
            args.packed64_cross_bits,
            label_to_packed64,
            fallback_count,
            invalid_count,
            queue_depth);
        write_internal_to_packed64_map(
            args.label_map_path,
            labels,
            n,
            label_to_packed64);

        save_layout_json(args.layout_json_path,
                         gp.offsets_path,
                         gp.neighbors_path,
                         gp.l1_seeds_path,
                         static_cast<std::size_t>(n),
                         l1_seed_count,
                         part.pseudo_seeds.size(),
                         cluster_order.size(),
                         write_order.size(),
                         total_edges,
                         write_order.size() == static_cast<std::size_t>(n));

        std::cout << "Dataset           : " << pc.dataset.dataset_name << "\n";
        std::cout << "Index path        : " << pc.dataset.hnsw_index_path << "\n";
        std::cout << "Graph prefix      : " << graph_prefix << "\n";
        std::cout << "Graph mode        : " << args.graph_mode << "\n";
        std::cout << "Graph action      : " << graph_action << "\n";
        std::cout << "L0 nodes          : " << n << "\n";
        std::cout << "L1 seeds          : " << l1_seed_count << "\n";
        std::cout << "Pseudo seeds      : " << part.pseudo_seeds.size() << "\n";
        std::cout << "Directed edges    : " << total_edges << "\n";
        std::cout << "Cluster count     : " << cluster_order.size() << "\n";
        std::cout << "Write order size  : " << write_order.size() << " (full coverage)\n";
        std::cout << "Trace upper bound : " << safe_upper << "\n";
        std::cout << "Trace fallback    : " << fallback_count << "\n";
        std::cout << "Trace invalid<=0  : " << invalid_count << " (size->1)\n";
        std::cout << "Trace file        : " << args.trace_path << "\n";
        std::cout << "Start offset      : " << args.start_offset << "\n";
        std::cout << "Device            : " << dev_path << "\n";
        std::cout << "Logic block size  : " << logic_block_size << "\n";
        std::cout << "Packed64 crossbit : " << args.packed64_cross_bits << "\n";
        std::cout << "Config io_depth   : " << GlobalConfig::instance().RING_DEPTH << "\n";
        std::cout << "Queue depth       : " << queue_depth << "\n";
        std::cout << "Total payload B   : " << stats.total_payload_bytes << "\n";
        std::cout << "Total record B    : " << stats.total_record_bytes << " (payload + 2B header each)\n";
        std::cout << "Manifest          : " << args.manifest_path << "\n";
        std::cout << "Label map (64bit) : " << args.label_map_path << "\n";
        std::cout << "Layout JSON       : " << args.layout_json_path << "\n";
        std::cout << "Graph offsets     : " << gp.offsets_path << "\n";
        std::cout << "Graph neighbors   : " << gp.neighbors_path << "\n";
        std::cout << "Graph labels      : " << gp.labels_path << "\n";
        std::cout << "Graph L1 seeds    : " << gp.l1_seeds_path << "\n";
        std::cout << "Graph meta        : " << gp.meta_path << "\n";
        std::cout << "Done.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
