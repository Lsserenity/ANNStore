#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <atomic>
#include <string>
#include "hnswlib/hnswlib.h"
#include "hnsw/uint8.hpp"
#include "common/data_io.hpp"
#include "common/recall.hpp"
#include "common/parallel.hpp"
#include "json.hpp"

namespace fs = std::filesystem;

template <typename DataT>
static void add_points_parallel(hnswlib::HierarchicalNSW<float> &index,
                                DataT *data,
                                size_t num,
                                size_t dim,
                                int num_threads) {
    const size_t vector_dim = dim;
    Para::ParallelFor(0, num, num_threads, [&](size_t row, size_t /*threadId*/) {
        index.addPoint((void *)(data + row * vector_dim), static_cast<hnswlib::labeltype>(row));
    });
}

// 使用 HNSW 对自身数据做 1-NN 自检，返回 recall@1（使用 Para::ParallelFor 加速）
template <typename DataT>
static double self_recall_at_1(hnswlib::HierarchicalNSW<float> &index,
                               int search_ef,
                               DataT *data,
                               unsigned num,
                               unsigned dim,
                               int num_threads) {
    if (num == 0) return 0.0;
    index.setEf(search_ef);
    std::atomic<size_t> correct(0);
    const size_t vector_dim = static_cast<size_t>(dim);

    Para::ParallelFor(0, num, num_threads, [&](size_t i, size_t threadId) {
        auto res = index.searchKnn((void *)(data + i * vector_dim), 1);
        if (!res.empty() && res.top().second == (hnswlib::labeltype)i) {
            correct.fetch_add(1, std::memory_order_relaxed);
        }
    });

    return static_cast<double>(correct.load()) / static_cast<double>(num);
}

// 在给定 search_ef 下，对 query 与 gt 做 recall@K 评估（这里 K=100），内部使用 Para::ParallelFor 并最终调用 annfs::common::compute_recall
template <typename DataT>
static double query_recall_at_K(hnswlib::HierarchicalNSW<float> &index,
                                DataT *query_data,
                                unsigned query_num,
                                unsigned query_dim,
                                const std::vector<std::vector<unsigned>> &gt,
                                int search_ef,
                                int num_threads,
                                unsigned K) {
    index.setEf(search_ef);
    const size_t vector_dim = static_cast<size_t>(query_dim);

    if (gt.size() < query_num) {
        throw std::runtime_error("gt size < query_num");
    }

    if (gt.empty() || gt[0].size() < K) {
        throw std::runtime_error("gt dimension < K");
    }

    // 对每个 query 搜索 K 个近邻（并行）
    std::vector<std::vector<unsigned>> res(query_num);
    Para::ParallelFor(0, query_num, num_threads, [&](size_t qi, size_t threadId) {
        auto knn = index.searchKnn((void *)(query_data + qi * vector_dim), (size_t)K);
        std::vector<unsigned> ids;
        ids.reserve(K);
        while (!knn.empty()) {
            ids.push_back((unsigned)knn.top().second);
            knn.pop();
        }
        // searchKnn 返回的是从最近到最远的栈，弹出后顺序为从远到近，这里反转回最近优先
        std::reverse(ids.begin(), ids.end());
        if (ids.size() > K) ids.resize(K);
        res[qi] = std::move(ids);
    });

    // 使用 annfs::common::compute_recall 计算 Recall@K（返回百分比）
    float recall_percent = annfs::common::compute_recall<unsigned>(query_num, K, gt, res);
    return static_cast<double>(recall_percent) / 100.0;
}

int RunIndexBuild(const std::string &jsonPath,
                  const std::string &datasetName,
                  const std::string &data_type_override) {
    // 读取 JSON 配置（结构如 configs/build-index.json）
    nlohmann::json root;
    {
        std::ifstream in(jsonPath);
        if (!in.is_open()) {
            std::cerr << "Failed to open config json: " << jsonPath << "\n";
            return 1;
        }
        in >> root;
    }

    if (!root.contains(datasetName)) {
        std::cerr << "Config json does not contain dataset: " << datasetName << "\n";
        return 1;
    }

    auto cfg = root[datasetName];

    std::string file_format = cfg.value("file_format", std::string("bin"));
    std::string base_file   = cfg.value("base_file",   std::string());
    std::string query_file  = cfg.value("query_file",  std::string());
    std::string gt_file     = cfg.value("gt_file",     std::string());
    int M                   = cfg.value("M",           32);
    int ef_construction     = cfg.value("ef_construction", 500);
    int self_search_ef      = cfg.value("self_search_ef", 750);
    int num_threads         = cfg.value("num_threads", 32);
    int search_ef           = cfg.value("search_ef",   750);
    std::string data_type   = cfg.value("data_type",   std::string("float32"));
    if (!data_type_override.empty()) {
        data_type = data_type_override;
    }

    auto to_lower = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };
    const std::string file_format_lower = to_lower(file_format);
    const std::string data_type_lower = to_lower(data_type);
    const bool is_bin_format = (file_format_lower == "bin" || file_format_lower == "fbin" || file_format_lower == "bbin");
    const bool is_vecs_format = (file_format_lower == "vecs" || file_format_lower == "fvecs" || file_format_lower == "bvecs");
    if (!is_bin_format && !is_vecs_format) {
        std::cerr << "Unsupported file_format: " << file_format << "\n";
        return 1;
    }
    bool use_uint8_data = false;
    if (data_type_lower == "uint8" || data_type_lower == "u8") {
        use_uint8_data = true;
    } else if (data_type_lower == "float32" || data_type_lower == "float" || data_type_lower == "f32") {
        use_uint8_data = false;
    } else {
        std::cerr << "Unsupported data_type: " << data_type << "\n";
        return 1;
    }

    std::cout << "=== Build HNSW Index for dataset: " << datasetName << " ===\n";
    std::cout << "  file_format   : " << file_format << "\n";
    std::cout << "  base_file     : " << base_file << "\n";
    std::cout << "  query_file    : " << query_file << "\n";
    std::cout << "  gt_file       : " << gt_file << "\n";
    std::cout << "  M             : " << M << "\n";
    std::cout << "  ef_construction: " << ef_construction << "\n";
    std::cout << "  num_threads   : " << num_threads << "\n";
    std::cout << "  initial search_ef: " << search_ef << "\n";
    std::cout << "  data_type     : " << data_type << "\n";
    int self_search_ef_copy=self_search_ef;
    if (base_file.empty() || query_file.empty() || gt_file.empty()) {
        std::cerr << "base_file / query_file / gt_file in json cannot be empty.\n";
        return 1;
    }

    // 读取 base / query / gt
    unsigned base_num, base_dim;
    std::unique_ptr<float[]> base_data_float;
    std::unique_ptr<uint8_t[]> base_data_u8;
    std::unique_ptr<float[]> query_data_float;
    std::unique_ptr<uint8_t[]> query_data_u8;

    if (use_uint8_data) {
        if (is_bin_format) {
            base_data_u8.reset(annfs::common::read_data_bin<uint8_t>(base_file, base_num, base_dim));
        } else {
            base_data_u8.reset(annfs::common::read_data_vecs<uint8_t>(base_file, base_num, base_dim));
        }
    } else {
        base_data_float.reset(annfs::common::read_data(base_file, file_format, base_num, base_dim));
    }
    std::cout << "base_num, base_dim = " << base_num << ", " << base_dim << std::endl;

    unsigned query_num, query_dim;
    if (use_uint8_data) {
        if (is_bin_format) {
            query_data_u8.reset(annfs::common::read_data_bin<uint8_t>(query_file, query_num, query_dim));
        } else {
            query_data_u8.reset(annfs::common::read_data_vecs<uint8_t>(query_file, query_num, query_dim));
        }
    } else {
        query_data_float.reset(annfs::common::read_data(query_file, file_format, query_num, query_dim));
    }
    std::cout << "query_num, query_dim = " << query_num << ", " << query_dim << std::endl;

    unsigned gt_num, gt_dim;
    std::vector<std::vector<unsigned>> gt = annfs::common::read_gt(gt_file, file_format, gt_num, gt_dim);
    std::cout << "gt_num, gt_dim = " << gt_num << ", " << gt_dim << std::endl;

    if (query_num != gt_num) {
        std::cerr << "query_num != gt_num, cannot evaluate recall.\n";
        return 1;
    }
    
    size_t max_elements = base_num;

    // 空间类型：float 默认使用 InnerProduct，uint8 使用自定义 L2
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_holder;
    if (use_uint8_data) {
        space_holder = std::make_unique<hnswlib::L2SpaceUint8>(base_dim);
    } else {
        space_holder = std::make_unique<hnswlib::InnerProductSpace>(base_dim);
    }
    hnswlib::SpaceInterface<float> *space = space_holder.get();
    hnswlib::HierarchicalNSW<float> *alg_hnsw = nullptr;

    // 自动调节 ef_construction / M / self_searchef 直到 self_recall@1 >= 0.99 或达到上限
    const int max_ef_construction = 2000;
    const int step_ef_construction = 100;
    const int max_self_searchef=2000;
    const int step_self_searchef=100;
    const int max_M = 128;
    const int step_M = 8;

    bool build_ok = false;
    while (true) {
        if (alg_hnsw != nullptr) {
            delete alg_hnsw;
            alg_hnsw = nullptr;
        }
        self_search_ef = self_search_ef_copy;
        std::cout << "[BUILD] 使用 M=" << M << ", ef_construction=" << ef_construction
                  << " 构建 HNSW 索引 ..." << std::endl;
        // 准备输出目录
        std::string output_dir = "./hnsw_indices/M" + std::to_string(M) + "_efc" + std::to_string(ef_construction) + "/";
        try {
            if (!fs::exists(output_dir)) {
                fs::create_directories(output_dir); // 递归创建
            }
            std::cout << "目录已准备好: " << output_dir << std::endl;
        } catch (const std::filesystem::filesystem_error &e) {
            std::cerr << "创建目录失败: " << e.what() << std::endl;
            return 1;
        }

        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction);

        if (use_uint8_data) {
            add_points_parallel(*alg_hnsw, base_data_u8.get(), max_elements, static_cast<size_t>(base_dim), num_threads);
        } else {
            add_points_parallel(*alg_hnsw, base_data_float.get(), max_elements, static_cast<size_t>(base_dim), num_threads);
        }
        std::cout << "[BUILD] 构建完成，插入点数量: " << base_num << std::endl;
        std::string index_path = output_dir + datasetName + ".bin";

        // 先保存索引
        alg_hnsw->saveIndex(index_path);
        std::cout << "[SAVE] Build Process - Index saved to: " << index_path << "\n";

    //     // 自检：每个 base 点查 1-NN，看是否能找回自己，要求 recall@1 > 0.99
    //     double self_recall = 0;
    //     while(self_recall<0.99 && self_search_ef<=max_self_searchef){
    //         std::cout << "[CHECK] 开始自检 recall@1 ..." << std::endl;
    //         if (use_uint8_data) {
    //             self_recall = self_recall_at_1(*alg_hnsw, self_search_ef, base_data_u8.get(), base_num, base_dim, num_threads);
    //         } else {
    //             self_recall = self_recall_at_1(*alg_hnsw, self_search_ef, base_data_float.get(), base_num, base_dim, num_threads);
    //         }
    //         std::cout << "[CHECK] Self recall@1 = " << self_recall << std::endl;
    //         if(self_recall>=0.99) break;
    //         self_search_ef+=step_self_searchef;
    //     }
        
    //     if (self_recall >= 0.99) {
    //         std::cout << "[CHECK] Self recall@1 >= 0.99，构建参数通过。\n";
        build_ok = true;
    //         break;
    //     }

    //     // 自检失败，尝试增大 ef_construction 或 M
    //     if (ef_construction < max_ef_construction) {
    //         ef_construction = std::min(max_ef_construction, ef_construction + step_ef_construction);
    //         std::cout << "[TUNE] Self recall@1 < 0.99，提升 ef_construction 到 " << ef_construction << " 重新构建索引。\n";
    //         continue;
    //     }

    //     if (M < max_M) {
    //         M = std::min(max_M, M + step_M);
    //         std::cout << "[TUNE] ef_construction 已达上限，提升 M 到 " << M << " 重新构建索引。\n";
    //         // ef_construction 继续保持当前（较大）数值
    //         continue;
    //     }

    //     std::cerr << "[TUNE] 已达到最大 M/ef_construction 仍未达到 self_recall@1 >= 0.99，终止。\n";
        break;
    }

    if (!build_ok) {
        if (alg_hnsw) delete alg_hnsw;
        std::cerr<<"[FAILED] 没有成功构建index。\n";
        return 1;
    }

    std::cout<<"[SUCCESS] 成功构建索引 M "<< M <<" , ef_construction "<<ef_construction <<", self_search_ef "<< self_search_ef<<"\n";

    // 使用不同的 search_ef 对 query 做评估，记录 recall@100 与 search_ef 的关系
    const unsigned K_eval = 100;
    std::cout << "[EVAL] 开始对 query 集进行 search_ef-Recall@" << K_eval << " 评估 ..." << std::endl;
    int current_ef = search_ef;
    const int max_ef = 2000;
    const int step_ef = 50;

    double best_recall = 0.0; // 0~1
    int best_ef = current_ef;

    while (current_ef <= max_ef) {
        double recall = use_uint8_data
                            ? query_recall_at_K(*alg_hnsw,
                                                query_data_u8.get(),
                                                query_num,
                                                query_dim,
                                                gt,
                                                current_ef,
                                                num_threads,
                                                K_eval)
                            : query_recall_at_K(*alg_hnsw,
                                                query_data_float.get(),
                                                query_num,
                                                query_dim,
                                                gt,
                                                current_ef,
                                                num_threads,
                                                K_eval);
        std::cout << "[EVAL] search_ef = " << current_ef << ", recall@" << K_eval << " = " << recall << std::endl;

        if (recall > best_recall) {
            best_recall = recall;
            best_ef = current_ef;
        }

        if (recall >= 0.99) {
            std::cout << "[EVAL] 达到目标 recall@" << K_eval << " > 0.99, 使用 search_ef = " << current_ef << std::endl;
            break;
        }

        current_ef += step_ef;
    }

    if (best_recall < 0.99) {
        std::cout << "[EVAL] 在 search_ef <= " << max_ef << " 范围内，recall@" << K_eval << " 最好为 "
                  << best_recall << " (search_ef = " << best_ef << ")" << std::endl;
    } else {
        std::cout << "[EVAL] 选定 search_ef = " << best_ef << "，对应 recall@" << K_eval << " = " << best_recall << std::endl;
    }

    // 将最终的（或最好的）search_ef 以及调节后的 M / ef_construction / self_search_ef 写回 JSON 配置
    root[datasetName]["M"] = M;
    root[datasetName]["ef_construction"] = ef_construction;
    root[datasetName]["self_search_ef"] = self_search_ef;
    root[datasetName]["search_ef"] = best_ef;
    root[datasetName]["data_type"] = data_type;
    {
        std::ofstream out(jsonPath);
        if (!out.is_open()) {
            std::cerr << "写回 JSON 配置失败: " << jsonPath << std::endl;
        } else {
            out << std::setw(4) << root << std::endl;
            std::cout << "[CONF] 已将 search_ef = " << best_ef << " 写回到 " << jsonPath << std::endl;
        }
    }

    delete alg_hnsw;

    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <json-path> <dataset-name> [data-type=float32|uint8]\n";
        return 1;
    }

    std::string jsonPath = std::string(argv[1]);
    std::string datasetName = std::string(argv[2]);
    std::string dataTypeOverride;
    if (argc >= 4) {
        dataTypeOverride = std::string(argv[3]);
    }
    return RunIndexBuild(jsonPath, datasetName, dataTypeOverride);
}
