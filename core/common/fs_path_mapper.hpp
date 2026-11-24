// fs_path_mapper.hpp
// -----------------------------------------------------------------------------
// 文件系统路径映射器：将索引映射到文件系统路径
// 
// 支持两种文件系统布局：
//   - Single layer (lab1): 单层结构，路径格式为 lab1/IDX.bin
//   - Multi layer (tree): 二层树结构
//       - 1M:  tree/l<i>/l<j>/f_LOCAL.bin
//       - 10M/100M: tree/l<round>/l<i>/l<j>/f_LOCAL.bin
//
// 用法示例：
//   // Single layer
//   IndexToPathMapper mapper("/mnt/nvme0n1/lab1", 10000000, LayoutType::SINGLE);
//   std::string path = mapper.get_path(123);  // "/mnt/nvme0n1/lab1/123.bin"
//
//   // Multi layer (tree)
//   IndexToPathMapper mapper("/mnt/nvme0n1/tree", 10000000, LayoutType::MULTI, 100);
//   std::string path = mapper.get_path(123);  // "/mnt/nvme0n1/tree/l0/l1/f_23.bin" (1M)
//   // 10M/100M: "/mnt/nvme0n1/tree/l0/l0/l1/f_23.bin" (round=0)
// -----------------------------------------------------------------------------

#pragma once

#include <string>
#include <cstddef>

namespace annfs {
namespace common {

enum class LayoutType {
    SINGLE,  // 单层结构 (lab1)
    MULTI    // 多层树结构 (tree)
};

class IndexToPathMapper {
public:
    // 构造函数
    // @param base_path: 基础路径（如 "/mnt/nvme0n1/lab1" 或 "/mnt/nvme0n1/tree"）
    // @param max_elements: 最大元素数量（文件总数）
    // @param layout: 布局类型（SINGLE 或 MULTI）
    // @param fanout: 树结构的 fanout（仅 MULTI 模式需要，默认 100）
    // @param round_size: 每轮文件数（用于 10M/100M 的 /tree/l{round}/... 结构，默认 1,000,000）
    IndexToPathMapper(const std::string& base_path,
                      size_t max_elements,
                      LayoutType layout,
                      int fanout = 100,
                      size_t file_count_limit = 0,
                      size_t round_size = 1000000);

    // 从路径自动检测布局类型（兼容旧代码）
    // 如果 base_path 包含 "/tree"，则使用 MULTI，否则使用 SINGLE
    IndexToPathMapper(const std::string& base_path,
                      size_t max_elements,
                      int fanout = 100,
                      size_t file_count_limit = 0,
                      size_t round_size = 1000000);

    // 获取索引对应的文件路径
    // @param index: 索引值（0 到 max_elements-1）
    // @return: 文件路径，如果 index 超出范围则返回空字符串
    std::string get_path(size_t index) const;

    // 获取布局类型
    LayoutType get_layout() const { return layout_; }

    // 获取 fanout（仅 MULTI 模式有效）
    int get_fanout() const { return fanout_; }

    // 获取最大元素数量（HNSW 索引大小）
    size_t get_max_elements() const { return max_elements_; }

    // 实际磁盘上的文件数量上限（用于取余映射）
    size_t get_file_count_limit() const { return file_count_limit_; }

    bool is_wrapping_enabled() const { return wrap_ids_; }

private:
    std::string base_path_;
    LayoutType layout_;
    int fanout_;
    size_t max_elements_;
    size_t file_count_limit_;
    bool wrap_ids_;
    size_t round_size_;
    bool round_layer_enabled_;

    // Tree 结构相关参数
    size_t total_leaves_;
    size_t per_leaf_;
    size_t remainder_;

    // 将索引映射到 tree 路径
    std::string index_to_tree_path(size_t index) const;

    // 将索引映射到 single 路径
    std::string index_to_flat_path(size_t index) const;

    // 初始化 tree 结构参数
    void init_tree_params(size_t files_per_round);

    size_t effective_file_count() const;
};

} // namespace common
} // namespace annfs
