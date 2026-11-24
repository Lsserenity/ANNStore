// fs_path_mapper.cpp
// -----------------------------------------------------------------------------
// 文件系统路径映射器的实现
// -----------------------------------------------------------------------------

#include "common/fs_path_mapper.hpp"
#include <algorithm>
#include <string>

namespace annfs {
namespace common {

IndexToPathMapper::IndexToPathMapper(const std::string& base_path,
                                     size_t max_elements,
                                     LayoutType layout,
                                     int fanout,
                                     size_t file_count_limit,
                                     size_t round_size)
    : base_path_(base_path),
      layout_(layout),
      fanout_(fanout),
      max_elements_(max_elements),
      file_count_limit_(file_count_limit),
      wrap_ids_(file_count_limit > 0 && file_count_limit < max_elements),
      round_size_(round_size),
      round_layer_enabled_(false),
      total_leaves_(0),
      per_leaf_(0),
      remainder_(0) {
    if (layout_ == LayoutType::MULTI) {
        size_t files = effective_file_count();
        bool base_has_round = base_path_.find("/tree/l") != std::string::npos;
        round_layer_enabled_ = (!base_has_round && round_size_ > 0 && files > round_size_);
        size_t files_per_round = round_layer_enabled_ ? round_size_ : files;
        init_tree_params(files_per_round);
    }
}

IndexToPathMapper::IndexToPathMapper(const std::string& base_path,
                                     size_t max_elements,
                                     int fanout,
                                     size_t file_count_limit,
                                     size_t round_size)
    : base_path_(base_path),
      fanout_(fanout),
      max_elements_(max_elements),
      file_count_limit_(file_count_limit),
      wrap_ids_(file_count_limit > 0 && file_count_limit < max_elements),
      round_size_(round_size),
      round_layer_enabled_(false),
      total_leaves_(0),
      per_leaf_(0),
      remainder_(0) {
    // 自动检测布局类型：如果路径包含 "/tree"，则使用 MULTI，否则使用 SINGLE
    if (base_path.find("/tree") != std::string::npos) {
        layout_ = LayoutType::MULTI;
        size_t files = effective_file_count();
        bool base_has_round = base_path_.find("/tree/l") != std::string::npos;
        round_layer_enabled_ = (!base_has_round && round_size_ > 0 && files > round_size_);
        size_t files_per_round = round_layer_enabled_ ? round_size_ : files;
        init_tree_params(files_per_round);
    } else {
        layout_ = LayoutType::SINGLE;
    }
}

void IndexToPathMapper::init_tree_params(size_t files_per_round) {
    total_leaves_ = static_cast<size_t>(fanout_) * static_cast<size_t>(fanout_);
    per_leaf_ = files_per_round / total_leaves_;
    remainder_ = files_per_round % total_leaves_;
}

std::string IndexToPathMapper::index_to_tree_path(size_t index) const {
    size_t round_index = 0;
    size_t local_index = index;
    if (round_layer_enabled_ && round_size_ > 0) {
        round_index = index / round_size_;
        local_index = index % round_size_;
    }

    size_t leaf_index = 0;

    size_t files_in_first_leaves = remainder_ * (per_leaf_ + 1);

    if (local_index < files_in_first_leaves) {
        leaf_index = local_index / (per_leaf_ + 1);
        local_index = local_index % (per_leaf_ + 1);
    } else {
        size_t remaining_index = local_index - files_in_first_leaves;
        leaf_index = remainder_ + remaining_index / per_leaf_;
        local_index = remaining_index % per_leaf_;
    }

    int i = static_cast<int>(leaf_index / static_cast<size_t>(fanout_));
    int j = static_cast<int>(leaf_index % static_cast<size_t>(fanout_));

    std::string prefix = base_path_;
    if (round_layer_enabled_) {
        prefix += "/l" + std::to_string(round_index);
    }

    return prefix + "/l" + std::to_string(i) +
           "/l" + std::to_string(j) +
           "/f_" + std::to_string(local_index) + ".bin";
}

std::string IndexToPathMapper::index_to_flat_path(size_t index) const {
    return base_path_ + "/" + std::to_string(index) + ".bin";
}

size_t IndexToPathMapper::effective_file_count() const {
    if (file_count_limit_ > 0) {
        return file_count_limit_;
    }
    return max_elements_;
}

std::string IndexToPathMapper::get_path(size_t index) const {
    size_t mapped_index = index;
    if (wrap_ids_) {
        size_t limit = effective_file_count();
        if (limit == 0) {
            return {};
        }
        mapped_index = index % limit;
    } else if (index >= max_elements_) {
        return {};
    }
    if (layout_ == LayoutType::MULTI) {
        return index_to_tree_path(mapped_index);
    }
    return index_to_flat_path(mapped_index);
}

} // namespace common
} // namespace annfs
