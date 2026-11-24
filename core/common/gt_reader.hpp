#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "npy_reader.hpp"

namespace annfs::common {

/**
 * @brief Ground Truth 读取器
 * 用于从分块的 NPY 文件中读取 ground truth 数据
 */
class GroundTruthReader {
private:
    std::string gt_dir;
    size_t batch_size;
    size_t total_queries;
    size_t k;
    
public:
    GroundTruthReader(const std::string& directory, size_t batch_sz, size_t total_q, size_t topk) 
        : gt_dir(directory), batch_size(batch_sz), total_queries(total_q), k(topk) {}
    
    // 获取指定 query id 的 top-k 结果 ID
    std::vector<int64_t> getTopKResults(size_t query_id)
    {
        if (query_id >= total_queries) {
            throw std::runtime_error("Query ID out of range");
        }
        
        size_t batch_idx = query_id / batch_size;
        size_t local_idx = query_id % batch_size;
        
        std::ostringstream oss;
        oss << gt_dir << "/indices_" << std::setfill('0') << std::setw(4) << batch_idx << ".npy";
        std::string filename = oss.str();
        
        try {
            NPYReader reader;
            auto indices = reader.readInt64Array(filename);
            
            if (local_idx >= indices.size()) {
                throw std::runtime_error("Local index out of range in batch file");
            }
            
            return indices[local_idx];
        } catch (const std::exception& e) {
            std::cerr << "Error reading " << filename << ": " << e.what() << std::endl;
            return std::vector<int64_t>();
        }
    }
    
    // 获取指定 query id 的距离
    std::vector<float> getTopKDistances(size_t query_id)
    {
        if (query_id >= total_queries) {
            throw std::runtime_error("Query ID out of range");
        }
        
        size_t batch_idx = query_id / batch_size;
        size_t local_idx = query_id % batch_size;
        
        std::ostringstream oss;
        oss << gt_dir << "/distances_" << std::setfill('0') << std::setw(4) << batch_idx << ".npy";
        std::string filename = oss.str();
        
        try {
            NPYReader reader;
            auto distances = reader.readFloat32Array(filename);
            
            if (local_idx >= distances.size()) {
                throw std::runtime_error("Local index out of range in batch file");
            }
            
            return distances[local_idx];
        } catch (const std::exception& e) {
            std::cerr << "Error reading " << filename << ": " << e.what() << std::endl;
            return std::vector<float>();
        }
    }
};

} // namespace annfs::common

