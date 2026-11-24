#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cctype>
#include <stdint.h>

namespace annfs::common {

/**
 * @brief NPY 文件读取器
 * 支持读取 NumPy .npy 格式的二进制文件
 */
class NPYReader {
private:
    struct NPYHeader {
        std::string dtype;
        std::vector<size_t> shape;
        bool fortran_order;
    };

    // 解析 NPY 头部
    NPYHeader parseHeader(std::ifstream& file)
    {
        NPYHeader header;
        
        // 读取魔数
        char magic[6];
        file.read(magic, 6);
        if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
            throw std::runtime_error("Invalid NPY file format");
        }
        
        // 读取版本
        uint8_t major_version, minor_version;
        file.read(reinterpret_cast<char*>(&major_version), 1);
        file.read(reinterpret_cast<char*>(&minor_version), 1);
        
        // 读取头部长度
        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        
        // 读取头部字典
        std::string header_str(header_len, '\0');
        file.read(&header_str[0], header_len);
        
        // 解析 dtype
        size_t descr_pos = header_str.find("'descr':");
        if (descr_pos != std::string::npos) {
            size_t start = header_str.find("'", descr_pos + 8) + 1;
            size_t end = header_str.find("'", start);
            header.dtype = header_str.substr(start, end - start);
        }
        
        // 解析 shape
        size_t shape_pos = header_str.find("'shape':");
        if (shape_pos != std::string::npos) {
            size_t start = header_str.find("(", shape_pos) + 1;
            size_t end = header_str.find(")", start);
            std::string shape_str = header_str.substr(start, end - start);
            
            std::stringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // 移除前后的空白字符
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                
                // 检查是否为空或只包含非数字字符
                if (!token.empty() && token.find_first_of("0123456789") != std::string::npos) {
                    try {
                        // 只保留数字字符
                        std::string num_only;
                        for (char c : token) {
                            if (std::isdigit(c)) {
                                num_only += c;
                            }
                        }
                        if (!num_only.empty()) {
                            header.shape.push_back(std::stoull(num_only));
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to parse shape token '" << token << "': " << e.what() << std::endl;
                    }
                }
            }
        }
        
        header.fortran_order = header_str.find("'fortran_order': True") != std::string::npos;
        return header;
    }

public:
    // 读取 int32 类型的 npy 文件
    std::vector<std::vector<int32_t>> readInt32Array(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        NPYHeader header = parseHeader(file);
        
        if (header.dtype != "<i4" && header.dtype != ">i4" && header.dtype != "i4") {
            throw std::runtime_error("Unsupported dtype: " + header.dtype);
        }
        
        if (header.shape.size() != 2) {
            throw std::runtime_error("Expected 2D array");
        }
        
        size_t rows = header.shape[0];
        size_t cols = header.shape[1];
        
        std::vector<std::vector<int32_t>> result(rows, std::vector<int32_t>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(result[i].data()), cols * sizeof(int32_t));
        }
        
        return result;
    }

    // 读取 int64 类型的 npy 文件
    std::vector<std::vector<int64_t>> readInt64Array(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        NPYHeader header = parseHeader(file);
        
        if (header.dtype != "<i8" && header.dtype != ">i8" && header.dtype != "i8") {
            throw std::runtime_error("Unsupported dtype: " + header.dtype);
        }
        
        if (header.shape.size() != 2) {
            throw std::runtime_error("Expected 2D array");
        }
        
        size_t rows = header.shape[0];
        size_t cols = header.shape[1];
        
        std::vector<std::vector<int64_t>> result(rows, std::vector<int64_t>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(result[i].data()), cols * sizeof(int64_t));
        }
        
        return result;
    }
    
    // 读取 float32 类型的 npy 文件
    std::vector<std::vector<float>> readFloat32Array(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        NPYHeader header = parseHeader(file);
        
        if (header.dtype != "<f4" && header.dtype != ">f4" && header.dtype != "f4") {
            throw std::runtime_error("Unsupported dtype: " + header.dtype);
        }
        
        if (header.shape.size() != 2) {
            throw std::runtime_error("Expected 2D array");
        }
        
        size_t rows = header.shape[0];
        size_t cols = header.shape[1];
        
        std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(result[i].data()), cols * sizeof(float));
        }
        
        return result;
    }
};

} // namespace annfs::common

