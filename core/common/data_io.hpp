#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace annfs::common {

/**
 * @brief 读取 fvecs 文件
 * @param path - 文件路径
 * @param max_vectors - 最大读取向量数量，0 表示读取所有
 * @return 二维向量数组
 */
template <typename T>
std::vector<std::vector<T>> read_fvecs(const std::string& path, size_t max_vectors = 0)
{
    std::vector<std::vector<T>> vecs;
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return vecs;
    }

    size_t i = 0;
    while (true) {
        int dim = 0;
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!fin.good()) break;
        if (dim <= 0) {
            std::cerr << "Invalid dim read: " << dim << " at index " << i << std::endl;
            break;
        }
        std::vector<float> arr(dim);
        fin.read(reinterpret_cast<char*>(arr.data()), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!fin.good()) break;

        vecs.push_back(std::move(arr));
        ++i;
        if (max_vectors > 0 && vecs.size() >= max_vectors) break;
    }

    fin.close();
    return vecs;
}

/**
 * @brief 按行归一化
 * @param X - 待归一化的二维向量
 * @param eps - 防止除零的小常数
 */
template <typename T>
void normalize_rows(std::vector<std::vector<T>>& X, float eps = 1e-10f)
{
    for (auto& row : X) {
        double norm2 = 0.0;
        for (float v : row) norm2 += double(v) * double(v);
        float norm = static_cast<float>(std::sqrt(norm2));
        if (norm < eps) norm = eps;
        for (float& v : row) v /= norm;
    }
}

/**
 * @brief 归一化数据（原地修改）
 * @param data - 数据指针
 * @param N - 向量数量
 * @param Dim - 向量维度
 */
inline void normalize_data(float *data, unsigned N, unsigned Dim)
{
    for (size_t i = 0; i < (size_t)N; i++)
    {
        float *point = data + (uint64_t)i * Dim;
        double norm = 0.0;
        for (size_t d = 0; d < (size_t)Dim; d++)
            norm += ((double)point[d] * point[d]);
        norm = sqrt(norm);
        if (norm == 0.0)
            throw std::runtime_error("error@normalize_data: norm can't be 0.0");
        for (size_t d = 0; d < (size_t)Dim; d++)
            point[d] /= norm;
    }
}

/**
 * @brief 获取文件大小（FILE* 版本）
 */
inline long int fsize(FILE *fp)
{
    long int prev = ftell(fp);
    fseek(fp, 0L, SEEK_END);
    long int sz = ftell(fp);
    fseek(fp, prev, SEEK_SET);
    return sz;
}

/**
 * @brief 读取 .vecs 格式数据（如 .fvecs, .bvecs）
 */
template <typename T>
inline T *read_data_vecs(const std::string &path_file, unsigned &N, unsigned &Dim)
{
    FILE *F = nullptr;
    F = fopen(path_file.c_str(), "rb");
    if (F == NULL)
        throw std::runtime_error("error@read_data_vecs: Dataset not found");

    [[maybe_unused]] size_t xxx = fread(&Dim, sizeof(unsigned), 1, F);
    long int sizebytes = fsize(F);
    N = sizebytes / (sizeof(unsigned) + sizeof(T) * Dim);
    rewind(F);
    T *data = new T[(uint64_t)N * Dim];
    for (size_t i = 0; i < (size_t)N; i++)
    {
        xxx = fread(&Dim, sizeof(unsigned), 1, F);
        xxx = fread(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
    }
    fclose(F);
    return data;
}

/**
 * @brief 读取 .bin 格式数据（如 .fbin, .bbin）
 */
template <typename T>
inline T *read_data_bin(const std::string &path_file, unsigned &N, unsigned &Dim)
{
    FILE *F = nullptr;
    F = fopen(path_file.c_str(), "rb");
    if (F == NULL)
        throw std::runtime_error("error@read_data_bin: Dataset not found");

    [[maybe_unused]] size_t xxx = fread(&N, sizeof(unsigned), 1, F);
    xxx = fread(&Dim, sizeof(unsigned), 1, F);
    T *data = new T[(uint64_t)N * Dim];
    xxx = fread(data, sizeof(T), (uint64_t)N * Dim, F);
    fclose(F);
    return data;
}

/**
 * @brief 读取不同格式的数据（统一接口）
 * @param path_file - 文件路径
 * @param format - 格式（"bin" 或 "vecs"）
 * @param N - 输出向量数量
 * @param Dim - 输出向量维度
 * @return 数据指针（需要调用者 delete[]）
 */
inline float *read_data(const std::string &path_file, const std::string &format, unsigned &N, unsigned &Dim)
{
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    
    if (fmt == "bin" || fmt == "fbin" || fmt == "bbin")
        return read_data_bin<float>(path_file, N, Dim);
    else if (fmt == "vecs" || fmt == "fvecs" || fmt == "bvecs")
        return read_data_vecs<float>(path_file, N, Dim);
    else
        throw std::runtime_error("error@read_data: unsupported file format: " + format);
}

/**
 * @brief 读取 .ivecs 格式 GT 数据
 */
template <typename T>
inline std::vector<std::vector<T>> read_gt_ivecs(const std::string &path_file, unsigned &N, unsigned &Dim)
{
    FILE *F = nullptr;
    F = fopen(path_file.c_str(), "rb");
    if (F == NULL)
        throw std::runtime_error("error@read_gt_ivecs: Dataset not found");

    [[maybe_unused]] size_t xxx = fread(&Dim, sizeof(unsigned), 1, F);
    long int sizebytes = fsize(F);
    N = sizebytes / (sizeof(unsigned) + sizeof(T) * Dim);
    rewind(F);
    std::vector<std::vector<T>> gt;
    for (size_t i = 0; i < (size_t)N; i++)
    {
        xxx = fread(&Dim, sizeof(unsigned), 1, F);
        T *nn = new T[Dim];
        xxx = fread(nn, sizeof(T), Dim, F);
        gt.push_back(std::vector<T>(nn, nn + Dim));
        delete[] nn;
    }
    fclose(F);
    return gt;
}

/**
 * @brief 读取 .ibin 格式 GT 数据
 */
template <typename T>
inline std::vector<std::vector<T>> read_gt_ibin(const std::string &path_file, unsigned &N, unsigned &Dim)
{
    FILE *F = nullptr;
    F = fopen(path_file.c_str(), "rb");
    if (F == NULL)
        throw std::runtime_error("error@read_gt_ibin: Dataset not found");

    [[maybe_unused]] size_t xxx = fread(&N, sizeof(unsigned), 1, F);
    xxx = fread(&Dim, sizeof(unsigned), 1, F);
    std::vector<std::vector<T>> gt;
    for (size_t i = 0; i < (size_t)N; i++)
    {
        T *nn = new T[Dim];
        xxx = fread(nn, sizeof(T), Dim, F);
        gt.push_back(std::vector<T>(nn, nn + Dim));
        delete[] nn;
    }
    fclose(F);
    return gt;
}

/**
 * @brief 读取不同格式的 GT 数据（统一接口）
 * @param path_file - 文件路径
 * @param format - 格式（"bin"/"ibin" 或 "vecs"/"ivecs"）
 * @param N - 输出查询数量
 * @param Dim - 输出每个查询的 GT 数量
 * @return GT 数据
 */
inline std::vector<std::vector<unsigned>> read_gt(const std::string &path_file, const std::string &format, unsigned &N, unsigned &Dim)
{
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    
    if (fmt == "bin" || fmt == "ibin")
        return read_gt_ibin<unsigned>(path_file, N, Dim);
    else if (fmt == "vecs" || fmt == "ivecs")
        return read_gt_ivecs<unsigned>(path_file, N, Dim);
    else
        throw std::runtime_error("error@read_gt: unsupported file format: " + format);
}

/**
 * @brief 写入 .vecs 格式数据
 */
template <typename T>
inline void write_data_vecs(const std::string &path_file, T *data, unsigned N, unsigned Dim)
{
    FILE *F = nullptr;
    F = fopen(path_file.c_str(), "wb");
    if (F == NULL)
        throw std::runtime_error("error@write_data_vecs: File not found");

    [[maybe_unused]] size_t xxx;
    for (size_t i = 0; i < (size_t)N; i++)
    {
        xxx = fwrite(&Dim, sizeof(unsigned), 1, F);
        xxx = fwrite(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
    }
    fclose(F);
}

/**
 * @brief 写入 .bin 格式数据
 */
template <typename T>
inline void write_data_bin(const std::string &path_file, T *data, unsigned N, unsigned Dim)
{
    FILE *F = nullptr;
    F = fopen(path_file.c_str(), "wb");
    if (F == NULL)
        throw std::runtime_error("error@write_data_bin: File not found");

    [[maybe_unused]] size_t xxx;
    xxx = fwrite(&N, sizeof(unsigned), 1, F);
    xxx = fwrite(&Dim, sizeof(unsigned), 1, F);
    for (size_t i = 0; i < (size_t)N; i++)
    {
        xxx = fwrite(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
    }
    fclose(F);
}

/**
 * @brief 写入不同格式的数据（统一接口）
 */
inline void write_data(const std::string &path_file, const std::string &format, const float *data, unsigned N, unsigned Dim)
{
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    
    if (fmt == "bin" || fmt == "fbin" || fmt == "bbin")
        write_data_bin<const float>(path_file, data, N, Dim);
    else if (fmt == "vecs" || fmt == "fvecs" || fmt == "bvecs")
        write_data_vecs<const float>(path_file, data, N, Dim);
    else
        throw std::runtime_error("error@write_data: unsupported file format: " + format);
}

} // namespace annfs::common
