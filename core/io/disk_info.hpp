#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <cmath>
#include <system_error>
#include <stdexcept>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <limits>
#include <spdlog/spdlog.h>

#include "../config/global_config.hpp"

inline void get_block_size(const std::string &device)
{
    auto& cfg = GlobalConfig::instance();

    int fd = open(device.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::system_error(errno, std::generic_category(), "open device failed");

    uint32_t block_size = 0;
    if (ioctl(fd, BLKBSZGET, &block_size) == -1) {
        close(fd);
        throw std::system_error(errno, std::generic_category(), "ioctl BLKBSZGET failed");
    }

    cfg.LOGIC_BLOCK_SIZE = block_size;
    cfg.BLOCK_OFF_MASK = block_size - 1;
    cfg.OFF_BITS_LEN = static_cast<int>(std::log2(static_cast<double>(block_size)));
    cfg.BLOCK_ID_MASK = ~cfg.BLOCK_OFF_MASK;
    cfg.PACKED64_CROSS_SHIFT = 64 - cfg.PACKED64_CROSS_BITS;
    cfg.PACKED64_OFF_BITS_LEN = cfg.PACKED64_CROSS_SHIFT;
    if (cfg.PACKED64_OFF_BITS_LEN >= 64) {
        cfg.PACKED64_OFF_MASK = std::numeric_limits<std::uint64_t>::max();
    } else {
        cfg.PACKED64_OFF_MASK =
            (1ULL << static_cast<unsigned>(cfg.PACKED64_OFF_BITS_LEN)) - 1ULL;
    }
    

    uint64_t disk_size = 0;
    if (ioctl(fd, BLKGETSIZE64, &disk_size) == -1) {
        close(fd);
        throw std::system_error(errno, std::generic_category(), "ioctl BLKGETSIZE64 failed");
    }
    close(fd);

    disk_size &= cfg.BLOCK_ID_MASK;
    cfg.DISK_SIZE = disk_size;

    if (cfg.logger)
        cfg.logger->debug("block={}, disk={}", block_size, disk_size);

    return;
}

namespace annfs::io {

struct DecodedPacked64 {
    std::uint32_t crossed_blocks = 0;
    std::uint64_t start_byte = 0;
    std::uint64_t block_base = 0;
    std::uint32_t blocks_to_read = 1;
    std::uint64_t in_block_offset = 0;
};

inline DecodedPacked64 DecodePacked64(std::uint64_t packed64,
                                      std::uint32_t block_size)
{
    const auto& cfg = GlobalConfig::instance();
    DecodedPacked64 d;
    const std::uint64_t cross_mask =
        (cfg.PACKED64_CROSS_BITS >= 64)
            ? std::numeric_limits<std::uint64_t>::max()
            : ((1ULL << static_cast<unsigned>(cfg.PACKED64_CROSS_BITS)) - 1ULL);
    d.crossed_blocks = static_cast<std::uint32_t>(
        (packed64 >> static_cast<unsigned>(cfg.PACKED64_CROSS_SHIFT)) & cross_mask);
    d.start_byte = packed64 & cfg.PACKED64_OFF_MASK;
    d.block_base = d.start_byte & ~(static_cast<std::uint64_t>(block_size) - 1ULL);
    d.blocks_to_read = static_cast<std::uint32_t>(d.crossed_blocks) + 1u;
    d.in_block_offset = d.start_byte - d.block_base;
    return d;
}

inline std::unordered_map<std::uint64_t, std::uint64_t> LoadLabelMapFile(
    const std::string& path)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open label_map: " + path);
    }
    std::unordered_map<std::uint64_t, std::uint64_t> mp;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::uint64_t label = 0;
        std::string hex_tok;
        std::uint64_t packed = 0;
        if (!(iss >> label >> hex_tok >> packed)) {
            continue;
        }
        mp[label] = packed;
    }
    return mp;
}

} // namespace annfs::io
