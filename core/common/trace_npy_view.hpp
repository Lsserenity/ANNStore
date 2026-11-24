#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

namespace annfs::io {

struct TraceView {
    int fd = -1;
    const std::uint8_t* map = nullptr;
    std::size_t map_len = 0;
    const std::int32_t* data = nullptr;
    std::size_t count = 0;

    ~TraceView() {
        if (map != nullptr && map_len > 0) {
            ::munmap(const_cast<std::uint8_t*>(map), map_len);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    TraceView() = default;
    TraceView(const TraceView&) = delete;
    TraceView& operator=(const TraceView&) = delete;

    TraceView(TraceView&& other) noexcept {
        fd = other.fd;
        map = other.map;
        map_len = other.map_len;
        data = other.data;
        count = other.count;
        other.fd = -1;
        other.map = nullptr;
        other.map_len = 0;
        other.data = nullptr;
        other.count = 0;
    }

    TraceView& operator=(TraceView&& other) noexcept {
        if (this == &other) return *this;
        if (map != nullptr && map_len > 0) {
            ::munmap(const_cast<std::uint8_t*>(map), map_len);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        fd = other.fd;
        map = other.map;
        map_len = other.map_len;
        data = other.data;
        count = other.count;
        other.fd = -1;
        other.map = nullptr;
        other.map_len = 0;
        other.data = nullptr;
        other.count = 0;
        return *this;
    }
};

inline std::string extract_header_value(const std::string& header,
                                        const std::string& key,
                                        char quote) {
    const std::size_t pos = header.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("NPY header missing key: " + key);
    }
    const std::size_t s = header.find(quote, pos + key.size());
    if (s == std::string::npos) throw std::runtime_error("NPY header parse error");
    const std::size_t e = header.find(quote, s + 1);
    if (e == std::string::npos) throw std::runtime_error("NPY header parse error");
    return header.substr(s + 1, e - s - 1);
}

inline std::vector<std::size_t> parse_shape(const std::string& header) {
    std::size_t pos = header.find("'shape':");
    if (pos == std::string::npos) pos = header.find("\"shape\":");
    if (pos == std::string::npos) {
        throw std::runtime_error("NPY header missing shape");
    }

    const std::size_t l = header.find('(', pos);
    const std::size_t r = header.find(')', l == std::string::npos ? pos : l);
    if (l == std::string::npos || r == std::string::npos || r <= l) {
        throw std::runtime_error("NPY header invalid shape");
    }

    std::vector<std::size_t> dims;
    std::size_t i = l + 1;
    while (i < r) {
        while (i < r && (header[i] == ' ' || header[i] == ',')) ++i;
        if (i >= r) break;
        if (header[i] < '0' || header[i] > '9') {
            while (i < r && header[i] != ',') ++i;
            continue;
        }
        std::size_t v = 0;
        while (i < r && header[i] >= '0' && header[i] <= '9') {
            v = v * 10 + static_cast<std::size_t>(header[i] - '0');
            ++i;
        }
        dims.push_back(v);
        while (i < r && header[i] != ',') ++i;
    }

    if (dims.empty()) {
        throw std::runtime_error("NPY header shape empty");
    }
    return dims;
}

inline TraceView load_trace_npy_int32(const std::string& path) {
    TraceView tv;

    tv.fd = ::open(path.c_str(), O_RDONLY);
    if (tv.fd < 0) {
        throw std::runtime_error("cannot open trace file: " + path);
    }

    struct stat st {};
    if (::fstat(tv.fd, &st) != 0) {
        throw std::runtime_error("fstat failed for trace file");
    }
    if (st.st_size <= 0) {
        throw std::runtime_error("trace file is empty");
    }

    tv.map_len = static_cast<std::size_t>(st.st_size);
    tv.map = static_cast<const std::uint8_t*>(
        ::mmap(nullptr, tv.map_len, PROT_READ, MAP_PRIVATE, tv.fd, 0));
    if (tv.map == MAP_FAILED) {
        tv.map = nullptr;
        throw std::runtime_error("mmap failed for trace file");
    }

    if (tv.map_len < 10) {
        throw std::runtime_error("trace file too small for npy");
    }

    const std::uint8_t* p = tv.map;
    if (std::memcmp(p, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic");
    }

    const std::uint8_t major = p[6];
    std::size_t header_len = 0;
    std::size_t header_pos = 0;
    if (major == 1) {
        if (tv.map_len < 10) throw std::runtime_error("invalid npy v1 header");
        std::uint16_t h16 = 0;
        std::memcpy(&h16, p + 8, sizeof(h16));
        header_len = static_cast<std::size_t>(h16);
        header_pos = 10;
    } else {
        if (tv.map_len < 12) throw std::runtime_error("invalid npy v2/v3 header");
        std::uint32_t h32 = 0;
        std::memcpy(&h32, p + 8, sizeof(h32));
        header_len = static_cast<std::size_t>(h32);
        header_pos = 12;
    }

    if (header_pos + header_len > tv.map_len) {
        throw std::runtime_error("npy header out of bounds");
    }

    const std::string header(reinterpret_cast<const char*>(p + header_pos), header_len);

    std::string descr;
    try {
        descr = extract_header_value(header, "'descr':", '\'');
    } catch (...) {
        descr = extract_header_value(header, "\"descr\":", '"');
    }

    const bool fortran_order =
        (header.find("'fortran_order': True") != std::string::npos) ||
        (header.find("\"fortran_order\": true") != std::string::npos);
    if (fortran_order) {
        throw std::runtime_error("fortran_order npy not supported");
    }

    if (!(descr == "<i4" || descr == "|i4" || descr == "i4")) {
        throw std::runtime_error("unsupported npy dtype: " + descr + " (need int32)");
    }

    const std::vector<std::size_t> shape = parse_shape(header);
    std::size_t elem_count = 1;
    for (std::size_t d : shape) elem_count *= d;

    const std::size_t data_pos = header_pos + header_len;
    const std::size_t need_bytes = elem_count * sizeof(std::int32_t);
    if (data_pos + need_bytes > tv.map_len) {
        throw std::runtime_error("npy data out of bounds");
    }

    tv.data = reinterpret_cast<const std::int32_t*>(p + data_pos);
    tv.count = elem_count;
    return tv;
}

}  // namespace annfs::io

