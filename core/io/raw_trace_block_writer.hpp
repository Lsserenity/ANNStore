#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <liburing.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

namespace annfs::io {

struct TraceWriteRecord {
    // source_label: 原始向量/节点 label。
    std::uint64_t source_label = 0;
    // start_byte: 该记录写入设备的起始字节偏移。
    std::uint64_t start_byte = 0;
    // payload_size: 实际负载大小（不含前置 2 字节 end offset）。
    std::uint32_t payload_size = 0;
    std::uint16_t end_offset_in_block = 0;
    std::uint8_t crossed_blocks = 0;
    std::uint32_t packed_label = 0;  // [3 bits crossed_blocks][29 bits start_byte_low29]
};

struct TraceVerifyResult {
    // checked: 校验条目数；passed/failed: 成功/失败数。
    std::size_t checked = 0;
    std::size_t passed = 0;
    std::size_t failed = 0;
};

// 打包映射字段：[3bit crossed_blocks][29bit start_offset_low29]。
// crossed_blocks: 跨越的额外块数（最大按 7 截断）。
// start_offset_29bit: 起始偏移低 29 位。
// 返回值: packed 32bit 值。
inline std::uint32_t pack_block_label(std::uint8_t crossed_blocks, std::uint32_t start_offset_29bit) {
    const std::uint32_t hi = static_cast<std::uint32_t>(std::min<std::uint8_t>(crossed_blocks, 7)) << 29;
    const std::uint32_t lo = start_offset_29bit & ((1u << 29) - 1);
    return hi | lo;
}

// 可靠读取：循环 pread 直到读满 len。
// fd: 文件描述符；data: 目标缓冲；len: 读取长度；offset: 起始偏移。
// 异常: 读取失败/提前 EOF 时抛异常。
inline void read_all(int fd, char* data, std::size_t len, std::uint64_t offset) {
    std::size_t done = 0;
    while (done < len) {
        const ssize_t n = ::pread(fd, data + done, len - done, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("pread failed: errno=" + std::to_string(errno));
        }
        if (n == 0) {
            throw std::runtime_error("pread reached EOF unexpectedly");
        }
        done += static_cast<std::size_t>(n);
    }
}

// 按顺序布局写入 trace block（允许跨块）。
// dev_path: 设备路径；logic_block_size: 逻辑块；start_offset_bytes: 初始写入偏移；
// label_and_sizes: (label, payload_size) 输入序列；queue_depth: io_uring 队列深度。
// 返回值: 每条记录的写入元信息（用于后续 map/校验）。
inline std::vector<TraceWriteRecord> write_trace_blocks_to_device(
    const std::string& dev_path,
    std::uint32_t logic_block_size,
    std::uint64_t start_offset_bytes,
    const std::vector<std::pair<std::uint64_t, std::uint32_t>>& label_and_sizes,
    unsigned queue_depth = 256) {
    if (logic_block_size == 0) {
        throw std::invalid_argument("logic_block_size must be > 0");
    }
    if (queue_depth == 0) {
        throw std::invalid_argument("queue_depth must be > 0");
    }

    int fd = ::open(dev_path.c_str(), O_WRONLY);
    if (fd < 0) {
        throw std::runtime_error("open device failed: " + dev_path);
    }

    io_uring ring {};
    if (io_uring_queue_init(static_cast<unsigned>(queue_depth), &ring, 0) != 0) {
        ::close(fd);
        throw std::runtime_error("io_uring_queue_init failed");
    }

    std::vector<TraceWriteRecord> records;
    records.reserve(label_and_sizes.size());

    struct InFlightSlot {
        bool used = false;
        std::vector<char> buf;
        std::size_t len = 0;
    };
    std::vector<InFlightSlot> slots(queue_depth);
    int inflight = 0;

    auto reap_one = [&]() {
        io_uring_cqe* cqe = nullptr;
        int rc = io_uring_wait_cqe(&ring, &cqe);
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

    std::uint64_t cur = start_offset_bytes;
    try {
        for (const auto& item : label_and_sizes) {
            while (inflight >= static_cast<int>(queue_depth)) {
                reap_one();
            }

            const std::uint64_t label = item.first;
            const std::uint32_t payload_size = std::max<std::uint32_t>(item.second, 1u);
            const std::uint32_t total_size = payload_size + 2u;

            std::vector<char> buf(total_size, 'z');
            const std::uint16_t end_off = static_cast<std::uint16_t>((cur + total_size - 1) % logic_block_size);
            std::memcpy(buf.data(), &end_off, sizeof(end_off));

            const std::uint32_t start_offset_low29 =
                static_cast<std::uint32_t>(cur & ((1ULL << 29) - 1));
            const std::uint32_t block_span = static_cast<std::uint32_t>(((cur % logic_block_size) + total_size - 1) / logic_block_size);
            const std::uint8_t crossed = static_cast<std::uint8_t>(std::min<std::uint32_t>(block_span, 7u));
            const std::uint32_t packed = pack_block_label(crossed, start_offset_low29);

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

            TraceWriteRecord rec;
            rec.source_label = label;
            rec.start_byte = cur;
            rec.payload_size = payload_size;
            rec.end_offset_in_block = end_off;
            rec.crossed_blocks = crossed;
            rec.packed_label = packed;
            records.push_back(rec);

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
    return records;
}

// Block-aligned placement policy:
// if (2 + payload_size) does not fit in current logic block remaining bytes,
// move start offset to next logic block boundary before writing this record.
// 块对齐布局写入：单条记录不跨逻辑块边界，不够就跳到下一块。
// 返回值: 写入记录元信息。
inline std::vector<TraceWriteRecord> write_trace_blocks_to_device_block_aligned(
    const std::string& dev_path,
    std::uint32_t logic_block_size,
    std::uint64_t start_offset_bytes,
    const std::vector<std::pair<std::uint64_t, std::uint32_t>>& label_and_sizes,
    unsigned queue_depth = 256) {
    if (logic_block_size == 0) {
        throw std::invalid_argument("logic_block_size must be > 0");
    }
    if (queue_depth == 0) {
        throw std::invalid_argument("queue_depth must be > 0");
    }

    int fd = ::open(dev_path.c_str(), O_WRONLY);
    if (fd < 0) {
        throw std::runtime_error("open device failed: " + dev_path);
    }

    io_uring ring {};
    if (io_uring_queue_init(static_cast<unsigned>(queue_depth), &ring, 0) != 0) {
        ::close(fd);
        throw std::runtime_error("io_uring_queue_init failed");
    }

    std::vector<TraceWriteRecord> records;
    records.reserve(label_and_sizes.size());

    struct InFlightSlot {
        bool used = false;
        std::vector<char> buf;
        std::size_t len = 0;
    };
    std::vector<InFlightSlot> slots(queue_depth);
    int inflight = 0;

    auto reap_one = [&]() {
        io_uring_cqe* cqe = nullptr;
        int rc = io_uring_wait_cqe(&ring, &cqe);
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

    std::uint64_t cur = start_offset_bytes;
    try {
        for (const auto& item : label_and_sizes) {
            while (inflight >= static_cast<int>(queue_depth)) {
                reap_one();
            }

            const std::uint64_t label = item.first;
            const std::uint32_t payload_size = std::max<std::uint32_t>(item.second, 1u);
            const std::uint32_t total_size = payload_size + 2u;

            const std::uint64_t rem = cur % logic_block_size;
            if (rem != 0 && rem + total_size > logic_block_size) {
                cur += (logic_block_size - rem);  // move to next block boundary
            }

            std::vector<char> buf(total_size, 'z');
            const std::uint16_t end_off = static_cast<std::uint16_t>((cur + total_size - 1) % logic_block_size);
            std::memcpy(buf.data(), &end_off, sizeof(end_off));

            const std::uint32_t start_offset_low29 =
                static_cast<std::uint32_t>(cur & ((1ULL << 29) - 1));
            const std::uint32_t block_span = static_cast<std::uint32_t>(((cur % logic_block_size) + total_size - 1) / logic_block_size);
            const std::uint8_t crossed = static_cast<std::uint8_t>(std::min<std::uint32_t>(block_span, 7u));
            const std::uint32_t packed = pack_block_label(crossed, start_offset_low29);

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

            TraceWriteRecord rec;
            rec.source_label = label;
            rec.start_byte = cur;
            rec.payload_size = payload_size;
            rec.end_offset_in_block = end_off;
            rec.crossed_blocks = crossed;
            rec.packed_label = packed;
            records.push_back(rec);

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
    return records;
}

inline TraceVerifyResult verify_trace_blocks_from_device(
    const std::string& dev_path,
    const std::vector<TraceWriteRecord>& records,
    std::size_t max_check = 0) {
    int fd = ::open(dev_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("open device for verify failed: " + dev_path);
    }

    const std::size_t limit = (max_check == 0)
        ? records.size()
        : std::min<std::size_t>(max_check, records.size());

    TraceVerifyResult res;
    res.checked = limit;

    try {
        for (std::size_t i = 0; i < limit; ++i) {
            const auto& r = records[i];
            const std::size_t total_size = static_cast<std::size_t>(r.payload_size) + 2;
            std::vector<char> buf(total_size);
            read_all(fd, buf.data(), buf.size(), r.start_byte);

            std::uint16_t got_end_off = 0;
            std::memcpy(&got_end_off, buf.data(), sizeof(got_end_off));

            bool ok = (got_end_off == r.end_offset_in_block);
            for (std::size_t j = 2; ok && j < buf.size(); ++j) {
                if (buf[j] != 'z') ok = false;
            }

            if (ok) ++res.passed;
            else ++res.failed;
        }
    } catch (...) {
        ::close(fd);
        throw;
    }

    ::close(fd);
    return res;
}

}  // namespace annfs::io
