// FsPrepareFilesTrace.cpp
// -----------------------------------------------------------------------------
// 在 FsPrepareFiles 基础上新增 trace 驱动的逐文件大小能力：
//   --size-trace PATH    .npy 文件（1D int32）
//   --trace-start N      从 trace[N] 开始
//   --trace-count N      允许使用的元素数量（默认 num-files）
//
// 当启用 trace 时：
//   - tree 模式下每个文件大小由 trace 对应元素决定
//   - lab1 仍使用固定 --file-size（与原行为一致）
// -----------------------------------------------------------------------------

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

struct Config {
    std::string mount_point = "/mnt/nvme0n1";
    std::uint64_t num_files = 1000001;
    std::size_t file_size = 4096;
    std::size_t threads = 64;
    bool create_tree = true;
    bool skip_lab1 = false;
    int fanout = 100;
    std::string tree_root = "";
    std::string pattern = "z";

    std::string size_trace = "";
    std::uint64_t trace_start = 0;
    std::uint64_t trace_count = 0;
};

void log_info(const std::string &msg) { std::cout << "[INFO] " << msg << std::endl; }
void log_warn(const std::string &msg) { std::cout << "[WARN] " << msg << std::endl; }
void log_error(const std::string &msg) { std::cerr << "[ERROR] " << msg << std::endl; }

struct TraceView {
    int fd = -1;
    const std::uint8_t *map = nullptr;
    std::size_t map_len = 0;
    const std::int32_t *data = nullptr;
    std::size_t count = 0;

    ~TraceView() {
        if (map != nullptr && map_len > 0) {
            ::munmap(const_cast<std::uint8_t *>(map), map_len);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    TraceView(const TraceView &) = delete;
    TraceView &operator=(const TraceView &) = delete;
    TraceView() = default;

    TraceView(TraceView &&other) noexcept {
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

    TraceView &operator=(TraceView &&other) noexcept {
        if (this == &other) return *this;
        if (map != nullptr && map_len > 0) {
            ::munmap(const_cast<std::uint8_t *>(map), map_len);
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

static std::string extract_header_value(const std::string &header,
                                        const std::string &key,
                                        char quote) {
    std::size_t pos = header.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("NPY header missing key: " + key);
    }
    std::size_t s = header.find(quote, pos + key.size());
    if (s == std::string::npos) throw std::runtime_error("NPY header parse error");
    std::size_t e = header.find(quote, s + 1);
    if (e == std::string::npos) throw std::runtime_error("NPY header parse error");
    return header.substr(s + 1, e - s - 1);
}

static std::vector<std::size_t> parse_shape(const std::string &header) {
    std::size_t pos = header.find("'shape':");
    if (pos == std::string::npos) pos = header.find("\"shape\":");
    if (pos == std::string::npos) {
        throw std::runtime_error("NPY header missing shape");
    }

    std::size_t l = header.find('(', pos);
    std::size_t r = header.find(')', l == std::string::npos ? pos : l);
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

static TraceView load_trace_npy_int32(const std::string &path) {
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
    tv.map = static_cast<const std::uint8_t *>(
        ::mmap(nullptr, tv.map_len, PROT_READ, MAP_PRIVATE, tv.fd, 0));
    if (tv.map == MAP_FAILED) {
        tv.map = nullptr;
        throw std::runtime_error("mmap failed for trace file");
    }

    if (tv.map_len < 10) {
        throw std::runtime_error("trace file too small for npy");
    }

    const std::uint8_t *p = tv.map;
    if (std::memcmp(p, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic");
    }

    std::uint8_t major = p[6];
    std::uint8_t minor = p[7];
    (void)minor;

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

    std::string header(reinterpret_cast<const char *>(p + header_pos), header_len);

    std::string descr;
    try {
        descr = extract_header_value(header, "'descr':", '\'');
    } catch (...) {
        descr = extract_header_value(header, "\"descr\":", '"');
    }

    bool fortran_order =
        (header.find("'fortran_order': True") != std::string::npos) ||
        (header.find("\"fortran_order\": true") != std::string::npos);
    if (fortran_order) {
        throw std::runtime_error("fortran_order npy not supported");
    }

    if (!(descr == "<i4" || descr == "|i4" || descr == "i4")) {
        throw std::runtime_error("unsupported npy dtype for trace: " + descr + " (need int32)");
    }

    std::vector<std::size_t> shape = parse_shape(header);
    std::size_t elem_count = 1;
    for (std::size_t d : shape) {
        elem_count *= d;
    }

    std::size_t data_pos = header_pos + header_len;
    std::size_t need_bytes = elem_count * sizeof(std::int32_t);
    if (data_pos + need_bytes > tv.map_len) {
        throw std::runtime_error("npy data out of bounds");
    }

    tv.data = reinterpret_cast<const std::int32_t *>(p + data_pos);
    tv.count = elem_count;
    return tv;
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for argument: " + arg);
            }
            return argv[++i];
        };

        if (arg == "--mount-point") cfg.mount_point = next();
        else if (arg == "--num-files") cfg.num_files = std::stoull(next());
        else if (arg == "--file-size") cfg.file_size = std::stoull(next());
        else if (arg == "--threads") cfg.threads = std::stoull(next());
        else if (arg == "--no-tree") cfg.create_tree = false;
        else if (arg == "--skip-lab1") cfg.skip_lab1 = true;
        else if (arg == "--fanout") cfg.fanout = std::stoi(next());
        else if (arg == "--tree-root") cfg.tree_root = next();
        else if (arg == "--pattern") cfg.pattern = next();
        else if (arg == "--size-trace") cfg.size_trace = next();
        else if (arg == "--trace-start") cfg.trace_start = std::stoull(next());
        else if (arg == "--trace-count") cfg.trace_count = std::stoull(next());
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return cfg;
}

static void apply_trace_env_overrides(Config &cfg) {
    const char *env_path = std::getenv("FS_TRACE_PATH");
    if (env_path != nullptr && env_path[0] != '\0') {
        cfg.size_trace = env_path;
    }
    const char *env_start = std::getenv("FS_TRACE_START");
    if (env_start != nullptr && env_start[0] != '\0') {
        cfg.trace_start = std::stoull(std::string(env_start));
    }
    const char *env_count = std::getenv("FS_TRACE_COUNT");
    if (env_count != nullptr && env_count[0] != '\0') {
        cfg.trace_count = std::stoull(std::string(env_count));
    }
}

static void refill_random_buffer(std::vector<char> &buf, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    for (char &c : buf) {
        c = static_cast<char>(dist(rng));
    }
}

static void write_one_file(const std::string &path,
                           std::size_t file_size,
                           const std::string &pattern,
                           std::vector<char> &scratch,
                           std::mt19937_64 &rng,
                           bool random_per_file) {
    if (fs::exists(path)) return;

    int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
        std::perror(("open " + path).c_str());
        return;
    }

    constexpr std::size_t kChunk = 1U << 20;
    if (scratch.empty()) {
        scratch.resize(std::min<std::size_t>(kChunk, std::max<std::size_t>(file_size, 1)));
    }
    if (scratch.size() > kChunk) {
        scratch.resize(kChunk);
    }

    if (pattern == "zero") {
        std::fill(scratch.begin(), scratch.end(), 0);
    } else if (pattern == "z") {
        std::fill(scratch.begin(), scratch.end(), 'z');
    } else if (pattern == "random" || pattern == "random-per-file") {
        if (random_per_file || pattern == "random") {
            refill_random_buffer(scratch, rng);
        }
    } else {
        ::close(fd);
        throw std::runtime_error("unknown pattern: " + pattern);
    }

    std::size_t remaining = file_size;
    while (remaining > 0) {
        std::size_t n = std::min<std::size_t>(remaining, scratch.size());
        if (pattern == "random-per-file") {
            refill_random_buffer(scratch, rng);
        }
        ssize_t written = ::write(fd, scratch.data(), n);
        if (written < 0) {
            std::perror(("write " + path).c_str());
            break;
        }
        remaining -= static_cast<std::size_t>(written);
    }

    ::close(fd);
}

void create_lab1_all_files(const Config &cfg) {
    std::string base = cfg.mount_point + "/lab1";
    fs::create_directories(base);

    log_info("开始创建 lab1 文件, 基础路径: " + base);
    auto start = std::chrono::steady_clock::now();

    std::atomic<std::uint64_t> index{0};
    std::atomic<std::uint64_t> done{0};

    auto worker = [&]() {
        std::uint64_t seed =
            static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^
            static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::mt19937_64 rng(seed);
        std::vector<char> scratch;
        const bool random_per_file = (cfg.pattern == "random-per-file");

        while (true) {
            std::uint64_t i = index.fetch_add(1, std::memory_order_relaxed);
            if (i >= cfg.num_files) break;

            std::string path = base + "/" + std::to_string(i) + ".bin";
            write_one_file(path, cfg.file_size, cfg.pattern, scratch, rng, random_per_file);

            std::uint64_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (d % 10000 == 0) {
                std::cout << "\r[lab1] 已创建: " << d << " / " << cfg.num_files << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(cfg.threads);
    for (std::size_t t = 0; t < cfg.threads; ++t) threads.emplace_back(worker);
    for (auto &th : threads) th.join();

    std::cout << "\r[lab1] 已创建: " << cfg.num_files << " / " << cfg.num_files << "\n";

    auto end = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    log_info("lab1 完成, 耗时 " + std::to_string(secs) + " 秒");
}

static std::string tree_path_for_index(const std::string &base,
                                       int fanout,
                                       std::uint64_t num_files,
                                       std::uint64_t index) {
    std::uint64_t total_leaves = static_cast<std::uint64_t>(fanout) * static_cast<std::uint64_t>(fanout);
    std::uint64_t per_leaf = num_files / total_leaves;
    std::uint64_t remainder = num_files % total_leaves;

    std::uint64_t leaf = 0;
    std::uint64_t local = 0;

    if (per_leaf == 0) {
        leaf = index;
        local = 0;
    } else {
        std::uint64_t first_part = remainder * (per_leaf + 1);
        if (index < first_part) {
            leaf = index / (per_leaf + 1);
            local = index % (per_leaf + 1);
        } else {
            std::uint64_t shifted = index - first_part;
            leaf = remainder + shifted / per_leaf;
            local = shifted % per_leaf;
        }
    }

    int i = static_cast<int>(leaf / static_cast<std::uint64_t>(fanout));
    int j = static_cast<int>(leaf % static_cast<std::uint64_t>(fanout));
    return base + "/l" + std::to_string(i) + "/l" + std::to_string(j) +
           "/f_" + std::to_string(local) + ".bin";
}

void create_uniform_tree_and_distribute_trace(const Config &cfg,
                                              const std::int32_t *trace_data,
                                              std::size_t trace_len) {
    if (cfg.fanout <= 0) {
        throw std::runtime_error("fanout must be > 0");
    }

    std::string base = cfg.tree_root.empty() ? (cfg.mount_point + "/tree") : cfg.tree_root;
    fs::create_directories(base);

    std::uint64_t total_leaves = static_cast<std::uint64_t>(cfg.fanout) * static_cast<std::uint64_t>(cfg.fanout);
    for (std::uint64_t leaf = 0; leaf < total_leaves; ++leaf) {
        int i = static_cast<int>(leaf / static_cast<std::uint64_t>(cfg.fanout));
        int j = static_cast<int>(leaf % static_cast<std::uint64_t>(cfg.fanout));
        fs::create_directories(base + "/l" + std::to_string(i) + "/l" + std::to_string(j));
    }

    log_info("开始构建 tree 目录, 基础路径: " + base);
    log_info("叶子目录数: " + std::to_string(total_leaves));
    if (trace_data != nullptr) {
        log_info("启用 trace 逐文件大小: trace_len=" + std::to_string(trace_len));
    } else {
        log_info("固定文件大小模式: file_size=" + std::to_string(cfg.file_size));
    }

    auto start = std::chrono::steady_clock::now();
    std::atomic<std::uint64_t> index{0};
    std::atomic<std::uint64_t> files_done{0};

    auto worker = [&]() {
        std::uint64_t seed =
            static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^
            static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::mt19937_64 rng(seed);
        std::vector<char> scratch;
        const bool random_per_file = (cfg.pattern == "random-per-file");

        while (true) {
            std::uint64_t idx = index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= cfg.num_files) break;

            std::size_t cur_size = cfg.file_size;
            if (trace_data != nullptr) {
                if (idx >= trace_len) {
                    throw std::runtime_error("trace length less than num_files");
                }
                std::int32_t v = trace_data[idx];
                if (v <= 0) {
                    throw std::runtime_error("invalid trace size at local index " + std::to_string(idx) + ": " + std::to_string(v));
                }
                cur_size = static_cast<std::size_t>(v);
            }

            std::string path = tree_path_for_index(base, cfg.fanout, cfg.num_files, idx);
            write_one_file(path, cur_size, cfg.pattern, scratch, rng, random_per_file);

            std::uint64_t d = files_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (d % 10000 == 0) {
                std::cout << "\r[tree] 已创建文件数: " << d << " / " << cfg.num_files << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(cfg.threads);
    for (std::size_t t = 0; t < cfg.threads; ++t) threads.emplace_back(worker);
    for (auto &th : threads) th.join();

    std::cout << "\r[tree] 已创建文件数: " << cfg.num_files << " / " << cfg.num_files << "\n";

    auto end = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    log_info("tree 完成, 耗时 " + std::to_string(secs) + " 秒");
}

int main(int argc, char **argv) {
    try {
        Config cfg = parse_args(argc, argv);
        apply_trace_env_overrides(cfg);

        log_info("mount_point = " + cfg.mount_point);
        log_info("num_files   = " + std::to_string(cfg.num_files));
        log_info("file_size   = " + std::to_string(cfg.file_size));
        log_info("threads     = " + std::to_string(cfg.threads));
        log_info(std::string("create_tree = ") + (cfg.create_tree ? "true" : "false"));
        log_info("pattern     = " + cfg.pattern);
        if (!cfg.tree_root.empty()) log_info("tree_root   = " + cfg.tree_root);

        if (!fs::exists(cfg.mount_point) || !fs::is_directory(cfg.mount_point)) {
            log_error("挂载点不存在或不是目录: " + cfg.mount_point);
            return 1;
        }

        TraceView trace_view;
        const std::int32_t *trace_ptr = nullptr;
        std::size_t trace_len = 0;

        if (!cfg.size_trace.empty()) {
            trace_view = load_trace_npy_int32(cfg.size_trace);
            if (cfg.trace_count == 0) {
                cfg.trace_count = cfg.num_files;
            }
            std::uint64_t end = cfg.trace_start + cfg.trace_count;
            if (end > trace_view.count) {
                throw std::runtime_error("trace 范围越界: trace_count=" + std::to_string(trace_view.count) +
                                         ", need_end=" + std::to_string(end));
            }
            if (cfg.trace_count < cfg.num_files) {
                throw std::runtime_error("trace-count 小于 num-files");
            }
            trace_ptr = trace_view.data + cfg.trace_start;
            trace_len = static_cast<std::size_t>(cfg.trace_count);

            log_info("size_trace  = " + cfg.size_trace);
            log_info("trace_start = " + std::to_string(cfg.trace_start));
            log_info("trace_count = " + std::to_string(cfg.trace_count));
        }

        if (!cfg.skip_lab1) {
            if (trace_ptr != nullptr) {
                log_warn("lab1 保持固定 file-size；trace 仅用于 tree 文件大小");
            }
            create_lab1_all_files(cfg);
        } else {
            log_info("跳过 lab1 文件生成 (--skip-lab1)");
        }

        if (cfg.create_tree) {
            log_info("开始 tree 生成 (fanout=" + std::to_string(cfg.fanout) + ")...");
            create_uniform_tree_and_distribute_trace(cfg, trace_ptr, trace_len);
        }

        log_info("全部完成");
        return 0;
    } catch (const std::exception &ex) {
        log_error(std::string("异常: ") + ex.what());
        return 1;
    }
}
