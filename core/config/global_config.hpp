#pragma once
#include <cstdint>
#include <memory>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include <utility>
struct GlobalConfig {
    using ull = unsigned long long;
    static constexpr std::uint64_t kFileReadTag = 1ULL << 63;
    static constexpr std::uint64_t kFileReadFdShift = 32;

    enum class LABEL_DECODE_MODE {
        PACKED_MAP,
        SHIFT_OFF_BITS
    };
    static const char* LabelDecodeModeToString(LABEL_DECODE_MODE mode)
    {
        switch (mode) {
            case LABEL_DECODE_MODE::SHIFT_OFF_BITS:
                return "shift_offbits";
            case LABEL_DECODE_MODE::PACKED_MAP:
            default:
                return "packed_map";
        }
    }

    // ---------- Disk / IO 配置 ----------
    uint32_t LOGIC_BLOCK_SIZE = 0;
    uint64_t DISK_SIZE = 0;
    std::vector<std::string> DEV_PATHS;
    int RING_DEPTH = 256;

    // packed64: [N bits crossed_blocks][64-N bits start_byte]
    // N is configurable via config key `packed64_cross_bits`.
    int PACKED64_CROSS_BITS = 16;
    int PACKED64_OFF_BITS_LEN = 48;
    uint64_t PACKED64_OFF_MASK = (1ULL << 48) - 1ULL;
    int PACKED64_CROSS_SHIFT = 48;

    size_t FS_MAPPER_FILE_LIMIT = 0;
    size_t FS_ROUND_SIZE = 1000000;

    int ASYNC_POLL_US = 500;
    int ASYNC_RECLAIM_BATCH = 64;
    int ASYNC_RECLAIM_THRESHOLD = 256;
    int ASYNC_RECLAIM_INTERVAL_US = 500;
    // 同步/退避路径下的 CQE 回收批次
    int SYNC_RECLAIM_BATCH = 32;
    // submit_and_wait 至少等待完成数
    int SYNC_SUBMIT_WAIT_NR = 1;


    std::string IO_WORKER_MODE = "per_thread";
    bool USE_LABEL_MAP = false;
    LABEL_DECODE_MODE LABEL_DECODE_MODE_CFG = LABEL_DECODE_MODE::PACKED_MAP;
    std::string MAPPED_SPAN_MODE = "bucket_pool";

    std::map<int, int> MAPPED_SPANPOOL_BUCKETS;

    int SPANPOOL_MAX_BLOCKS = 0;
    
    uint64_t BLOCK_OFF_MASK = 0;
    uint64_t BLOCK_ID_MASK = 0;
    int OFF_BITS_LEN = 0;

    // ---------- 全局 Logger ----------
    std::shared_ptr<spdlog::logger> logger;

public:
    static GlobalConfig& instance() {
        static GlobalConfig inst;
        return inst;
    }

    GlobalConfig(const GlobalConfig&) = delete;
    GlobalConfig& operator=(const GlobalConfig&) = delete;

    /**
     * @brief 初始化日志系统
     * @param is_debug - 是否为调试模式
     */
    void init_logger(bool is_debug)
    {
        std::filesystem::create_directories("logs");

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::debug);

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/run.log", true);
        file_sink->set_level(spdlog::level::debug);

        std::vector<spdlog::sink_ptr> sinks { console_sink, file_sink };
        logger = std::make_shared<spdlog::logger>("global_logger", sinks.begin(), sinks.end());

        logger->set_level(is_debug ? spdlog::level::debug : spdlog::level::info);
        logger->flush_on(spdlog::level::debug);
    }

private:
    GlobalConfig() = default;
};
