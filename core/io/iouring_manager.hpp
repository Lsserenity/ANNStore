#pragma once

#include <liburing.h>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <thread>
#include <map>
#include <unordered_map>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#include "../config/global_config.hpp"
#include "common/query_io_events.hpp"
#include "mapped_io_plan.hpp"
#include "mapped_label_offset_resolver.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"

// #ifndef io_uring_sqe_set_data64
// static inline void io_uring_sqe_set_data64(struct io_uring_sqe* sqe, std::uint64_t data) {
//     io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<std::uintptr_t>(data)));
// }
// #endif

// #ifndef io_uring_cqe_get_data64
// static inline std::uint64_t io_uring_cqe_get_data64(const struct io_uring_cqe* cqe) {
//     return static_cast<std::uint64_t>(
//         reinterpret_cast<std::uintptr_t>(io_uring_cqe_get_data(cqe))
//     );
// }
// #endif

class IOuringManager
{
public:
    // 作用：提交一批 req（可携带 parse_hints）。异步模式下只入队，不等待完成。
    int prefetch_reqs_overlap(
        const std::vector<annfs::io::BlockReq>& req_list,
        int query_id = -1)
    {
        if (req_list.empty()) return 0;
        if (fds_.empty()) {
            throw std::runtime_error("IOuringManager: no device registered (fds_ empty).");
        }
        AsyncTask task;
        task.type = AsyncTask::Type::Reqs;
        task.reqs = req_list;
        task.query_id = query_id;
        task.preaccounted_count = clamp_preaccount_count(req_list.size());
        return submit_or_enqueue_overlap(
            static_cast<std::size_t>(task.preaccounted_count),
            std::move(task),
            [&]() { return submit_reqs_sync(req_list, query_id); });
    }

    // ================================================================================================
    // 新增：用于 ANN 搜索过程中 overlap 的“纯 submit”预取接口（不等待完成）
    // ================================================================================================
    // 作用：提交一批 offset。异步模式下只入队，不等待完成。
    // 细节：offset 并不总是“已解码字节偏移”。
    // - SHIFT_OFF_BITS: 视作 label-id，需左移 OFF_BITS_LEN 再读 1-block；
    // - PACKED_MAP: 视作 packed64，需 DecodePacked64 后按 split/合并策略构造 req。
    int prefetch_offsets_overlap(const std::vector<off_t>& offset_list,
                                 int query_id = -1)
    {
        auto reqs = build_reqs_from_offsets(offset_list);
        return prefetch_reqs_overlap(reqs, query_id);
    }
    // 搜索循环里偶尔调一下，非阻塞回收完成的 IO
    // 作用：同步模式下回收已完成的 CQE，不阻塞。
    void poll_completions_nonblock(int max_to_reclaim = 32)
    {
        if (use_async_submit_) {
            return;
        }
        reclaim_completions(max_to_reclaim);
    }

    // 某些场景（比如 query 完成、统计）想收干净所有 IO（非阻塞版本）
    // 只回收已经完成的IO，不等待未完成的IO
    // 作用：同步模式下尽可能回收已完成 IO，不等待未完成 IO。
    void drain_all_completions()
    {
        if (use_async_submit_) {
            return;
        }
        reclaim_completions(-1);  // -1 表示不限，但只回收已完成的
    }

    // 阻塞等待所有IO完成（用于query结束时确保所有IO都完成）
    // 作用：等待本次 pending_ios 对应的 IO 完成。异步模式下通过 outstanding 计数和 CV 等待。
    void wait_all_completions(int pending_ios)
    {   
        if (pending_ios <= 0) {
            return;
        }
        if (!use_async_submit_) {
            wait_all_completions_sync(pending_ios);
            return;
        }
        // auto logger = GlobalConfig::instance().logger;
        // if (logger) {
        //     logger->debug("[IOU][async] wait_all_completions pending={} outstanding={}",
        //                   pending_ios,
        //                   outstanding_ios_.load(std::memory_order_relaxed));
        // }
        long target = outstanding_ios_.load(std::memory_order_relaxed) -
                      static_cast<long>(pending_ios);
        if (target < 0) {
            target = 0;
        }
        std::unique_lock<std::mutex> lk(completion_mutex_);
        while (outstanding_ios_.load(std::memory_order_relaxed) > target ||
               async_has_pending_work()) {
            completion_cv_.wait_for(lk, std::chrono::seconds(1));
            debug_query_io("async_wait_tick",
                           -1,
                           pending_ios,
                           outstanding_ios_.load(std::memory_order_relaxed),
                           target);
            // if (logger) {
            //     logger->debug("[IOU][async] wait_all_completions still waiting outstanding={} target={}",
            //                  outstanding_ios_.load(std::memory_order_relaxed),
            //                  target);
            // }
        }
        // if (logger) {
        //     logger->debug("[IOU][async] wait_all_completions done outstanding={}",
        //                   outstanding_ios_.load(std::memory_order_relaxed));
        // }
    }

    // 作用：等待所有 outstanding IO 完成（同步/异步通用）。
    void wait_all_completions_all()
    {
        if (!use_async_submit_) {
            int pending = static_cast<int>(
                outstanding_ios_.load(std::memory_order_relaxed));
            if (pending > 0) {
                wait_all_completions_sync(pending);
            }
            return;
        }
        // auto logger = GlobalConfig::instance().logger;
        // if (logger) {
        //     logger->debug("[IOU][async] wait_all_completions_all outstanding={}",
        //                   outstanding_ios_.load(std::memory_order_relaxed));
        // }
        std::unique_lock<std::mutex> lk(completion_mutex_);
        while (outstanding_ios_.load(std::memory_order_relaxed) != 0 ||
               async_has_pending_work()) {
            completion_cv_.wait_for(lk, std::chrono::seconds(1));
        }
        // if (logger) {
        //     logger->debug("[IOU][async] wait_all_completions_all done");
        // }
    }

    // 作用：同步提交 offset，并用 submit_and_wait 阻塞等待完成。
    int submit_offsets_and_wait(const std::vector<off_t>& offset_list,
                                int query_id = -1)
    {
        if (offset_list.empty()) {
            debug_query_io("submit_offsets_and_wait_empty", query_id, 0, 0);
            return 0;
        }
        debug_query_io("submit_offsets_and_wait_begin",
                       query_id,
                       offset_list.size(),
                       outstanding_ios_.load(std::memory_order_relaxed));
        return submit_and_wait_common(
            offset_list.size(),
            [&]() { return submit_offsets_sync(offset_list, query_id); },
            true,
            query_id);
    }

    // 作用：按文件路径批量读（提交后阻塞等待完成）。
    int submit_files_and_wait(const std::vector<std::string>& file_paths,
                              int query_id = -1)
    {
        if (file_paths.empty()) {
            debug_query_io("submit_files_and_wait_empty", query_id, 0, 0);
            return 0;
        }
        debug_query_io("submit_files_build_begin",
                       query_id,
                       file_paths.size(),
                       outstanding_ios_.load(std::memory_order_relaxed));
        auto file_reqs = build_file_read_reqs(file_paths, query_id);
        if (file_reqs.empty()) {
            debug_query_io("submit_files_build_empty",
                           query_id,
                           file_paths.size(),
                           outstanding_ios_.load(std::memory_order_relaxed));
            return 0;
        }
        debug_query_io("submit_files_and_wait_begin",
                       query_id,
                       file_reqs.size(),
                       outstanding_ios_.load(std::memory_order_relaxed));
        return submit_and_wait_common(
            file_reqs.size(),
            [&]() { return submit_file_reqs_sync(file_reqs, query_id); },
            true,
            query_id);
    }

    // 作用：按文件路径批量读（只提交不等待，用于 overlap）。
    int prefetch_files_overlap(const std::vector<std::string>& file_paths,
                               int query_id = -1)
    {
        if (file_paths.empty()) {
            return 0;
        }
        auto file_reqs = build_file_read_reqs(file_paths, query_id);
        if (file_reqs.empty()) {
            return 0;
        }
        AsyncTask task;
        task.type = AsyncTask::Type::Files;
        task.file_reqs = std::move(file_reqs);
        task.query_id = query_id;
        task.preaccounted_count = clamp_preaccount_count(task.file_reqs.size());
        return submit_or_enqueue_overlap(
            static_cast<std::size_t>(task.preaccounted_count),
            std::move(task),
            [&]() { return submit_file_reqs_sync(task.file_reqs, query_id); });
    }

    // ================================================================================================
    // 单次批读测试（不依赖 iovec 注册）
    // ================================================================================================
    // 作用：测试接口，按文件路径批量读（不使用 fixed buffer）。
    bool batch_read_file(const std::vector<std::string>& file_paths)
    {

        int total = file_paths.size();
        if (total_slots_ < total) {
            throw std::runtime_error("Not enough registered buffers for batch read");
        }
        for (int i = 0; i < total; ++i) 
        {
            int fd = open(file_paths[i].c_str(),O_RDONLY | O_DIRECT);
            // int fd = open(file_paths[i].c_str(),O_RDONLY);
            if (fd < 0) {
                perror("open");
                return false;
            }

            auto sqe = io_uring_get_sqe(&ring_);
            iovec* iv = iovec_by_global_id(i);
            if (!iv) {
                close(fd);
                return false;
            }
            io_uring_prep_read(sqe, fd, iv->iov_base, iv->iov_len, 0);
            io_uring_sqe_set_data(sqe, (void*)(intptr_t)fd);
        }

        io_uring_submit(&ring_);

        int completed = 0;

        while (completed < total)
        {
            struct io_uring_cqe* cqe;
            int ret = io_uring_wait_cqe(&ring_, &cqe);
            if (ret < 0)
                throw std::runtime_error("wait_cqe error");

            int fd = (int)(intptr_t)io_uring_cqe_get_data(cqe);
            if (cqe->res < 0)
                printf("IO error %d\n", cqe->res);

            close(fd);
            completed++;

            io_uring_cqe_seen(&ring_, cqe);
        }
        return true;
    }

    // 作用：同步提交 offset 并等待完成。
    bool batch_read_offset(const std::vector<off_t>& offset_list,
                           int query_id = -1)
    {
        std::vector<annfs::io::BlockReq> reqs;
        reqs.reserve(offset_list.size());
        for (off_t off : offset_list) {
            reqs.push_back(annfs::io::BlockReq{off, 1u, {}});
        }
        return batch_read_reqs(reqs, query_id);
    }

    // 作用：同步提交 req 并等待完成（可携带 parse_hints）。
    bool batch_read_reqs(const std::vector<annfs::io::BlockReq>& req_list,
                         int query_id = -1)
    {
        if (req_list.empty()) {
            return true;
        }
        int submitted = submit_and_wait_common(
            req_list.size(),
            [&]() { return submit_reqs_sync(req_list, query_id); },
            false,
            query_id);
        return submitted == static_cast<int>(req_list.size());
    }

    // 统一 mapped 同步入口：
    // labels 根据全局参数（use_label_map + decode_mode + mapped_span_mode）构造请求并同步提交。
    int mapped_read_sync(const std::vector<unsigned long long>& labels,
                         const ParamConfig& pc,
                         int query_id,
                         std::size_t* missing,
                         std::size_t* issued_io)
    {
        debug_query_io("mapped_read_build_begin",
                       query_id,
                       labels.size(),
                       outstanding_ios_.load(std::memory_order_relaxed));
        auto reqs = build_reqs_from_labels(labels, pc, missing, issued_io, query_id);
        debug_query_io("mapped_read_build_done",
                       query_id,
                       reqs.size(),
                       outstanding_ios_.load(std::memory_order_relaxed),
                       missing ? static_cast<long long>(*missing) : -1);
        const bool ok = batch_read_reqs(reqs, query_id);
        debug_query_io("mapped_read_done",
                       query_id,
                       ok ? reqs.size() : 0,
                       outstanding_ios_.load(std::memory_order_relaxed),
                       missing ? static_cast<long long>(*missing) : -1);
        return ok ? static_cast<int>(reqs.size()) : 0;
    }

    // 统一 mapped 异步入口：
    // wait_per_query=true 时提交后等待；false 仅提交（overlap）。
    int mapped_prefetch(const std::vector<unsigned long long>& labels,
                        const ParamConfig& pc,
                        int query_id,
                        bool wait_per_query,
                        std::size_t* missing,
                        std::size_t* issued_io)
    {
        auto reqs = build_reqs_from_labels(labels, pc, missing, issued_io, query_id);
        debug_query_io("mapped_prefetch_build_done",
                       query_id,
                       reqs.size(),
                       outstanding_ios_.load(std::memory_order_relaxed),
                       missing ? static_cast<long long>(*missing) : -1);
        int pending = prefetch_reqs_overlap(reqs, query_id);
        debug_query_io("mapped_prefetch_submitted",
                       query_id,
                       pending,
                       outstanding_ios_.load(std::memory_order_relaxed),
                       missing ? static_cast<long long>(*missing) : -1);
        if (wait_per_query && pending > 0) {
            debug_query_io("mapped_prefetch_wait_begin",
                           query_id,
                           pending,
                           outstanding_ios_.load(std::memory_order_relaxed));
            wait_all_completions(pending);
            debug_query_io("mapped_prefetch_wait_done",
                           query_id,
                           pending,
                           outstanding_ios_.load(std::memory_order_relaxed));
        }
        return pending;
    }

    // ================================================================================================
    // 构造函数
    // ================================================================================================
    // register_buffer: 是否注册 fixed buffer；numa_node_for_iovecs: iovec 绑定 NUMA 节点；
    // use_async_submit: 是否启动异步提交线程模式。
    IOuringManager(bool register_buffer = true,
                   int numa_node_for_iovecs = -1,
                   bool use_async_submit = false)
    {
        const auto& gc = GlobalConfig::instance();
        io_depth_ =  gc.RING_DEPTH;
        block_size_ = gc.LOGIC_BLOCK_SIZE;
        disk_size_ = gc.DISK_SIZE;
        numa_node_for_iovecs_ = numa_node_for_iovecs;
        async_poll_us_ = gc.ASYNC_POLL_US;
        async_reclaim_batch_ = gc.ASYNC_RECLAIM_BATCH;
        async_reclaim_threshold_ = gc.ASYNC_RECLAIM_THRESHOLD;
        async_reclaim_interval_us_ = gc.ASYNC_RECLAIM_INTERVAL_US;
        sync_reclaim_batch_ = std::max(1, gc.SYNC_RECLAIM_BATCH);
        sync_submit_wait_nr_ = std::max(1, gc.SYNC_SUBMIT_WAIT_NR);
        io_debug_diagnostics_enabled_ =
            gc.logger && gc.logger->should_log(spdlog::level::debug);

        int ret = io_uring_queue_init(io_depth_, &ring_, 0);
        if (ret != 0) {
            if (gc.logger) {
                auto logger = gc.logger;
                logger->error("[IOU] io_uring_queue_init failed ret={} errno={}", ret, errno);
            }
            throw std::system_error(errno, std::generic_category(),
                                    "io_uring_queue_init failed");
        }

        if (register_buffer) {
            initRegisterByBucketMap(gc.MAPPED_SPANPOOL_BUCKETS);
        }

        registerDevpath(gc.DEV_PATHS);

        if (use_async_submit) {
            enable_async_submit();
        }
    }

    // 析构函数：停止异步线程、注销文件、释放 ring 和注册缓冲。
    ~IOuringManager()
    {
        if (use_async_submit_) {
            stop_async_submit();
        }
        if (!fds_.empty()) {
            io_uring_unregister_files(&ring_);
            for (int fd : fds_) {
                if (fd >= 0) {
                    close(fd);
                }
            }
            fds_.clear();
        }
        io_uring_queue_exit(&ring_);
        // 注意：registered_iovecs_ 里的内存是我们自己 posix_memalign 的，需要手动 free
        for (auto& iv : registered_iovecs_) {
            if (iv.iov_base) {
                free(iv.iov_base);
                iv.iov_base = nullptr;
            }
        }
    }

    // 开启异步提交模式。
    // multi_producer=true 时使用 MPSC 队列，适合多线程共享 worker。
    void enable_async_submit(bool multi_producer = false)
    {
        if (use_async_submit_) {
            return;
        }
        async_queue_mode_ = multi_producer ? AsyncQueueMode::Mpsc
                                           : AsyncQueueMode::Spsc;
        use_async_submit_ = true;
        stop_async_submit_.store(false, std::memory_order_relaxed);
        ensure_task_queue();
        submit_thread_ = std::thread([this] { async_submit_loop(); });
    }

    // 配置异步模式参数。
    // use_submit_and_wait: 是否使用 submit_and_wait 驱动；
    // poll_us: 空轮询间隔；reclaim_batch/threshold/interval_us: 回收策略参数。
    void configure_async_submit(bool use_submit_and_wait)
    {
        const auto& gc = GlobalConfig::instance();
        async_use_submit_and_wait_ = use_submit_and_wait;
        async_poll_us_ = gc.ASYNC_POLL_US;
        async_reclaim_batch_ = gc.ASYNC_RECLAIM_BATCH;
        async_reclaim_threshold_ = gc.ASYNC_RECLAIM_THRESHOLD;
        async_reclaim_interval_us_ = gc.ASYNC_RECLAIM_INTERVAL_US;
        sync_reclaim_batch_ = std::max(1, gc.SYNC_RECLAIM_BATCH);
        sync_submit_wait_nr_ = std::max(1, gc.SYNC_SUBMIT_WAIT_NR);
    }

    // 设置查询事件监听器（记录每个 query 的 IO 提交/完成计数）。
    void set_query_listener(annfs::io::QueryIoEventListener* listener)
    {
        query_listener_ = listener;
    }

    // 返回首个设备 fd，失败返回 -1。
    int primary_fd() const
    {
        if (fds_.empty()) {
            return -1;
        }
        return fds_[0];
    }

    std::uint64_t submitted_total() const
    {
        return submitted_total_.load(std::memory_order_relaxed);
    }

    std::uint64_t completed_total() const
    {
        return completed_total_.load(std::memory_order_relaxed);
    }

    std::uint64_t wait_calls_total() const
    {
        return wait_calls_total_.load(std::memory_order_relaxed);
    }

    std::uint64_t wait_requested_total() const
    {
        return wait_requested_total_.load(std::memory_order_relaxed);
    }

    bool diagnostics_enabled() const
    {
        return io_debug_diagnostics_enabled_;
    }

private:
    // ================================================================================================
    // 成员变量
    // ================================================================================================
    int io_depth_ = 0;

    struct io_uring ring_{};
    struct io_uring_params params_{};

    std::vector<int> fds_;

    struct FileFdState {
        int fd = -1;
        std::atomic<int> remaining{0};
    };

    // 每个注册 buffer（iovec）对应一个 slot
    struct Slot {
        // 当前请求的起始 offset（调试/统计用途）。
        off_t offset = 0;
        // 槽位容量（固定值）：该槽位属于几块的 bucket（例如 1/2/4 block）。
        int capacity_blocks = 1;
        // 当前槽位绑定的 query id（完成时回传计数用）。
        int query_id = -1;
        // 当前请求基准偏移（用于解析和日志）。
        off_t req_base = 0;
        // 当前请求实际读取的连续块数。
        std::uint32_t req_blocks = 1;
        // 相对 req_base 的偏移列表（用于 split/merge 后反解析定位）。
        std::vector<std::uint32_t> off_list;
        // 文件路径读场景绑定的临时 fd（用于 completion 兜底关闭）。
        int file_fd = -1;
        std::shared_ptr<FileFdState> file_fd_state;
    };

    struct BucketRange {
        std::size_t begin = 0;
        std::size_t end = 0;
    };
    // 一维注册 iovec：global slot_id == 下标，直接用于 read_fixed 的 buf_index。
    std::vector<iovec> registered_iovecs_;
    // 一维 slot 运行态：global slot_id == 下标。
    std::vector<Slot> slots_;
    // blocks -> 在一维数组中的连续区间 [begin, end)。
    std::map<int, BucketRange> bucket_ranges_;
    // 只读索引：blocks -> bucket 下标（用于 O(1) 找到范围和游标）。
    std::unordered_map<int, std::size_t> bucket_index_;
    std::vector<BucketRange> bucket_ranges_flat_;
    // 每个 bucket 的原子轮询游标（分配起点），减少热点槽位竞争。
    std::unique_ptr<std::atomic<std::uint32_t>[]> bucket_probe_cursors_;
    // 每个 slot 的占用位：0=free, 1=in_use；通过 CAS 实现无锁抢占。
    std::unique_ptr<std::atomic<std::uint8_t>[]> slot_in_use_;
    int total_slots_ = 0;
    // 所有注册 bucket 中最大的 blocks，用于限制单次请求上界（防止越界到不存在的 bucket）。
    std::uint32_t max_registered_blocks_ = 1;
    // slot 元数据锁：保护分配/释放/上下文写入，避免多线程 shared-worker 下竞争。
    mutable std::mutex slot_mutex_;

    GlobalConfig::ull block_size_ = 0;
    GlobalConfig::ull disk_size_ = 0;

    int numa_node_for_iovecs_ = -1;

    std::atomic<long> outstanding_ios_{0};
    std::atomic<std::uint64_t> submitted_total_{0};
    std::atomic<std::uint64_t> completed_total_{0};
    std::atomic<std::uint64_t> wait_calls_total_{0};
    std::atomic<std::uint64_t> wait_requested_total_{0};
    bool io_debug_diagnostics_enabled_ = false;
    bool use_async_submit_ = false;
    bool async_use_submit_and_wait_ = false;
    int async_poll_us_ = 0;
    int async_reclaim_batch_ = 0;
    int async_reclaim_threshold_ = 0;
    int async_reclaim_interval_us_ = 0;
    // 同步/退避路径 reclaim 批次（替代硬编码 32）。
    int sync_reclaim_batch_ = 32;
    // submit_and_wait 最少等待完成数（替代硬编码 1）。
    int sync_submit_wait_nr_ = 1;
    std::chrono::steady_clock::time_point last_reclaim_ts_ =
        std::chrono::steady_clock::now();
    std::atomic<bool> stop_async_submit_{false};
    std::atomic<int> async_active_tasks_{0};
    std::thread submit_thread_;
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
    annfs::io::QueryIoEventListener* query_listener_ = nullptr;

    const size_t async_queue_capacity_ = 1024;

    struct FileReadReq {
        std::string path;
        off_t offset = 0;
        std::uint32_t blocks = 1;
        std::shared_ptr<FileFdState> fd_state;
    };

    struct AsyncTask {
        enum class Type {
            Reqs,
            Files
        };

        Type type = Type::Reqs;
        std::vector<annfs::io::BlockReq> reqs;
        std::vector<FileReadReq> file_reqs;
        int query_id = -1;
        long preaccounted_count = 0;
    };

    class AsyncTaskQueue {
    public:
        virtual ~AsyncTaskQueue() = default;
        virtual bool enqueue(AsyncTask&& item) = 0;
        virtual bool dequeue(AsyncTask& out) = 0;
        virtual bool empty() const = 0;
    };

    class SpscQueue final : public AsyncTaskQueue {
    public:
        // 作用：构造单生产者单消费者环形队列，capacity_ = N+1 便于区分空/满。
        explicit SpscQueue(size_t capacity)
            : capacity_(capacity + 1),
              buffer_(capacity_),
              head_(0),
              tail_(0) {}

        // SPSC 环形队列：单生产者单消费者，无互斥锁（原子 head/tail）。
        bool enqueue(AsyncTask&& item) override
        {
            const size_t tail = tail_.load(std::memory_order_relaxed);
            const size_t next = (tail + 1) % capacity_;
            if (next == head_.load(std::memory_order_acquire)) {
                return false;
            }
            buffer_[tail] = std::move(item);
            tail_.store(next, std::memory_order_release);
            return true;
        }

        // 作用：消费者无锁出队一个任务；队空返回 false。
        bool dequeue(AsyncTask& out) override
        {
            const size_t head = head_.load(std::memory_order_relaxed);
            if (head == tail_.load(std::memory_order_acquire)) {
                return false;
            }
            out = std::move(buffer_[head]);
            head_.store((head + 1) % capacity_, std::memory_order_release);
            return true;
        }

        // 作用：判空（SPSC 仅依据 head/tail）。
        bool empty() const override
        {
            return head_.load(std::memory_order_acquire) ==
                   tail_.load(std::memory_order_acquire);
        }

    private:
        size_t capacity_;
        std::vector<AsyncTask> buffer_;
        std::atomic<size_t> head_;
        std::atomic<size_t> tail_;
    };

    class MpscQueue final : public AsyncTaskQueue {
    public:
        // 作用：构造多生产者单消费者队列，使用 mutex 保护 head/tail。
        explicit MpscQueue(size_t capacity)
            : capacity_(capacity + 1),
              buffer_(capacity_),
              head_(0),
              tail_(0) {}

        // MPSC 模式下使用轻量互斥保护 tail/head，保证多生产者并发安全。
        bool enqueue(AsyncTask&& item) override
        {
            std::lock_guard<std::mutex> lk(mutex_);
            const size_t next = (tail_ + 1) % capacity_;
            if (next == head_) {
                return false;
            }
            buffer_[tail_] = std::move(item);
            tail_ = next;
            return true;
        }

        // 作用：消费者出队一个任务；队空返回 false。
        bool dequeue(AsyncTask& out) override
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (head_ == tail_) {
                return false;
            }
            out = std::move(buffer_[head_]);
            head_ = (head_ + 1) % capacity_;
            return true;
        }

        // 作用：判空（MPSC 在锁内读取游标）。
        bool empty() const override
        {
            std::lock_guard<std::mutex> lk(mutex_);
            return head_ == tail_;
        }

    private:
        size_t capacity_;
        std::vector<AsyncTask> buffer_;
        size_t head_;
        size_t tail_;
        mutable std::mutex mutex_;
    };

    enum class AsyncQueueMode {
        Spsc,
        Mpsc
    };

    std::unique_ptr<AsyncTaskQueue> task_queue_;
    AsyncQueueMode async_queue_mode_ = AsyncQueueMode::Spsc;

    // ================================================================================================
    // NUMA 对齐分配
    // ================================================================================================
    // 作用：按页对齐分配内存，可选绑定到 NUMA 节点。
    int allocate_aligned_on_node(void*& buf, size_t alignment, size_t size, int numa_node = -1)
    {
        if (numa_node < 0 || numa_available() < 0)
            return posix_memalign(&buf, alignment, size);

        if (posix_memalign(&buf, alignment, size) != 0)
            return -1;

        size_t page_size = getpagesize();
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;

        unsigned long nodemask = 1ULL << numa_node;
        unsigned long maxnode  = sizeof(nodemask) * 8;

        if (mbind(buf, aligned_size, MPOL_BIND, &nodemask, maxnode,
                  MPOL_MF_MOVE | MPOL_MF_STRICT) != 0)
        {
            free(buf);
            return -1;
        }

        return 0;
    }

    // ================================================================================================
    // 注册 IOVEC 缓冲区
    // ================================================================================================
    // 作用：按 map<blocks,count> 注册 fixed buffer，用于 read_fixed。
    // 过程：
    // 1) 为每个 bucket 申请对齐内存并构造 iovec；
    // 2) 同时建立 global slot_id <-> (blocks, local_idx) O(1) 映射；
    // 3) 展平成 flat_iovecs 交给 io_uring_register_buffers 一次性注册。
    // 注意：
    // - io_uring 对同一 ring 只维护“一套当前生效的 registered buffers”；
    // - 需要重配时应先 unregister，再按新 flat_iovecs 重新 register；
    // - 业务侧仍可保留 bucket map 结构，kernel 侧必须是扁平 iovec 数组。
    void initRegisterByBucketMap(const std::map<int, int>& raw_buckets)
    {
        // 先清空旧注册态（仅清理本地容器；真正替换 kernel 注册集依赖后面的 register_buffers 调用）。
        registered_iovecs_.clear();
        // 与 registered_iovecs_ 同下标的运行态槽位元数据，必须同步清空。
        slots_.clear();
        // blocks -> [begin,end) 的区间映射（有序 map，便于稳定遍历）重建前清空。
        bucket_ranges_.clear();
        // blocks -> bucket 下标的 O(1) 索引重建前清空。
        bucket_index_.clear();
        // bucket_ranges_ 的扁平副本（vector，下标访问更快）重建前清空。
        bucket_ranges_flat_.clear();
        // 每个 bucket 的原子轮询游标数组，旧指针释放。
        bucket_probe_cursors_.reset();
        // 每个 slot 的占用位数组，旧指针释放。
        slot_in_use_.reset();
        // 槽位总数重置；后面按 registered_iovecs_.size() 重新计算。
        total_slots_ = 0;
        // 最大可注册块数重置为 1，后面在遍历 bucket 时更新。
        max_registered_blocks_ = 1;

        // 遍历用户给的 bucket 配置：key=连续块数 blocks，value=该 blocks 下要分配几个 slot。
        for (const auto& kv : raw_buckets) {
            // 跳过非法 bucket（blocks<=0 或 count<=0），避免分配无意义/越界配置。
            if (kv.first <= 0 || kv.second <= 0) {
                continue;
            }
            // begin: 当前 bucket 在一维 registered_iovecs_ 里的起始下标（含）。
            const std::size_t begin = registered_iovecs_.size();
            // len: 该 bucket 每个 slot 的字节容量 = block_size_ * blocks。
            const std::size_t len = static_cast<std::size_t>(block_size_) * kv.first;
            // 在该 bucket 下分配 kv.second 个同容量 slot。
            for (int i = 0; i < kv.second; ++i) {
                // buf: 单个 slot 的数据缓冲起始地址（后续塞入 iovec.iov_base）。
                void* buf = nullptr;
                // 对齐分配：按 block_size_ 对齐，长度 len，可选绑到 numa_node_for_iovecs_。
                if (allocate_aligned_on_node(
                        buf, block_size_, len, numa_node_for_iovecs_) != 0) {
                    if (GlobalConfig::instance().logger) {
                        auto logger = GlobalConfig::instance().logger;
                        logger->error(
                            "[IOU] bucket allocation failed blocks={} local_index={} len={} numa_node={}",
                            kv.first,
                            i,
                            len,
                            numa_node_for_iovecs_);
                    }
                    throw std::runtime_error("Memory alignment failed");
                }
                // 可选的安全初始化：清零，减少调试时脏数据干扰。
                memset(buf, 0, len);

                // iv: io_uring 注册用的单项 iovec 描述符。
                iovec iv{};
                // iov_base 指向刚分配的对齐缓冲。
                iv.iov_base = buf;
                // iov_len 是该 slot 总容量（字节）。
                iv.iov_len = len;
                // push 到全局一维数组，最终作为 register_buffers 输入。
                registered_iovecs_.push_back(iv);
                // slot: 与 registered_iovecs_ 同下标的运行态元数据。
                Slot slot{};
                // capacity_blocks 固定记录这个 slot 属于哪个 blocks bucket。
                slot.capacity_blocks = kv.first;
                // 放入 slots_，保证 slot_id -> slot 元数据 O(1)。
                slots_.push_back(std::move(slot));
                // 维护全局最大 blocks；后续用于请求裁剪，避免申请不存在的 bucket。
                max_registered_blocks_ = std::max<std::uint32_t>(
                    max_registered_blocks_,
                    static_cast<std::uint32_t>(kv.first));
            }
            // end: 当前 bucket 在一维数组里的结束下标（不含）。
            const std::size_t end = registered_iovecs_.size();
            // 仅当这个 bucket 实际产生 slot 时，记录区间映射 blocks -> [begin,end)。
            if (end > begin) {
                bucket_ranges_[kv.first] = BucketRange{begin, end};
            }
        }

        // 所有合法 bucket 都没产出 slot，属于配置错误，直接抛异常。
        if (registered_iovecs_.empty()) {
            throw std::runtime_error("initRegisterByBucketMap: empty valid buckets");
        }
        // 全局 slot 总数 = 一维 iovec 数组长度（两者严格同下标一一对应）。
        total_slots_ = static_cast<int>(registered_iovecs_.size());
        // 分配每个 slot 的占用位数组（0=free, 1=in_use）。
        slot_in_use_ = std::make_unique<std::atomic<std::uint8_t>[]>(
            static_cast<std::size_t>(total_slots_));
        // 初始化全部 slot 占用位为 0（空闲）。
        for (int i = 0; i < total_slots_; ++i) {
            slot_in_use_[static_cast<std::size_t>(i)].store(
                0u, std::memory_order_relaxed);
        }

        // 预留空间，避免 push_back 扩容；大小等于 bucket 数量。
        bucket_ranges_flat_.reserve(bucket_ranges_.size());
        // bidx: bucket 的扁平下标（给 bucket_index_ 和 bucket_probe_cursors_ 使用）。
        std::size_t bidx = 0;
        // 遍历有序 map，把“按 blocks 键值”结构转成“按下标”结构。
        for (const auto& kv : bucket_ranges_) {
            // 建立 O(1) 索引：blocks -> 扁平 bucket 下标。
            bucket_index_[kv.first] = bidx++;
            // 保存 bucket 区间到 vector，后续 acquire_free_slot 可直接按 bid 访问。
            bucket_ranges_flat_.push_back(kv.second);
        }
        // 为每个 bucket 分配一个原子 probe cursor（用于轮询起点，降低热点竞争）。
        bucket_probe_cursors_ = std::make_unique<std::atomic<std::uint32_t>[]>(
            bucket_ranges_flat_.size());
        // 所有 bucket 的 probe cursor 初值设为 0。
        for (std::size_t i = 0; i < bucket_ranges_flat_.size(); ++i) {
            bucket_probe_cursors_[i].store(0u, std::memory_order_relaxed);
        }

        // 一次性把扁平 iovec 数组注册到当前 ring，供 read_fixed 使用 slot_id 直接索引。
        int ret = io_uring_register_buffers(
            &ring_,
            registered_iovecs_.data(),
            static_cast<unsigned>(registered_iovecs_.size()));
        // ret!=0 代表注册失败；抛 system_error 交给上层处理。
        if (ret != 0) {
            if (GlobalConfig::instance().logger) {
                auto logger = GlobalConfig::instance().logger;
                logger->error("[IOU] io_uring_register_buffers failed ret={} errno={} slots={}",
                              ret,
                              errno,
                              registered_iovecs_.size());
            }
            throw std::system_error(errno, std::generic_category(),
                                    "io_uring_register_buffers failed");
        }
    }

    // ================================================================================================
    // 注册文件路径
    // ================================================================================================
    // 作用：注册设备文件 fd 到 io_uring。
    void registerDevpath(const std::vector<std::string>& devpaths)
    {
        int n = devpaths.size();

        fds_.resize(n * 2);

        for (int i = 0; i < n; i++)
        {
            int fdr = open(devpaths[i].c_str(), O_RDONLY | O_DIRECT);
            int fdw = open(devpaths[i].c_str(), O_WRONLY | O_DIRECT);

            if (fdr < 0 || fdw < 0) {
                if (GlobalConfig::instance().logger) {
                    auto logger = GlobalConfig::instance().logger;
                    logger->error("[IOU] open device failed path={} fdr={} fdw={} errno={}",
                                  devpaths[i],
                                  fdr,
                                  fdw,
                                  errno);
                }
                throw std::system_error(errno, std::generic_category(),
                                        "open device failed");
            }

            fds_[2*i] = fdr;
            fds_[2*i + 1] = fdw;
        }

        if (io_uring_register_files(&ring_, fds_.data(), fds_.size()) < 0) {
            if (GlobalConfig::instance().logger) {
                auto logger = GlobalConfig::instance().logger;
                logger->error("[IOU] io_uring_register_files failed errno={} fds={}",
                              errno,
                              fds_.size());
            }
            throw std::system_error(errno, std::generic_category(),
                                    "io_uring_register_files failed");
        }
    }

    // ================================================================================================
    // slot / 完成事件管理
    // ================================================================================================
    // 作用：获取空闲 fixed-buffer slot。
    // 策略：按 req_blocks 精确命中对应 bucket，在该 bucket 内线性找空槽并标记 in_use。
    int acquire_free_slot(std::uint32_t req_blocks = 1u)
    {
        if (!slot_in_use_) {
            return -1;
        }
        auto idx_it = bucket_index_.find(static_cast<int>(req_blocks));
        if (idx_it == bucket_index_.end()) {
            return -1;
        }
        const std::size_t bid = idx_it->second;
        if (bid >= bucket_ranges_flat_.size()) {
            return -1;
        }
        const BucketRange range = bucket_ranges_flat_[bid];
        if (range.end <= range.begin) {
            return -1;
        }
        const std::size_t len = range.end - range.begin;
        const std::uint32_t start = bucket_probe_cursors_[bid].fetch_add(
            1u, std::memory_order_relaxed);
        for (std::size_t probe = 0; probe < len; ++probe) {
            const std::size_t offset = (static_cast<std::size_t>(start) + probe) % len;
            const std::size_t sid = range.begin + offset;
            std::uint8_t expected = 0u;
            if (slot_in_use_[sid].compare_exchange_strong(
                    expected,
                    static_cast<std::uint8_t>(1u),
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return static_cast<int>(sid);
            }
        }
        return -1;
    }

    int try_acquire_best_available_slot(std::uint32_t remaining_blocks,
                                        std::uint32_t& chosen_blocks)
    {
        chosen_blocks = 0;
        if (!slot_in_use_ || bucket_ranges_.empty()) {
            return -1;
        }
        remaining_blocks = std::max<std::uint32_t>(1u, remaining_blocks);

        for (auto it = bucket_ranges_.rbegin(); it != bucket_ranges_.rend(); ++it) {
            if (it->first <= 0) {
                continue;
            }
            const auto blocks = static_cast<std::uint32_t>(it->first);
            if (blocks > remaining_blocks) {
                continue;
            }
            int slot_id = acquire_free_slot(blocks);
            if (slot_id >= 0) {
                chosen_blocks = blocks;
                return slot_id;
            }
        }

        for (const auto& kv : bucket_ranges_) {
            if (kv.first <= 0) {
                continue;
            }
            const auto blocks = static_cast<std::uint32_t>(kv.first);
            if (blocks < remaining_blocks) {
                continue;
            }
            int slot_id = acquire_free_slot(blocks);
            if (slot_id >= 0) {
                chosen_blocks = blocks;
                return slot_id;
            }
        }
        return -1;
    }

    int acquire_best_available_slot_wait(std::uint32_t remaining_blocks,
                                         std::uint32_t& chosen_blocks)
    {
        if (!slot_in_use_ || bucket_ranges_.empty()) {
            throw std::runtime_error(
                "IOuringManager: no registered buffer buckets");
        }
        remaining_blocks = std::max<std::uint32_t>(1u, remaining_blocks);
        while (true) {
            int slot_id = try_acquire_best_available_slot(
                remaining_blocks,
                chosen_blocks);
            if (slot_id >= 0) {
                return slot_id;
            }
            int reclaimed = reclaim_completions(sync_reclaim_batch_);
            if (reclaimed <= 0) {
                io_uring_submit(&ring_);
                wait_one_cqe_for_reclaim();
            }
        }
    }

    annfs::io::BlockReq make_req_chunk(const annfs::io::BlockReq& src,
                                       off_t chunk_base,
                                       std::uint32_t chunk_blocks) const
    {
        annfs::io::BlockReq chunk;
        chunk.base = chunk_base;
        chunk.blocks = std::max<std::uint32_t>(1u, chunk_blocks);
        if (src.parse_hints.empty() || block_size_ == 0) {
            return chunk;
        }

        const std::uint64_t src_base = static_cast<std::uint64_t>(src.base);
        const std::uint64_t chunk_start = static_cast<std::uint64_t>(chunk_base);
        const std::uint64_t chunk_len =
            static_cast<std::uint64_t>(chunk.blocks) *
            static_cast<std::uint64_t>(block_size_);
        const std::uint64_t chunk_end = chunk_start + chunk_len;
        chunk.parse_hints.reserve(src.parse_hints.size());
        for (const auto& h : src.parse_hints) {
            const std::uint64_t abs_start =
                src_base + static_cast<std::uint64_t>(h.start_in_req);
            if (abs_start < chunk_start || abs_start >= chunk_end) {
                continue;
            }
            auto nh = h;
            nh.start_in_req = static_cast<std::uint32_t>(abs_start - chunk_start);
            chunk.parse_hints.push_back(nh);
        }
        return chunk;
    }

    void account_submitted_io(long& preaccounted_budget)
    {
        if (preaccounted_budget > 0) {
            --preaccounted_budget;
            if (io_debug_diagnostics_enabled_) {
                submitted_total_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        if (io_debug_diagnostics_enabled_) {
            submitted_total_.fetch_add(1, std::memory_order_relaxed);
        }
        outstanding_ios_.fetch_add(1, std::memory_order_relaxed);
    }

    void release_unused_preaccount(long preaccounted_budget)
    {
        if (preaccounted_budget <= 0) {
            return;
        }
        outstanding_ios_.fetch_sub(preaccounted_budget, std::memory_order_relaxed);
        completion_cv_.notify_all();
    }

    long clamp_preaccount_count(unsigned long long count) const
    {
        const auto max_long =
            static_cast<unsigned long long>(
                std::min<long>(
                    std::numeric_limits<long>::max(),
                    static_cast<long>(std::numeric_limits<int>::max())));
        if (count > max_long) {
            return static_cast<long>(max_long);
        }
        return static_cast<long>(count);
    }

    bool async_has_pending_work() const
    {
        if (async_active_tasks_.load(std::memory_order_acquire) > 0) {
            return true;
        }
        return task_queue_ && !task_queue_->empty();
    }

    io_uring_sqe* get_sqe_wait()
    {
        while (true) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (sqe) {
                return sqe;
            }
            io_uring_submit(&ring_);
            wait_one_cqe_for_reclaim();
        }
    }

    // 作用：global slot_id 合法性校验（slot_id 即一维数组下标）。
    bool resolve_slot_id(int slot_id) const
    {
        if (slot_id < 0 || slot_id >= static_cast<int>(slots_.size()) ||
            slot_id >= static_cast<int>(registered_iovecs_.size())) {
            return false;
        }
        return true;
    }

    // 作用：slot_id 解析后返回对应 slot 元数据指针；失败返回 nullptr。
    Slot* slot_by_global_id(int slot_id)
    {
        if (!resolve_slot_id(slot_id)) {
            return nullptr;
        }
        return &slots_[slot_id];
    }

    // 作用：slot_id 解析后返回对应注册 iovec 指针；失败返回 nullptr。
    iovec* iovec_by_global_id(int slot_id)
    {
        if (!resolve_slot_id(slot_id)) {
            return nullptr;
        }
        return &registered_iovecs_[slot_id];
    }

    // 作用：释放 fixed-buffer slot（清理请求上下文）。
    void release_slot(int idx)
    {
        std::lock_guard<std::mutex> lk(slot_mutex_);
        Slot* slot = slot_by_global_id(idx);
        if (!slot) {
            return;
        }
        slot->query_id = -1;
        slot->req_base = 0;
        slot->req_blocks = 1;
        slot->off_list.clear();
        slot->file_fd = -1;
        slot->file_fd_state.reset();
        if (idx >= 0 && slot_in_use_ &&
            static_cast<std::size_t>(idx) < static_cast<std::size_t>(total_slots_)) {
            slot_in_use_[static_cast<std::size_t>(idx)].store(
                0u, std::memory_order_release);
        }
    }

    // 作用：提交前绑定 slot 的请求上下文（query/base/blocks/hints）。
    bool prepare_slot_for_req(int slot_id,
                              int query_id,
                              const annfs::io::BlockReq& req)
    {
        std::lock_guard<std::mutex> lk(slot_mutex_);
        Slot* slot = slot_by_global_id(slot_id);
        if (!slot || !slot_in_use_ ||
            static_cast<std::size_t>(slot_id) >= static_cast<std::size_t>(total_slots_) ||
            slot_in_use_[static_cast<std::size_t>(slot_id)].load(
                std::memory_order_acquire) == 0u) {
            return false;
        }
        slot->offset = req.base;
        bind_slot_req_context(*slot, query_id, req);
        return true;
    }

    // 作用：文件路径读场景绑定 query_id 和共享 fd 状态。
    bool prepare_slot_for_file(int slot_id,
                               int query_id,
                               const std::shared_ptr<FileFdState>& fd_state)
    {
        std::lock_guard<std::mutex> lk(slot_mutex_);
        Slot* slot = slot_by_global_id(slot_id);
        if (!slot || !slot_in_use_ ||
            static_cast<std::size_t>(slot_id) >= static_cast<std::size_t>(total_slots_) ||
            slot_in_use_[static_cast<std::size_t>(slot_id)].load(
                std::memory_order_acquire) == 0u) {
            return false;
        }
        slot->query_id = query_id;
        slot->file_fd = fd_state ? fd_state->fd : -1;
        slot->file_fd_state = fd_state;
        return true;
    }

    struct ReleasedSlotInfo {
        int query_id = -1;
        int file_fd = -1;
        std::shared_ptr<FileFdState> file_fd_state;
    };

    // 作用：完成时回收 slot，并返回被回收 slot 的 query_id/file_fd（用于事件上报和 fd 释放）。
    ReleasedSlotInfo release_slot_and_get_info(int slot_id)
    {
        std::lock_guard<std::mutex> lk(slot_mutex_);
        Slot* slot = slot_by_global_id(slot_id);
        if (!slot) {
            return {};
        }
        ReleasedSlotInfo info;
        info.query_id = slot->query_id;
        info.file_fd = slot->file_fd;
        info.file_fd_state = std::move(slot->file_fd_state);
        slot->query_id = -1;
        slot->req_base = 0;
        slot->req_blocks = 1;
        slot->off_list.clear();
        slot->file_fd = -1;
        slot->file_fd_state.reset();
        if (slot_in_use_ &&
            static_cast<std::size_t>(slot_id) < static_cast<std::size_t>(total_slots_)) {
            slot_in_use_[static_cast<std::size_t>(slot_id)].store(
                0u, std::memory_order_release);
        }
        return info;
    }

    void bind_slot_req_context(Slot& slot,
                               int query_id,
                               const annfs::io::BlockReq& req)
    {
        // 槽位运行态只保留“本次请求最小必需元数据”：
        // - query_id: completion 后把完成数回传给哪个 query
        // - req_base/req_blocks: 本次读请求覆盖的块区间
        // - off_list: 读回 buffer 内，每条记录的相对起始位置（相对 req_base）
        slot.query_id = query_id;
        slot.req_base = req.base;
        slot.req_blocks = std::max<std::uint32_t>(1u, req.blocks);
        slot.off_list.clear();
        slot.file_fd = -1;
        slot.file_fd_state.reset();
        if (req.parse_hints.empty()) {
            return;
        }
        const std::uint64_t req_len =
            static_cast<std::uint64_t>(slot.req_blocks) * block_size_;
        slot.off_list.reserve(req.parse_hints.size());
        for (const auto& h : req.parse_hints) {
            // 防御：hint 越界则忽略，避免后续解析访问越界。
            if (static_cast<std::uint64_t>(h.start_in_req) >= req_len) {
                continue;
            }
            slot.off_list.push_back(h.start_in_req);
        }
    }

    // max_to_reclaim < 0 表示不限
    // 作用：回收 CQE（同步/异步通用）。
    int reclaim_completions(int max_to_reclaim)
    {
        return reclaim_completions_locked(max_to_reclaim);
    }

    // 作用：同步模式下阻塞等待完成。
    void wait_all_completions_sync(int pending_ios, int debug_query_id = -1)
    {
        const int original_pending = pending_ios;
        while (pending_ios > 0) {
            io_uring_cqe* cqe = nullptr;
            int ret = wait_cqe_timeout(&cqe);
            if (ret == -ETIME) {
                debug_query_io("sync_wait_timeout",
                               debug_query_id,
                               pending_ios,
                               outstanding_ios_.load(std::memory_order_relaxed),
                               original_pending);
                continue;
            }
            if (ret < 0) {
                debug_query_io("sync_wait_error",
                               debug_query_id,
                               ret,
                               outstanding_ios_.load(std::memory_order_relaxed),
                               pending_ios);
                break;
            }
            if (cqe) {
                handle_completion_data(io_uring_cqe_get_data64(cqe));
                pending_ios--;
                io_uring_cqe_seen(&ring_, cqe);
                debug_query_io("sync_wait_cqe",
                               debug_query_id,
                               pending_ios,
                               outstanding_ios_.load(std::memory_order_relaxed),
                               original_pending);
            }
        }
        reclaim_completions(-1);
    }

    // 作用：同步提交 req（同步/异步通用的底层提交）。
    int submit_reqs_sync(const std::vector<annfs::io::BlockReq>& req_list,
                         int query_id,
                         long preaccounted_budget = 0)
    {
        if (req_list.empty()) {
            return 0;
        }

        int fd = fds_[0];
        unsigned pending_submit = 0;
        unsigned total_submitted = 0;
        int submitted_for_query = 0;
        int dropped = 0;
        debug_query_io("submit_reqs_sync_begin",
                       query_id,
                       req_list.size(),
                       outstanding_ios_.load(std::memory_order_relaxed));

        for (const auto& req : req_list)
        {
            const std::uint32_t requested_blocks =
                std::max<std::uint32_t>(1u, req.blocks);
            std::uint32_t remaining_blocks = requested_blocks;
            off_t chunk_base = req.base;
            while (remaining_blocks > 0) {
                std::uint32_t req_blocks = remaining_blocks;
                int slot_id = acquire_best_available_slot_wait(
                    remaining_blocks,
                    req_blocks);

                annfs::io::BlockReq req_chunk =
                    make_req_chunk(req, chunk_base, req_blocks);
                req_chunk.base = chunk_base;
                req_chunk.blocks = req_blocks;
                if (!prepare_slot_for_req(slot_id, query_id, req_chunk)) {
                    release_slot(slot_id);
                    throw std::runtime_error("IOuringManager: failed to bind req slot context");
                }

                io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
                if (!sqe && pending_submit > 0) {
                    io_uring_submit(&ring_);
                    pending_submit = 0;
                }
                if (!sqe) {
                    sqe = get_sqe_wait();
                }

                iovec* iv = iovec_by_global_id(slot_id);
                if (!iv) {
                    release_slot(slot_id);
                    throw std::runtime_error("IOuringManager: invalid iovec slot for req read");
                }
                void* buf = iv->iov_base;
                size_t len =
                    static_cast<size_t>(req_blocks) * static_cast<size_t>(block_size_);

                io_uring_prep_read_fixed(
                    sqe,
                    fd,
                    buf,
                    len,
                    chunk_base,
                    static_cast<unsigned>(slot_id)
                );
                // user_data 只塞 slot_id：
                // completion 时 O(1) 反查槽位并回收 buffer，不再维护 token->req map。
                io_uring_sqe_set_data64(
                    sqe,
                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(slot_id)));
                account_submitted_io(preaccounted_budget);
                ++pending_submit;
                ++total_submitted;
                ++submitted_for_query;

                const std::uint32_t consumed = std::min(req_blocks, remaining_blocks);
                remaining_blocks -= consumed;
                chunk_base += static_cast<off_t>(
                    static_cast<std::uint64_t>(consumed) *
                    static_cast<std::uint64_t>(block_size_));
            }
        }

        if (pending_submit > 0) {
            io_uring_submit(&ring_);
        }
        debug_query_io("submit_reqs_sync_submitted",
                       query_id,
                       total_submitted,
                       outstanding_ios_.load(std::memory_order_relaxed),
                       submitted_for_query);

        if (submitted_for_query > 0) {
            notify_query_io_submitted(query_id, submitted_for_query);
        }

        if (dropped > 0) {
            outstanding_ios_.fetch_sub(dropped, std::memory_order_relaxed);
            completion_cv_.notify_all();
        }
        release_unused_preaccount(preaccounted_budget);
        debug_query_io("submit_reqs_sync_done",
                       query_id,
                       total_submitted,
                       outstanding_ios_.load(std::memory_order_relaxed),
                       dropped);
        return static_cast<int>(total_submitted);
    }

    // 作用：同步 offset 提交入口（先 decode/build req，再复用 req 提交路径）。
    int submit_offsets_sync(const std::vector<off_t>& offset_list,
                            int query_id)
    {
        auto reqs = build_reqs_from_offsets(offset_list);
        return submit_reqs_sync(reqs, query_id);
    }

    std::uint32_t choose_file_req_blocks(std::uint64_t remaining_blocks) const
    {
        if (remaining_blocks <= 1) {
            return 1u;
        }
        const auto& gc = GlobalConfig::instance();
        if (gc.MAPPED_SPAN_MODE == "split") {
            return 1u;
        }

        std::uint32_t max_bucket = 0;
        std::uint32_t ceil_bucket = 0;
        for (const auto& kv : bucket_ranges_) {
            if (kv.first <= 0) {
                continue;
            }
            const auto blocks = static_cast<std::uint32_t>(kv.first);
            max_bucket = std::max(max_bucket, blocks);
            if (blocks >= remaining_blocks &&
                (ceil_bucket == 0 || blocks < ceil_bucket)) {
                ceil_bucket = blocks;
            }
        }
        if (ceil_bucket > 0) {
            return ceil_bucket;
        }
        return max_bucket > 0 ? max_bucket : 1u;
    }

    std::vector<FileReadReq> build_file_read_reqs(
        const std::vector<std::string>& file_paths,
        int query_id = -1) const
    {
        std::vector<FileReadReq> reqs;
        if (file_paths.empty() || block_size_ == 0) {
            return reqs;
        }

        std::size_t stat_failed = 0;
        std::size_t zero_size = 0;
        std::size_t open_failed = 0;
        std::size_t readable_files = 0;
        int first_stat_errno = 0;
        int first_open_errno = 0;
        std::string first_stat_failed_path;
        std::string first_zero_size_path;
        std::string first_open_failed_path;

        for (const auto& path : file_paths) {
            struct stat st {};
            if (::stat(path.c_str(), &st) != 0) {
                ++stat_failed;
                if (first_stat_failed_path.empty()) {
                    first_stat_failed_path = path;
                    first_stat_errno = errno;
                }
                continue;
            }
            if (st.st_size <= 0) {
                ++zero_size;
                if (first_zero_size_path.empty()) {
                    first_zero_size_path = path;
                }
                continue;
            }
            std::uint64_t remaining_blocks =
                (static_cast<std::uint64_t>(st.st_size) +
                 static_cast<std::uint64_t>(block_size_) - 1u) /
                static_cast<std::uint64_t>(block_size_);
            std::vector<FileReadReq> path_reqs;
            off_t offset = 0;
            while (remaining_blocks > 0) {
                std::uint32_t blocks = choose_file_req_blocks(remaining_blocks);
                blocks = std::max<std::uint32_t>(1u, blocks);
                path_reqs.push_back(FileReadReq{path, offset, blocks, {}});
                offset += static_cast<off_t>(
                    static_cast<std::uint64_t>(blocks) *
                    static_cast<std::uint64_t>(block_size_));
                remaining_blocks -= std::min<std::uint64_t>(
                    remaining_blocks,
                    static_cast<std::uint64_t>(blocks));
            }
            if (path_reqs.empty()) {
                continue;
            }
            int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) {
                ++open_failed;
                if (first_open_failed_path.empty()) {
                    first_open_failed_path = path;
                    first_open_errno = errno;
                }
                continue;
            }
            ++readable_files;
            auto fd_state = std::make_shared<FileFdState>();
            fd_state->fd = fd;
            fd_state->remaining.store(
                static_cast<int>(path_reqs.size()),
                std::memory_order_relaxed);
            for (auto& req : path_reqs) {
                req.fd_state = fd_state;
                reqs.push_back(std::move(req));
            }
        }
        auto& gc = GlobalConfig::instance();
        if (gc.logger && gc.logger->should_log(spdlog::level::debug)) {
            gc.logger->debug(
                "[IOU_QUERY] stage=submit_files_build_summary query_id={} paths={} reqs={} readable_files={} stat_failed={} zero_size={} open_failed={} first_stat_errno={} first_open_errno={} first_stat_failed_path={} first_zero_size_path={} first_open_failed_path={}",
                query_id,
                file_paths.size(),
                reqs.size(),
                readable_files,
                stat_failed,
                zero_size,
                open_failed,
                first_stat_errno,
                first_open_errno,
                first_stat_failed_path,
                first_zero_size_path,
                first_open_failed_path);
        }
        return reqs;
    }

    void release_file_fd_ref(const std::shared_ptr<FileFdState>& fd_state)
    {
        if (!fd_state) {
            return;
        }
        const int prev = fd_state->remaining.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1 && fd_state->fd >= 0) {
            close(fd_state->fd);
            fd_state->fd = -1;
        }
    }

    // 作用：同步文件路径提交入口（FullFS：按文件大小拆成 1/multi-block 槽，提交 read + completion 回收）。
    int submit_file_reqs_sync(const std::vector<FileReadReq>& file_reqs,
                              int query_id = -1,
                              long preaccounted_budget = 0)
    {
        if (file_reqs.empty()) {
            return 0;
        }
        if (total_slots_ <= 0) {
            throw std::runtime_error("IOuringManager: no registered buffers for file read.");
        }

        unsigned pending_submit = 0;
        unsigned total_submitted = 0;
        unsigned dropped = 0;
        debug_query_io("submit_file_reqs_sync_begin",
                       query_id,
                       file_reqs.size(),
                       outstanding_ios_.load(std::memory_order_relaxed));

        for (const auto& req : file_reqs) {
            std::uint32_t remaining_blocks = std::max<std::uint32_t>(1u, req.blocks);
            off_t chunk_offset = req.offset;
            bool first_chunk_for_req = true;
            while (remaining_blocks > 0) {
                if (!req.fd_state || req.fd_state->fd < 0) {
                    throw std::runtime_error("IOuringManager: invalid shared file fd state");
                }
                if (!first_chunk_for_req) {
                    req.fd_state->remaining.fetch_add(1, std::memory_order_relaxed);
                }

                std::uint32_t req_blocks = remaining_blocks;
                int slot_id = acquire_best_available_slot_wait(
                    remaining_blocks,
                    req_blocks);

                if (!prepare_slot_for_file(slot_id, query_id, req.fd_state)) {
                    release_slot(slot_id);
                    release_file_fd_ref(req.fd_state);
                    throw std::runtime_error("IOuringManager: failed to bind file slot context");
                }

                io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
                if (!sqe && pending_submit > 0) {
                    io_uring_submit(&ring_);
                    pending_submit = 0;
                }
                if (!sqe) {
                    sqe = get_sqe_wait();
                }

                iovec* iv = iovec_by_global_id(slot_id);
                if (!iv) {
                    const ReleasedSlotInfo info = release_slot_and_get_info(slot_id);
                    release_file_fd_ref(info.file_fd_state);
                    throw std::runtime_error("IOuringManager: invalid iovec slot for file read");
                }
                void* buf = iv->iov_base;
                size_t len =
                    static_cast<size_t>(req_blocks) * static_cast<size_t>(block_size_);
                io_uring_prep_read(sqe, req.fd_state->fd, buf, len, chunk_offset);

                std::uint64_t data = GlobalConfig::kFileReadTag |
                                     static_cast<std::uint32_t>(slot_id);
                io_uring_sqe_set_data64(sqe, data);
                account_submitted_io(preaccounted_budget);
                ++pending_submit;
                ++total_submitted;

                const std::uint32_t consumed = std::min(req_blocks, remaining_blocks);
                remaining_blocks -= consumed;
                chunk_offset += static_cast<off_t>(
                    static_cast<std::uint64_t>(consumed) *
                    static_cast<std::uint64_t>(block_size_));
                first_chunk_for_req = false;
            }
        }

        if (pending_submit > 0) {
            io_uring_submit(&ring_);
        }
        debug_query_io("submit_file_reqs_sync_submitted",
                       query_id,
                       total_submitted,
                       outstanding_ios_.load(std::memory_order_relaxed));

        if (dropped > 0) {
            outstanding_ios_.fetch_sub(
                static_cast<long>(dropped),
                std::memory_order_relaxed);
            completion_cv_.notify_all();
        }
        release_unused_preaccount(preaccounted_budget);
        if (total_submitted > 0) {
            notify_query_io_submitted(query_id, static_cast<int>(total_submitted));
        }
        debug_query_io("submit_file_reqs_sync_done",
                       query_id,
                       total_submitted,
                       outstanding_ios_.load(std::memory_order_relaxed),
                       dropped);
        return static_cast<int>(total_submitted);
    }

    // 作用：按当前并发模式懒创建异步任务队列（SPSC/MPSC）。
    void ensure_task_queue()
    {
        if (task_queue_) {
            return;
        }
        if (async_queue_mode_ == AsyncQueueMode::Mpsc) {
            task_queue_ = std::make_unique<MpscQueue>(async_queue_capacity_);
        } else {
            task_queue_ = std::make_unique<SpscQueue>(async_queue_capacity_);
        }
    }

    // 作用：异步模式下将任务入队（带退避重试）；返回“逻辑提交数”。
    int enqueue_async_task(const AsyncTask& task_proto, std::size_t logical_count)
    {
        ensure_task_queue();
        while (true) {
            AsyncTask task = task_proto;
            if (task_queue_->enqueue(std::move(task))) {
                break;
            }
            if (async_poll_us_ > 0) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(async_poll_us_));
            } else {
                std::this_thread::yield();
            }
        }
        return static_cast<int>(logical_count);
    }

    // 作用：overlap 通用提交入口：
    // - 异步模式：入队 AsyncTask；
    // - 同步模式：直接执行 sync_submit()。
    template <typename SyncSubmitFn>
    int submit_or_enqueue_overlap(std::size_t logical_count,
                                  const AsyncTask& task,
                                  SyncSubmitFn&& sync_submit)
    {
        if (use_async_submit_) {
            outstanding_ios_.fetch_add(
                static_cast<long>(logical_count),
                std::memory_order_relaxed);
            return enqueue_async_task(task, logical_count);
        }
        return sync_submit();
    }

    // 作用：同步提交 + 阻塞等待的通用入口。
    // use_submit_and_wait=true 使用 submit_and_wait 驱动等待；
    // 否则使用 wait_cqe 驱动等待（供 batch_read_reqs 复用旧语义）。
    template <typename SyncSubmitFn>
    int submit_and_wait_common(std::size_t logical_count,
                               SyncSubmitFn&& sync_submit,
                               bool use_submit_and_wait = true,
                               int debug_query_id = -1)
    {
        const long outstanding_baseline =
            outstanding_ios_.load(std::memory_order_relaxed);
        debug_query_io("submit_wait_begin",
                       debug_query_id,
                       logical_count,
                       outstanding_baseline,
                       use_submit_and_wait ? 1 : 0);
        int submitted = sync_submit();
        const long outstanding_after_submit =
            outstanding_ios_.load(std::memory_order_relaxed);
        debug_query_io("submit_wait_after_submit",
                       debug_query_id,
                       submitted,
                       outstanding_after_submit,
                       static_cast<long long>(logical_count));
        if (submitted <= 0) {
            debug_query_io("submit_wait_no_submit",
                           debug_query_id,
                           submitted,
                           outstanding_after_submit);
            return submitted;
        }
        const int pending_to_wait = static_cast<int>(
            std::max<long>(0, outstanding_after_submit - outstanding_baseline));
        if (pending_to_wait > 0) {
            if (io_debug_diagnostics_enabled_) {
                wait_calls_total_.fetch_add(1, std::memory_order_relaxed);
                wait_requested_total_.fetch_add(
                    static_cast<std::uint64_t>(pending_to_wait),
                    std::memory_order_relaxed);
            }
            debug_query_io("submit_wait_wait_begin",
                           debug_query_id,
                           pending_to_wait,
                           outstanding_ios_.load(std::memory_order_relaxed));
            if (use_submit_and_wait) {
                wait_all_completions_submit_and_wait(pending_to_wait, debug_query_id);
            } else {
                wait_all_completions_sync(pending_to_wait, debug_query_id);
            }
            debug_query_io("submit_wait_wait_done",
                           debug_query_id,
                           pending_to_wait,
                           outstanding_ios_.load(std::memory_order_relaxed));
        }
        debug_query_io("submit_wait_done",
                       debug_query_id,
                       submitted,
                       outstanding_ios_.load(std::memory_order_relaxed));
        return submitted;
    }

    // 作用：labels -> reqs 的公共构建入口（同步/异步路径复用）。
    std::vector<annfs::io::BlockReq> build_reqs_from_labels(
        const std::vector<unsigned long long>& labels,
        const ParamConfig& pc,
        std::size_t* missing,
        std::size_t* issued_io,
        int query_id = -1) const
    {
        auto plan = annfs::io::BuildReqPlanU64(labels, pc);
        if (missing) {
            *missing = plan.missing;
        }
        auto& gc = GlobalConfig::instance();
        if (gc.logger && gc.logger->should_log(spdlog::level::debug)) {
            gc.logger->debug(
                "[IOU_QUERY] stage=mapped_req_plan_summary query_id={} labels={} raw_reqs={} normalized_reqs={} bucketized_reqs={} missing={} dropped_unsupported={} max_blocks_raw={} max_blocks_normalized={} use_label_map={} label_map_fallback={}",
                query_id,
                plan.input_labels,
                plan.raw_reqs,
                plan.normalized_reqs,
                plan.total_reqs,
                plan.missing,
                plan.dropped_unsupported,
                plan.max_blocks_raw,
                plan.max_blocks_normalized,
                plan.used_label_map ? "true" : "false",
                plan.label_map_fallback ? "true" : "false");
        }
        std::vector<annfs::io::BlockReq> reqs;
        reqs.reserve(plan.total_reqs);
        for (const auto& kv : plan.by_blocks) {
            for (const auto& r : kv.second) {
                reqs.push_back(r);
            }
        }
        if (issued_io) {
            *issued_io = reqs.size();
        }
        return reqs;
    }

    int wait_cqe_timeout(io_uring_cqe** cqe)
    {
        __kernel_timespec ts{};
        ts.tv_sec = 1;
        ts.tv_nsec = 0;
        return io_uring_wait_cqe_timeout(&ring_, cqe, &ts);
    }

    int wait_one_cqe_for_reclaim()
    {
        io_uring_cqe* cqe = nullptr;
        int ret = wait_cqe_timeout(&cqe);
        if (ret == 0 && cqe) {
            handle_completion_data(io_uring_cqe_get_data64(cqe));
            io_uring_cqe_seen(&ring_, cqe);
            return 1;
        }
        return 0;
    }

    // 作用：内部批量回收 CQE（peek_batch）。
    int reclaim_completions_locked(int max_to_reclaim)
    {
        int reclaimed = 0;
        const int batch_max = max_to_reclaim > 0 ? max_to_reclaim : 128;
        std::vector<io_uring_cqe*> cqes(batch_max);

        while (true) {
            int to_peek = batch_max;
            if (max_to_reclaim > 0) {
                to_peek = std::min(batch_max, max_to_reclaim - reclaimed);
                if (to_peek <= 0) {
                    break;
                }
            }
            int ret = io_uring_peek_batch_cqe(&ring_, cqes.data(), to_peek);
            if (ret <= 0) {
                break;
            }
            for (int i = 0; i < ret; ++i) {
                io_uring_cqe* cqe = cqes[i];
                handle_completion_data(io_uring_cqe_get_data64(cqe));
                io_uring_cqe_seen(&ring_, cqe);
                ++reclaimed;
            }
            if (max_to_reclaim > 0 && reclaimed >= max_to_reclaim) {
                break;
            }
        }
        return reclaimed;
    }

    // 作用：submit_and_wait 驱动的阻塞回收路径，直到 pending_ios 全部完成。
    void wait_all_completions_submit_and_wait(int pending_ios, int debug_query_id = -1)
    {
        const int original_pending = pending_ios;
        while (pending_ios > 0) {
            io_uring_submit(&ring_);
            int reclaimed = reclaim_completions_locked(
                std::min(sync_reclaim_batch_, pending_ios));
            if (reclaimed <= 0) {
                debug_query_io("submit_and_wait_wait_cqe",
                               debug_query_id,
                               pending_ios,
                               outstanding_ios_.load(std::memory_order_relaxed),
                               original_pending);
                reclaimed = wait_one_cqe_for_reclaim();
            }
            pending_ios -= std::max(0, reclaimed);
            debug_query_io("submit_and_wait_reclaimed",
                           debug_query_id,
                           reclaimed,
                           outstanding_ios_.load(std::memory_order_relaxed),
                           pending_ios);
        }
    }

    // 作用：解析 CQE user_data，回收 slot/fd，并更新 query 完成计数与 outstanding。
    void handle_completion_data(std::uint64_t data)
    {
        bool is_file_read = (data & GlobalConfig::kFileReadTag) != 0;
        int query_id = -1;
        if (is_file_read) {
            int slot_id = static_cast<int>(data & 0xFFFFFFFFu);
            const ReleasedSlotInfo info = release_slot_and_get_info(slot_id);
            query_id = info.query_id;
            release_file_fd_ref(info.file_fd_state);
        } else {
            // 普通 fixed-buffer 读路径：data 直接是 slot_id。
            int slot_id = static_cast<int>(data);
            query_id = release_slot_and_get_info(slot_id).query_id;
        }
        if (query_id >= 0) {
            notify_query_io_completed(query_id, 1);
        }
        if (io_debug_diagnostics_enabled_) {
            completed_total_.fetch_add(1, std::memory_order_relaxed);
        }
        outstanding_ios_.fetch_sub(1, std::memory_order_relaxed);
        completion_cv_.notify_all();
    }

    // 作用：异步线程主循环：出队 batch -> 提交 -> 回收 CQE。
    // 说明：
    // - 生产者线程只 enqueue（几乎不阻塞）；
    // - 消费者线程统一和 ring 交互，减少多线程同时 get_sqe/submit 的锁争用；
    // - 多对1模式下推荐 MPSC 队列，1对1模式可用 SPSC。
    void async_submit_loop()
    {
        // auto logger = GlobalConfig::instance().logger;
        // if (logger) {
        //     logger->debug("[IOU][async] submit thread start");
        // }
        while (true) {
            AsyncTask task;
            if (stop_async_submit_.load(std::memory_order_relaxed) &&
                (!task_queue_ || task_queue_->empty())) {
                // if (logger) {
                //     logger->debug("[IOU][async] submit thread stop");
                // }
                break;
            }
            if (!task_queue_ || !task_queue_->dequeue(task)) {
                long outstanding = outstanding_ios_.load(std::memory_order_relaxed);
                auto now = std::chrono::steady_clock::now();
                bool time_to_reclaim =
                    async_reclaim_interval_us_ >= 0 &&
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        now - last_reclaim_ts_).count() >= async_reclaim_interval_us_;
                if (outstanding >= async_reclaim_threshold_ || time_to_reclaim) {
                    int reclaimed = reclaim_completions(async_reclaim_batch_);
                    if (reclaimed > 0) {
                        last_reclaim_ts_ = now;
                        // if (logger) {
                        //     logger->debug("[IOU][async] reclaimed={} outstanding={}",
                        //                   reclaimed,
                        //                   outstanding_ios_.load(std::memory_order_relaxed));
                        // }
                    }
                }
                if (async_poll_us_ > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(async_poll_us_));
                } else {
                    std::this_thread::yield();
                }
                continue;
            }
            async_active_tasks_.fetch_add(1, std::memory_order_acq_rel);
            if (task.type == AsyncTask::Type::Reqs && !task.reqs.empty()) {
                debug_query_io("async_task_submit_reqs_begin",
                               task.query_id,
                               task.reqs.size(),
                               outstanding_ios_.load(std::memory_order_relaxed));
                submit_reqs_sync(
                    task.reqs,
                    task.query_id,
                    task.preaccounted_count);
                debug_query_io("async_task_submit_reqs_done",
                               task.query_id,
                               task.reqs.size(),
                               outstanding_ios_.load(std::memory_order_relaxed));
            } else if (task.type == AsyncTask::Type::Files && !task.file_reqs.empty()) {
                debug_query_io("async_task_submit_files_begin",
                               task.query_id,
                               task.file_reqs.size(),
                               outstanding_ios_.load(std::memory_order_relaxed));
                submit_file_reqs_sync(
                    task.file_reqs,
                    task.query_id,
                    task.preaccounted_count);
                debug_query_io("async_task_submit_files_done",
                               task.query_id,
                               task.file_reqs.size(),
                               outstanding_ios_.load(std::memory_order_relaxed));
                // if (logger) {
                //     logger->debug("[IOU][async] submit file batch size={} outstanding={}",
                //                   task.file_reqs.size(),
                //                   outstanding_ios_.load(std::memory_order_relaxed));
                // }
            }
            async_active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
            completion_cv_.notify_all();
            if (async_use_submit_and_wait_ &&
                outstanding_ios_.load(std::memory_order_relaxed) > 0) {
                io_uring_submit_and_wait(&ring_, sync_submit_wait_nr_);
            }
            // int reclaimed = reclaim_completions(async_reclaim_batch_);
            // if (logger && reclaimed > 0) {
            //     logger->debug("[IOU][async] reclaimed={} outstanding={}",
            //                   reclaimed,
            //                   outstanding_ios_.load(std::memory_order_relaxed));
            // }
        }
        reclaim_completions(-1);
        // if (logger) {
        //     logger->debug("[IOU][async] submit thread exit");
        // }
    }

    // 作用：把 offset_list 统一转换成“可提交的 BlockReq 列表”。
    // - SHIFT_OFF_BITS: raw 当 label-id，转成 (raw << OFF_BITS_LEN, blocks=1)
    // - PACKED_MAP:
    //   split=true  -> 拆成多个 1-block 请求；
    //   split=false -> 保留连续块请求，但上限裁剪到 max_registered_blocks_。
    // 备注：这保证了“请求长度永不超过已注册 buffer 最大块数”。
    std::vector<annfs::io::BlockReq> build_reqs_from_offsets(
        const std::vector<off_t>& offset_list) const
    {
        std::vector<annfs::io::BlockReq> reqs;
        if (offset_list.empty()) {
            return reqs;
        }
        const auto& gc = GlobalConfig::instance();
        const bool split = (gc.MAPPED_SPAN_MODE == "split");
        reqs.reserve(offset_list.size());
        for (off_t raw_off : offset_list) {
            const std::uint64_t raw = static_cast<std::uint64_t>(raw_off);
            if (gc.LABEL_DECODE_MODE_CFG ==
                GlobalConfig::LABEL_DECODE_MODE::SHIFT_OFF_BITS) {
                reqs.push_back(annfs::io::BlockReq{
                    static_cast<off_t>(raw << gc.OFF_BITS_LEN),
                    1u,
                    {}});
                continue;
            }
            const auto d = annfs::io::DecodePacked64(
                raw, static_cast<std::uint32_t>(block_size_));
            std::uint64_t remaining_blocks =
                std::max<std::uint32_t>(1u, d.blocks_to_read);
            std::uint64_t chunk_base = d.block_base;
            while (remaining_blocks > 0) {
                std::uint32_t blocks = split
                    ? 1u
                    : choose_file_req_blocks(remaining_blocks);
                blocks = std::max<std::uint32_t>(1u, blocks);
                reqs.push_back(annfs::io::BlockReq{
                    static_cast<off_t>(chunk_base),
                    blocks,
                    {}});
                chunk_base += static_cast<std::uint64_t>(blocks) * block_size_;
                remaining_blocks -= std::min<std::uint64_t>(
                    remaining_blocks,
                    static_cast<std::uint64_t>(blocks));
            }
        }
        return reqs;
    }

    void debug_query_io(const char* stage,
                        int query_id,
                        long long count,
                        long long outstanding,
                        long long detail = -1) const
    {
        auto logger = GlobalConfig::instance().logger;
        if (!io_debug_diagnostics_enabled_ ||
            !logger ||
            !logger->should_log(spdlog::level::debug)) {
            return;
        }
        static std::mutex debug_mu;
        std::lock_guard<std::mutex> lk(debug_mu);
        logger->debug(
            "[IOU_QUERY] stage={} query_id={} count={} outstanding={} detail={} use_async={} ring_depth={}",
            stage,
            query_id,
            count,
            outstanding,
            detail,
            use_async_submit_ ? "true" : "false",
            io_depth_);
    }

    // 作用：向监听器上报“query 已提交 IO 数量”事件。
    void notify_query_io_submitted(int query_id, int count)
    {
        if (!query_listener_ || query_id < 0 || count <= 0) {
            return;
        }
        query_listener_->OnQueryIoSubmitted(query_id, count);
    }

    // 作用：向监听器上报“query 已完成 IO 数量”事件。
    void notify_query_io_completed(int query_id, int count)
    {
        if (!query_listener_ || query_id < 0 || count <= 0) {
            return;
        }
        query_listener_->OnQueryIoCompleted(query_id, count);
    }
    // 作用：停止异步线程并 join。
    void stop_async_submit()
    {
        stop_async_submit_.store(true, std::memory_order_relaxed);
        if (submit_thread_.joinable()) {
            submit_thread_.join();
        }
    }
};
