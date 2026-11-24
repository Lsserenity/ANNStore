#include "tests/HnswInternalMonitors.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include "tests/HnswFsRunnerShared.hpp"

namespace fs = std::filesystem;

namespace annfs::tests {
namespace {

std::string lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return s;
}

std::string trim_copy(std::string s)
{
    auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), is_ws));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), is_ws).base(), s.end());
    return s;
}

std::vector<std::string> parse_monitor_kinds()
{
    const char* raw = std::getenv("ANNFS_INTERNAL_MONITORS");
    if (!raw) {
        return {};
    }
    std::stringstream ss{std::string(raw)};
    std::string item;
    std::vector<std::string> out;
    while (std::getline(ss, item, ',')) {
        item = lower_copy(trim_copy(item));
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

bool kind_selected(const std::vector<std::string>& kinds, const std::string& key)
{
    return std::find(kinds.begin(), kinds.end(), lower_copy(key)) != kinds.end();
}

std::string env_or_default(const char* name, const std::string& fallback)
{
    const char* v = std::getenv(name);
    if (!v || std::string(v).empty()) {
        return fallback;
    }
    return std::string(v);
}

std::vector<int> detect_imc_channel_ids()
{
    std::vector<int> ids;
    const fs::path pmu_root("/sys/bus/event_source/devices");
    std::error_code ec;
    if (!fs::exists(pmu_root, ec) || ec) {
        return ids;
    }
    const std::regex pat("^uncore_imc_([0-9]+)$");
    for (const auto& ent : fs::directory_iterator(pmu_root, ec)) {
        if (ec) {
            break;
        }
        if (!ent.is_directory(ec) || ec) {
            continue;
        }
        const std::string name = ent.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, pat)) {
            continue;
        }
        try {
            const int id = std::stoi(m[1].str());
            if (id >= 0) {
                const fs::path events_dir = ent.path() / "events";
                const fs::path rd = events_dir / "cas_count_read";
                const fs::path wr = events_dir / "cas_count_write";
                if (fs::exists(rd) && fs::exists(wr)) {
                    ids.push_back(id);
                }
            }
        } catch (...) {
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

std::string default_mem_imc_all_channel_events()
{
    const auto ids = detect_imc_channel_ids();
    if (ids.empty()) {
        return "uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/";
    }
    std::string events;
    for (size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (i != 0) {
            events += ",";
        }
        events += "uncore_imc_" + std::to_string(id) + "/cas_count_read/";
        events += ",";
        events += "uncore_imc_" + std::to_string(id) + "/cas_count_write/";
    }
    return events;
}

int env_int_or_default(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (!v || std::string(v).empty()) {
        return fallback;
    }
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

bool env_flag_or_default(const char* name, bool fallback)
{
    const char* v = std::getenv(name);
    if (!v || std::string(v).empty()) {
        return fallback;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (s == "1" || s == "true" || s == "yes" || s == "on") {
        return true;
    }
    if (s == "0" || s == "false" || s == "no" || s == "off") {
        return false;
    }
    return fallback;
}

bool prefer_disk_scope()
{
    return env_flag_or_default("ANNFS_INTERNAL_SCOPE_PREFER_DISK", true);
}

bool fallback_to_global_scope()
{
    return env_flag_or_default("ANNFS_INTERNAL_SCOPE_FALLBACK_GLOBAL", true);
}

bool strict_disk_scope()
{
    return env_flag_or_default("ANNFS_INTERNAL_SCOPE_STRICT", false);
}

std::unordered_set<long> parse_tid_filter_env(const char* name)
{
    std::unordered_set<long> tids;
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return tids;
    }
    std::stringstream ss(raw);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim_copy(tok);
        if (tok.empty()) {
            continue;
        }
        try {
            const long tid = std::stol(tok);
            if (tid > 0) {
                tids.insert(tid);
            }
        } catch (...) {
            // Ignore malformed token.
        }
    }
    return tids;
}

bool is_trace_mm_pagecache_event_line(const std::string& line)
{
    return line.find("mm_filemap_add_to_page_cache") != std::string::npos ||
           line.find("mm_filemap_delete_from_page_cache") != std::string::npos ||
           line.find("mm_vmscan_") != std::string::npos;
}

long extract_trace_tid_from_line(const std::string& line)
{
    // Typical prefix:
    // "            exe-757284 [010] 123.456: mm_filemap_add_to_page_cache: ..."
    const size_t cpu_bracket = line.find('[');
    if (cpu_bracket == std::string::npos) {
        return -1;
    }
    std::string prefix = trim_copy(line.substr(0, cpu_bracket));
    if (prefix.empty()) {
        return -1;
    }
    const size_t dash = prefix.rfind('-');
    if (dash == std::string::npos || dash + 1 >= prefix.size()) {
        return -1;
    }
    const std::string pid_str = trim_copy(prefix.substr(dash + 1));
    if (pid_str.empty()) {
        return -1;
    }
    for (char c : pid_str) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return -1;
        }
    }
    try {
        return std::stol(pid_str);
    } catch (...) {
        return -1;
    }
}

std::string monitor_env_token(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       if (std::isalnum(c) != 0) {
                           return static_cast<char>(std::toupper(c));
                       }
                       return '_';
                   });
    return s;
}

int monitor_min_runtime_sec(const std::string& monitor_name)
{
    const std::string specific_key =
        "ANNFS_INTERNAL_" + monitor_env_token(monitor_name) + "_MIN_RUNTIME_SEC";
    const int specific = env_int_or_default(specific_key.c_str(), -1);
    if (specific >= 0) {
        return std::max(0, specific);
    }
    const int global_default = env_int_or_default("ANNFS_INTERNAL_MIN_RUNTIME_SEC", -1);
    if (global_default >= 0) {
        return std::max(0, global_default);
    }
    const std::string name = lower_copy(monitor_name);
    if (name == "dcstat" || name == "cachestat" || name == "vfsstat" ||
        name == "biosnoop" || name == "biolatency") {
        return 15;
    }
    if (name == "ext4slower" || name == "ext4dist" ||
        name == "xfsslower" || name == "xfsdist" ||
        name == "xfssync" || name == "trace_mm_pagecache") {
        return 20;
    }
    return 0;
}

std::string run_first_available(const std::vector<std::string>& cmds)
{
    if (cmds.empty()) {
        return "true";
    }
    std::ostringstream os;
    bool first = true;
    for (const auto& cmd : cmds) {
        if (cmd.empty()) {
            continue;
        }
        if (!first) {
            os << " || ";
        }
        os << "(" << cmd << ")";
        first = false;
    }
    if (first) {
        return "true";
    }
    return os.str();
}

std::string unbuffered_cmd(const std::string& cmd)
{
    if (cmd.empty()) {
        return cmd;
    }
    return "PYTHONUNBUFFERED=1 stdbuf -oL -eL " + cmd;
}

std::string now_compact_utc()
{
    using std::chrono::system_clock;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count() %
                    1000;
    std::tm tm {};
    gmtime_r(&t, &tm);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "%04d%02d%02d_%02d%02d%02d_%03lld_%lld",
                  tm.tm_year + 1900,
                  tm.tm_mon + 1,
                  tm.tm_mday,
                  tm.tm_hour,
                  tm.tm_min,
                  tm.tm_sec,
                  static_cast<long long>(ms),
                  static_cast<long long>(::getpid()));
    return std::string(buf);
}

std::string basename_device(const std::string& path)
{
    if (path.empty()) {
        return "nvme0n1";
    }
    fs::path p(path);
    std::string b = p.filename().string();
    return b.empty() ? path : b;
}

std::string normalize_blktrace_device(const std::string& device_or_path)
{
    if (device_or_path.empty()) {
        return "/dev/nvme0n1";
    }
    if (device_or_path.rfind("/dev/", 0) == 0) {
        return device_or_path;
    }
    return "/dev/" + device_or_path;
}

std::string sanitize_token(std::string s)
{
    if (s.empty()) {
        return "unknown";
    }
    for (char& c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) != 0 || c == '-' || c == '_' || c == '.') {
            continue;
        }
        c = '_';
    }
    return s;
}

std::string env_or_empty(const char* name)
{
    const char* v = std::getenv(name);
    if (!v) {
        return "";
    }
    return std::string(v);
}

std::string monitor_report_path(const std::string& root_dir,
                                const std::string& monitor_name,
                                const std::string& policy_name,
                                const std::string& dataset_name,
                                const std::string& run_id,
                                const std::string& suffix)
{
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string ds = dataset_name.empty() ? "dataset" : dataset_name;
    const std::string base_dir = env_or_empty("ANNFS_INTERNAL_REPORT_BASE_DIR");
    if (!base_dir.empty()) {
        const std::string th = env_or_default("ANNFS_INTERNAL_REPORT_THREADS", "NA");
        const std::string rep = env_or_default("ANNFS_INTERNAL_REPORT_REPEAT_ID", "NA");
        std::string run_idx = env_or_empty("ANNFS_INTERNAL_REPORT_RUN_INDEX");
        if (run_idx.empty()) {
            run_idx = env_or_default("ANNFS_RUN_INDEX_HINT", "NA");
        }
        fs::path p(base_dir);
        p /= sanitize_token(ds);
        p /= "t" + sanitize_token(th);
        p /= "repeat_" + sanitize_token(rep);
        p /= "run_" + sanitize_token(run_idx);
        p /= sanitize_token(st);
        p /= sanitize_token(monitor_name);
        const std::string file =
            sanitize_token(monitor_name) + "_" + sanitize_token(run_id) + suffix;
        p /= file;
        return p.string();
    }
    return root_dir + "/" + monitor_name + "/" + monitor_name + "_" + st + "_" + ds + "_" + run_id + suffix;
}

std::string replace_extension_or_append(const std::string& path, const std::string& new_ext)
{
    fs::path p(path);
    if (p.has_extension()) {
        p.replace_extension(new_ext);
        return p.string();
    }
    return path + new_ext;
}

class ShellCommandMonitor : public IInternalMonitor {
public:
    ShellCommandMonitor(std::string name,
                        std::string command,
                        std::string report_path)
        : name_(std::move(name)),
          command_(std::move(command)),
          report_path_(std::move(report_path)),
          min_runtime_sec_(monitor_min_runtime_sec(name_)) {}

    const char* Name() const override { return name_.c_str(); }
    bool Enabled() const override { return true; }

    bool Start() override
    {
        if (running_) {
            return true;
        }
        const pid_t pid = ::fork();
        if (pid < 0) {
            std::cerr << "[INTERNAL_" << name_ << "] fork failed\n";
            return false;
        }
        if (pid == 0) {
            (void)::setpgid(0, 0);
            ::execl("/bin/bash", "bash", "-lc", command_.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        (void)::setpgid(pid, pid);
        pid_ = pid;
        started_at_ = std::chrono::steady_clock::now();
        running_ = true;
        std::cout << "[INTERNAL_" << name_ << "] started pid=" << pid_
                  << " min_runtime_sec=" << min_runtime_sec_ << "\n";
        return true;
    }

    bool Stop() override
    {
        if (!running_) {
            return true;
        }
        if (min_runtime_sec_ > 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - started_at_);
            if (elapsed.count() < min_runtime_sec_) {
                std::this_thread::sleep_for(
                    std::chrono::seconds(min_runtime_sec_ - elapsed.count()));
            }
        }
        ::kill(-pid_, SIGINT);
        int status = 0;
        int tries = 0;
        while (tries < 50) { // ~5s
            const pid_t rc = ::waitpid(pid_, &status, WNOHANG);
            if (rc == pid_) {
                break;
            }
            if (rc < 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ++tries;
        }
        if (::waitpid(pid_, &status, WNOHANG) == 0) {
            ::kill(-pid_, SIGTERM);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            if (::waitpid(pid_, &status, WNOHANG) == 0) {
                ::kill(-pid_, SIGKILL);
            }
            (void)::waitpid(pid_, &status, 0);
        }
        running_ = false;
        pid_ = -1;
        std::cout << "[INTERNAL_" << name_ << "] REPORT_FILE=" << report_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_" + name_ + "_report"] = report_path_;
    }

private:
    std::string name_;
    std::string command_;
    std::string report_path_;
    int min_runtime_sec_ = 0;
    bool running_ = false;
    pid_t pid_ = -1;
    std::chrono::steady_clock::time_point started_at_ {};
};

class InternalFsBlktraceMonitor : public IInternalMonitor {
public:
    InternalFsBlktraceMonitor(const HnswIoCliArgs& args,
                              const ParamConfig& pc,
                              const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        const std::string dev_guess = basename_device(pc.dev_path);
        device_ = normalize_blktrace_device(env_or_default("ANNFS_INTERNAL_BLKTRACE_DEVICE", dev_guess));
        tmp_dir_ = env_or_default("ANNFS_INTERNAL_BLKTRACE_TMP_DIR", "./tmp");
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_REPORT",
            monitor_report_path("./blktrace_report", "blktrace", st, ds, run_id, ".txt"));
        blkparse_bin_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BLKPARSE_BIN",
            replace_extension_or_append(report_path_, ".bin"));
        btt_summary_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_SUMMARY",
            replace_extension_or_append(report_path_, ".btt.txt"));
        btt_q2c_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_Q2C",
            replace_extension_or_append(report_path_, ".btt_q2c_latency.txt"));
        btt_q2d_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_Q2D",
            replace_extension_or_append(report_path_, ".btt_q2d_latency.txt"));
        btt_d2c_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_D2C",
            replace_extension_or_append(report_path_, ".btt_d2c_latency.txt"));
        btt_qdepth_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_QDEPTH",
            replace_extension_or_append(report_path_, ".btt_queue_depth.txt"));
        btt_iostat_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_IOSTAT",
            replace_extension_or_append(report_path_, ".btt_iostat.txt"));
        btt_blocknos_path_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BTT_BLOCKNOS",
            replace_extension_or_append(report_path_, ".btt_blocknos.txt"));
        enable_btt_ = env_flag_or_default("ANNFS_INTERNAL_BLKTRACE_ENABLE_BTT", true);
        btt_required_ = env_flag_or_default("ANNFS_INTERNAL_BLKTRACE_BTT_REQUIRED", false);
        keep_tmp_trace_ = env_flag_or_default("ANNFS_INTERNAL_BLKTRACE_KEEP_TMP", false);
        trace_base_ = env_or_default(
            "ANNFS_INTERNAL_BLKTRACE_BASE",
            "blktrace_internal_" + std::to_string(static_cast<long long>(::getpid())) + "_" + run_id);
        err_path_ = tmp_dir_ + "/" + trace_base_ + ".err.log";
        trace_prefix_path_ = tmp_dir_ + "/" + trace_base_;
    }

    const char* Name() const override { return "blktrace"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(tmp_dir_, ec);
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        fs::create_directories(fs::path(blkparse_bin_path_).parent_path(), ec);
        fs::create_directories(fs::path(btt_summary_path_).parent_path(), ec);

        const pid_t pid = ::fork();
        if (pid < 0) {
            std::cerr << "[INTERNAL_BLKTRACE] fork failed for blktrace\n";
            return false;
        }
        if (pid == 0) {
            int errfd = ::open(err_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (errfd >= 0) {
                ::dup2(errfd, STDERR_FILENO);
                ::close(errfd);
            }
            int nullfd = ::open("/dev/null", O_WRONLY);
            if (nullfd >= 0) {
                ::dup2(nullfd, STDOUT_FILENO);
                ::close(nullfd);
            }
            ::execlp("blktrace", "blktrace",
                     "-d", device_.c_str(),
                     "-D", tmp_dir_.c_str(),
                     "-o", trace_base_.c_str(),
                     static_cast<char*>(nullptr));
            _exit(127);
        }
        blktrace_pid_ = pid;
        const int warmup_ms = std::max(0, env_int_or_default("ANNFS_INTERNAL_BLKTRACE_WARMUP_MS", 3000));
        if (warmup_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(warmup_ms));
        }
        running_ = true;
        std::cout << "[INTERNAL_BLKTRACE] started pid=" << blktrace_pid_
                  << " device=" << device_
                  << " warmup_ms=" << warmup_ms
                  << " prefix=" << trace_prefix_path_ << "\n";
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        ::kill(blktrace_pid_, SIGINT);
        int status = 0;
        ::waitpid(blktrace_pid_, &status, 0);
        running_ = false;
        blktrace_pid_ = -1;

        const pid_t pid = ::fork();
        if (pid < 0) {
            std::cerr << "[INTERNAL_BLKTRACE] fork failed for blkparse\n";
            return false;
        }
        if (pid == 0) {
            int outfd = ::open(report_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (outfd < 0) {
                _exit(126);
            }
            ::dup2(outfd, STDOUT_FILENO);
            ::close(outfd);
            int nullfd = ::open("/dev/null", O_WRONLY);
            if (nullfd >= 0) {
                ::dup2(nullfd, STDERR_FILENO);
                ::close(nullfd);
            }
            ::execlp("blkparse", "blkparse",
                     "-i", trace_prefix_path_.c_str(),
                     "-d", blkparse_bin_path_.c_str(),
                     static_cast<char*>(nullptr));
            _exit(127);
        }
        int parse_status = 0;
        ::waitpid(pid, &parse_status, 0);
        const bool ok = WIFEXITED(parse_status) && WEXITSTATUS(parse_status) == 0;
        if (!ok) {
            std::cerr << "[INTERNAL_BLKTRACE] blkparse failed for prefix=" << trace_prefix_path_ << "\n";
            return false;
        }
        std::cout << "[INTERNAL_BLKTRACE] REPORT_FILE=" << report_path_ << "\n";
        std::cout << "[INTERNAL_BLKTRACE] BLKPARSE_BIN=" << blkparse_bin_path_ << "\n";

        if (enable_btt_) {
            const pid_t btt_pid = ::fork();
            if (btt_pid < 0) {
                std::cerr << "[INTERNAL_BLKTRACE] fork failed for btt\n";
                if (btt_required_) {
                    return false;
                }
            } else if (btt_pid == 0) {
                const int nullfd = ::open("/dev/null", O_WRONLY);
                if (nullfd >= 0) {
                    ::dup2(nullfd, STDOUT_FILENO);
                    ::dup2(nullfd, STDERR_FILENO);
                    ::close(nullfd);
                }
                ::execlp("btt", "btt",
                         "-i", blkparse_bin_path_.c_str(),
                         "-o", btt_summary_path_.c_str(),
                         "-q", btt_q2c_path_.c_str(),
                         "-z", btt_q2d_path_.c_str(),
                         "-l", btt_d2c_path_.c_str(),
                         "-Q", btt_qdepth_path_.c_str(),
                         "-I", btt_iostat_path_.c_str(),
                         "-B", btt_blocknos_path_.c_str(),
                         static_cast<char*>(nullptr));
                _exit(127);
            } else {
                int btt_status = 0;
                ::waitpid(btt_pid, &btt_status, 0);
                const bool btt_ok = WIFEXITED(btt_status) && WEXITSTATUS(btt_status) == 0;
                if (!btt_ok) {
                    std::cerr << "[INTERNAL_BLKTRACE] btt failed for input=" << blkparse_bin_path_ << "\n";
                    if (btt_required_) {
                        return false;
                    }
                } else {
                    std::cout << "[INTERNAL_BLKTRACE] BTT_SUMMARY_FILE=" << btt_summary_path_ << "\n";
                    std::cout << "[INTERNAL_BLKTRACE] BTT_Q2C_FILE=" << btt_q2c_path_ << "\n";
                    std::cout << "[INTERNAL_BLKTRACE] BTT_Q2D_FILE=" << btt_q2d_path_ << "\n";
                    std::cout << "[INTERNAL_BLKTRACE] BTT_D2C_FILE=" << btt_d2c_path_ << "\n";
                    std::cout << "[INTERNAL_BLKTRACE] BTT_QDEPTH_FILE=" << btt_qdepth_path_ << "\n";
                    std::cout << "[INTERNAL_BLKTRACE] BTT_IOSTAT_FILE=" << btt_iostat_path_ << "\n";
                    std::cout << "[INTERNAL_BLKTRACE] BTT_BLOCKNOS_FILE=" << btt_blocknos_path_ << "\n";
                }
            }
        }

        if (!keep_tmp_trace_) {
            for (const auto& e : fs::directory_iterator(tmp_dir_)) {
                const std::string p = e.path().string();
                if (p.rfind(trace_prefix_path_, 0) == 0) {
                    std::error_code ec;
                    fs::remove(e.path(), ec);
                }
            }
        }
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_blktrace_report"] = report_path_;
        io_meta["internal_blktrace_blkparse_bin"] = blkparse_bin_path_;
        io_meta["internal_blktrace_btt_enabled"] = enable_btt_;
        io_meta["internal_blktrace_btt_required"] = btt_required_;
        io_meta["internal_blktrace_keep_tmp"] = keep_tmp_trace_;
        io_meta["internal_blktrace_btt_summary"] = btt_summary_path_;
        io_meta["internal_blktrace_btt_q2c"] = btt_q2c_path_;
        io_meta["internal_blktrace_btt_q2d"] = btt_q2d_path_;
        io_meta["internal_blktrace_btt_d2c"] = btt_d2c_path_;
        io_meta["internal_blktrace_btt_qdepth"] = btt_qdepth_path_;
        io_meta["internal_blktrace_btt_iostat"] = btt_iostat_path_;
        io_meta["internal_blktrace_btt_blocknos"] = btt_blocknos_path_;
    }

private:
    bool enabled_ = false;
    bool running_ = false;
    pid_t blktrace_pid_ = -1;
    std::string device_;
    std::string tmp_dir_;
    std::string report_path_;
    std::string blkparse_bin_path_;
    std::string btt_summary_path_;
    std::string btt_q2c_path_;
    std::string btt_q2d_path_;
    std::string btt_d2c_path_;
    std::string btt_qdepth_path_;
    std::string btt_iostat_path_;
    std::string btt_blocknos_path_;
    std::string trace_base_;
    std::string trace_prefix_path_;
    std::string err_path_;
    bool enable_btt_ = true;
    bool btt_required_ = false;
    bool keep_tmp_trace_ = false;
};

class InternalRssMonitor : public IInternalMonitor {
public:
    InternalRssMonitor(const HnswIoCliArgs& args,
                       const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_RSS_REPORT",
            monitor_report_path("./rss_report", "rss", st, ds, run_id, ".tsv"));
        interval_ms_ = std::max(10, env_int_or_default("ANNFS_INTERNAL_RSS_INTERVAL_MS", 100));
    }

    ~InternalRssMonitor() override
    {
        if (running_) {
            Stop();
        }
    }

    const char* Name() const override { return "rss"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        out_.open(report_path_, std::ios::out | std::ios::trunc);
        if (!out_.is_open()) {
            std::cerr << "[INTERNAL_RSS] failed to open report: " << report_path_ << "\n";
            return false;
        }
        out_ << "ts_ms\tvmrss_kib\tvmhwm_kib\n";
        sample_once();
        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
                if (!running_) {
                    break;
                }
                sample_once();
            }
        });
        std::cout << "[INTERNAL_RSS] started interval_ms=" << interval_ms_
                  << " report=" << report_path_ << "\n";
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        sample_once();
        out_.flush();
        out_.close();
        std::cout << "[INTERNAL_RSS] REPORT_FILE=" << report_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_rss_report"] = report_path_;
        io_meta["internal_rss_interval_ms"] = interval_ms_;
    }

private:
    std::pair<long long, long long> read_proc_status() const
    {
        std::ifstream in("/proc/self/status");
        if (!in.is_open()) {
            return {0, 0};
        }
        std::string line;
        long long rss = 0;
        long long hwm = 0;
        while (std::getline(in, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                std::stringstream ss(line.substr(6));
                ss >> rss;
            } else if (line.rfind("VmHWM:", 0) == 0) {
                std::stringstream ss(line.substr(6));
                ss >> hwm;
            }
        }
        return {rss, hwm};
    }

    void sample_once()
    {
        auto [rss, hwm] = read_proc_status();
        std::lock_guard<std::mutex> lk(mu_);
        if (out_.is_open()) {
            out_ << fsrunner::epoch_ms() << "\t" << rss << "\t" << hwm << "\n";
        }
    }

    bool enabled_ = false;
    std::atomic<bool> running_ {false};
    int interval_ms_ = 100;
    std::string report_path_;
    std::ofstream out_;
    std::thread worker_;
    std::mutex mu_;
};

std::unique_ptr<IInternalMonitor> build_perf_monitor(const HnswIoCliArgs& args,
                                                     const std::string& policy_name)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PERF_REPORT",
        monitor_report_path("./perf_report", "perf", st, ds, run_id, ".csv"));
    const std::string events = env_or_default("ANNFS_INTERNAL_PERF_EVENTS", "LLC-loads,LLC-load-misses");
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && perf stat -e '" + events + "' -x, -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            " -o '" + report + "'";
    return std::make_unique<ShellCommandMonitor>("perf", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_mem_access_monitor(const HnswIoCliArgs& args,
                                                           const std::string& policy_name)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_MEM_ACCESS_REPORT",
        monitor_report_path("./mem_access_report", "mem_access", st, ds, run_id, ".csv"));
    const std::string events = env_or_default(
        "ANNFS_INTERNAL_MEM_ACCESS_EVENTS",
        "cache-references,cache-misses,LLC-loads,LLC-load-misses");
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && perf stat -e '" + events + "' -x, -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            " -o '" + report + "'";
    return std::make_unique<ShellCommandMonitor>("mem_access", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_mem_imc_monitor(const HnswIoCliArgs& args,
                                                        const std::string& policy_name)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_MEM_IMC_REPORT",
        monitor_report_path("./mem_imc_report", "mem_imc", st, ds, run_id, ".csv"));
    // 默认使用本机全部可用 IMC 通道（imc_0..imc_N）的 CAS 读写计数汇总。
    const std::string events = env_or_default(
        "ANNFS_INTERNAL_MEM_IMC_EVENTS",
        default_mem_imc_all_channel_events());
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && perf stat -a -e '" + events + "' -x, -o '" + report + "'";
    return std::make_unique<ShellCommandMonitor>("mem_imc", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_mem_imc0_monitor(const HnswIoCliArgs& args,
                                                         const std::string& policy_name)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_MEM_IMC0_REPORT",
        monitor_report_path("./mem_imc0_report", "mem_imc_0", st, ds, run_id, ".csv"));
    const std::string events = env_or_default(
        "ANNFS_INTERNAL_MEM_IMC0_EVENTS",
        "uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/");
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && perf stat -a -e '" + events + "' -x, -o '" + report + "'";
    return std::make_unique<ShellCommandMonitor>("mem_imc_0", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_numa_node_monitor(const HnswIoCliArgs& args,
                                                          const std::string& policy_name)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_NUMA_NODE_REPORT",
        monitor_report_path("./numa_node_report", "numa_node", st, ds, run_id, ".csv"));
    const std::string events = env_or_default(
        "ANNFS_INTERNAL_NUMA_NODE_EVENTS",
        "node-loads,node-stores,node-load-misses,node-store-misses");
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && perf stat -a -e '" + events + "' -x, -o '" + report + "'";
    return std::make_unique<ShellCommandMonitor>("numa_node", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_pcm_memory_monitor(const HnswIoCliArgs& args,
                                                           const std::string& policy_name)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PCM_MEMORY_REPORT",
        monitor_report_path("./pcm_memory_report", "pcm_memory", st, ds, run_id, ".log"));
    // pcm-memory 默认每秒采样；可通过环境变量覆盖参数（例如 "1 -i=10"）
    const std::string args_str = env_or_default("ANNFS_INTERNAL_PCM_MEMORY_ARGS", "1");
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && (pcm-memory " + args_str + " || pcm-memory.x " + args_str +
                            ") > '" + report + "' 2>&1";
    return std::make_unique<ShellCommandMonitor>("pcm_memory", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_pidstat_monitor(const std::string& policy_name,
                                                        const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PIDSTAT_REPORT",
        monitor_report_path("./pidstat_report", "pidstat", st, ds, run_id, ".log"));
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && pidstat -u -w -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            " 1 > '" + report + "' 2>&1";
    return std::make_unique<ShellCommandMonitor>("pidstat", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_iostat_monitor(const std::string& policy_name,
                                                       const HnswIoCliArgs& args,
                                                       const ParamConfig& pc)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_IOSTAT_REPORT",
        monitor_report_path("./iostat_report", "iostat", st, ds, run_id, ".log"));
    const std::string dev = env_or_default("ANNFS_INTERNAL_IOSTAT_DEVICE", basename_device(pc.dev_path));
    const bool prefer_disk = prefer_disk_scope();
    const bool allow_global_fallback = fallback_to_global_scope();
    const bool strict_scope = strict_disk_scope();
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (prefer_disk) {
        cmd += "dev='" + dev + "'; dev=${dev##*/}; "
               "if [ -e \"/sys/class/block/${dev}/stat\" ]; then "
               "  if ! iostat -dx 1 \"$dev\" > '" + report + "' 2>&1; then ";
        if (allow_global_fallback && !strict_scope) {
            cmd +=
                "echo \"# scope=disk failed for iostat device=$dev; fallback=global\" >> '" + report + "'; "
                "iostat -dx 1 >> '" + report + "' 2>&1; ";
        } else {
            cmd +=
                "echo \"# scope=disk failed for iostat device=$dev; strict mode\" >> '" + report + "'; "
                "exit 1; ";
        }
        cmd +=
            "  fi; "
            "else ";
        if (allow_global_fallback && !strict_scope) {
            cmd +=
                "echo \"# scope=disk unavailable for iostat device=$dev; fallback=global\" > '" + report + "'; "
                "iostat -dx 1 >> '" + report + "' 2>&1; ";
        } else {
            cmd +=
                "echo \"# scope=disk unavailable for iostat device=$dev; strict mode\" > '" + report + "'; "
                "exit 1; ";
        }
        cmd += "fi";
    } else {
        cmd += "iostat -dx 1 > '" + report + "' 2>&1";
    }
    return std::make_unique<ShellCommandMonitor>("iostat", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_pidstat_r_monitor(const std::string& policy_name,
                                                          const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PIDSTAT_R_REPORT",
        monitor_report_path("./pidstat_r_report", "pidstat_r", st, ds, run_id, ".log"));
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && pidstat -r -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            " 1 > '" + report + "' 2>&1";
    return std::make_unique<ShellCommandMonitor>("pidstat_r", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_pidstat_d_monitor(const std::string& policy_name,
                                                          const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PIDSTAT_D_REPORT",
        monitor_report_path("./pidstat_d_report", "pidstat_d", st, ds, run_id, ".log"));
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && pidstat -d -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            " 1 > '" + report + "' 2>&1";
    return std::make_unique<ShellCommandMonitor>("pidstat_d", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_strace_monitor(const std::string& policy_name,
                                                       const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_STRACE_REPORT",
        monitor_report_path("./strace_report", "strace", st, ds, run_id, "_c.txt"));
    const std::string trace_list = env_or_default(
        "ANNFS_INTERNAL_STRACE_TRACE",
        "io_uring_enter,io_uring_setup,io_uring_register,openat,openat2,close,read,pread64");
    const int max_sec = env_int_or_default("ANNFS_INTERNAL_STRACE_MAX_SEC", 0);
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (max_sec > 0) {
        cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
    }
    cmd += "(strace -f -c -e trace=" + trace_list +
           " -p " + std::to_string(static_cast<long long>(::getpid())) +
           " -o '" + report + "' 2>/dev/null || " +
           "strace -f -c -p " + std::to_string(static_cast<long long>(::getpid())) +
           " -o '" + report + "' 2>/dev/null) || true";
    return std::make_unique<ShellCommandMonitor>("strace", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_perf_fs_monitor(const std::string& policy_name,
                                                        const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PERF_FS_REPORT",
        monitor_report_path("./perf_fs_report", "perf_fs", st, ds, run_id, ".data"));
    const int max_sec = env_int_or_default("ANNFS_INTERNAL_PERF_FS_MAX_SEC", 0);
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (max_sec > 0) {
        cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
    }
    cmd += "perf record -g -p " + std::to_string(static_cast<long long>(::getpid())) +
           " -o '" + report + "' || true";
    return std::make_unique<ShellCommandMonitor>("perf_fs", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_perf_sched_monitor(const std::string& policy_name,
                                                           const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_PERF_SCHED_REPORT",
        monitor_report_path("./perf_sched_report", "perf_sched", st, ds, run_id, ".data"));
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && perf sched record -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            " -o '" + report + "'";
    return std::make_unique<ShellCommandMonitor>("perf_sched", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_numastat_monitor(const std::string& policy_name,
                                                         const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_NUMASTAT_REPORT",
        monitor_report_path("./numastat_report", "numastat", st, ds, run_id, ".log"));
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && while true; do numastat -p " +
                            std::to_string(static_cast<long long>(::getpid())) +
                            "; sleep 1; done > '" + report + "' 2>&1";
    return std::make_unique<ShellCommandMonitor>("numastat", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_dcstat_monitor(const std::string& policy_name,
                                                       const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_DCSTAT_REPORT",
        monitor_report_path("./dcstat_report", "dcstat", st, ds, run_id, ".log"));
    const int interval_sec = std::max(1, env_int_or_default("ANNFS_INTERNAL_DCSTAT_INTERVAL_SEC", 1));
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/dcstat-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("dcstat-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("/usr/share/bcc/tools/dcstat " + std::to_string(interval_sec)),
        unbuffered_cmd("dcstat " + std::to_string(interval_sec)),
    });
    const bool prefer_disk = prefer_disk_scope();
    const bool strict_scope = strict_disk_scope();
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (prefer_disk) {
        if (strict_scope) {
            cmd += "echo \"# scope=disk requested but unsupported for dcstat; strict mode\" > '" + report +
                   "'; exit 1";
        } else {
            cmd += "echo \"# scope=disk requested but unsupported for dcstat; fallback=global\" > '" + report +
                   "'; ((" + run_expr + " || true) >> '" + report + "' 2>&1)";
        }
    } else {
        cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    }
    return std::make_unique<ShellCommandMonitor>("dcstat", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_cachestat_monitor(const std::string& policy_name,
                                                          const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_CACHESTAT_REPORT",
        monitor_report_path("./cachestat_report", "cachestat", st, ds, run_id, ".log"));
    const int interval_sec = std::max(1, env_int_or_default("ANNFS_INTERNAL_CACHESTAT_INTERVAL_SEC", 1));
    const std::string compat_cmd = env_or_default(
        "ANNFS_INTERNAL_CACHESTAT_COMPAT_CMD",
        "python3 fs-bottleneck-lab/scripts/cachestat_compat_bpfcc.py");
    const std::string run_expr = run_first_available({
        unbuffered_cmd(compat_cmd + " " + std::to_string(interval_sec)),
        unbuffered_cmd("/usr/sbin/cachestat-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("cachestat-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("/usr/share/bcc/tools/cachestat " + std::to_string(interval_sec)),
        unbuffered_cmd("cachestat " + std::to_string(interval_sec)),
    });
    const bool prefer_disk = prefer_disk_scope();
    const bool strict_scope = strict_disk_scope();
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (prefer_disk) {
        if (strict_scope) {
            cmd += "echo \"# scope=disk requested but unsupported for cachestat; strict mode\" > '" + report +
                   "'; exit 1";
        } else {
            cmd += "echo \"# scope=disk requested but unsupported for cachestat; fallback=global\" > '" + report +
                   "'; ((" + run_expr + " || true) >> '" + report + "' 2>&1)";
        }
    } else {
        cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    }
    return std::make_unique<ShellCommandMonitor>("cachestat", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_vfsstat_monitor(const std::string& policy_name,
                                                        const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_VFSSTAT_REPORT",
        monitor_report_path("./vfsstat_report", "vfsstat", st, ds, run_id, ".log"));
    const int interval_sec = std::max(1, env_int_or_default("ANNFS_INTERNAL_VFSSTAT_INTERVAL_SEC", 1));
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/vfsstat-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("vfsstat-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("/usr/share/bcc/tools/vfsstat " + std::to_string(interval_sec)),
        unbuffered_cmd("vfsstat " + std::to_string(interval_sec)),
    });
    const bool prefer_disk = prefer_disk_scope();
    const bool strict_scope = strict_disk_scope();
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (prefer_disk) {
        if (strict_scope) {
            cmd += "echo \"# scope=disk requested but unsupported for vfsstat; strict mode\" > '" + report +
                   "'; exit 1";
        } else {
            cmd += "echo \"# scope=disk requested but unsupported for vfsstat; fallback=global\" > '" + report +
                   "'; ((" + run_expr + " || true) >> '" + report + "' 2>&1)";
        }
    } else {
        cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    }
    return std::make_unique<ShellCommandMonitor>("vfsstat", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_biosnoop_monitor(const std::string& policy_name,
                                                         const HnswIoCliArgs& args,
                                                         const ParamConfig& pc)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_BIOSNOOP_REPORT",
        monitor_report_path("./biosnoop_report", "biosnoop", st, ds, run_id, ".log"));
    const std::string dev = env_or_default("ANNFS_INTERNAL_BIOSNOOP_DEVICE", basename_device(pc.dev_path));
    const int max_sec = env_int_or_default("ANNFS_INTERNAL_BIOSNOOP_MAX_SEC", 0);
    const std::string run_expr_disk = run_first_available({
        unbuffered_cmd("/usr/sbin/biosnoop-bpfcc -d " + dev),
        unbuffered_cmd("biosnoop-bpfcc -d " + dev),
        unbuffered_cmd("/usr/share/bcc/tools/biosnoop -d " + dev),
        unbuffered_cmd("biosnoop -d " + dev),
    });
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/biosnoop-bpfcc"),
        unbuffered_cmd("biosnoop-bpfcc"),
        unbuffered_cmd("/usr/share/bcc/tools/biosnoop"),
        unbuffered_cmd("biosnoop"),
        unbuffered_cmd(
            "bpftrace -e 'BEGIN { printf(\"TIME(s) PID DEV RWBS BYTES LAT(us)\\\\n\"); } "
            "tracepoint:block:block_rq_issue { "
            "@ts[args->dev, args->sector] = nsecs; "
            "@pid[args->dev, args->sector] = pid; "
            "} "
            "tracepoint:block:block_rq_complete /@ts[args->dev, args->sector]/ { "
            "$lat_us = (nsecs - @ts[args->dev, args->sector]) / 1000; "
            "$bytes = args->nr_sector * 512; "
            "printf(\"%llu %-7d %-10d %-3s %-8llu %llu\\\\n\", "
            "elapsed / 1000000000, "
            "@pid[args->dev, args->sector], "
            "args->dev, args->rwbs, $bytes, $lat_us); "
            "delete(@ts[args->dev, args->sector]); "
            "delete(@pid[args->dev, args->sector]); "
            "}'"),
    });
    const bool prefer_disk = prefer_disk_scope();
    const bool allow_global_fallback = fallback_to_global_scope();
    const bool strict_scope = strict_disk_scope();
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (prefer_disk) {
        cmd += "dev='" + dev + "'; dev=${dev##*/}; "
               "if [ -e \"/sys/class/block/${dev}/stat\" ]; then "
               "  ";
        if (max_sec > 0) {
            cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
        }
        cmd += "((" + run_expr_disk + " || true) > '" + report + "' 2>&1); "
               "if [ ! -s '" + report + "' ]; then ";
        if (allow_global_fallback && !strict_scope) {
            cmd += "echo \"# scope=disk produced empty output for biosnoop device=$dev; fallback=global\" > '" +
                   report + "'; ";
            if (max_sec > 0) {
                cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
            }
            cmd += "((" + run_expr + " || true) >> '" + report + "' 2>&1); ";
        } else {
            cmd += "echo \"# scope=disk produced empty output for biosnoop device=$dev; strict mode\" > '" + report +
                   "'; exit 1; ";
        }
        cmd += "fi; "
               "else ";
        if (allow_global_fallback && !strict_scope) {
            cmd += "echo \"# scope=disk unavailable for biosnoop device=$dev; fallback=global\" > '" + report +
                   "'; ";
            if (max_sec > 0) {
                cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
            }
            cmd += "((" + run_expr + " || true) >> '" + report + "' 2>&1); ";
        } else {
            cmd += "echo \"# scope=disk unavailable for biosnoop device=$dev; strict mode\" > '" + report +
                   "'; exit 1; ";
        }
        cmd += "fi";
        return std::make_unique<ShellCommandMonitor>("biosnoop", cmd, report);
    }
    if (max_sec > 0) {
        cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
    }
    cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    return std::make_unique<ShellCommandMonitor>("biosnoop", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_biolatency_monitor(const std::string& policy_name,
                                                           const HnswIoCliArgs& args,
                                                           const ParamConfig& pc)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_BIOLATENCY_REPORT",
        monitor_report_path("./biolatency_report", "biolatency", st, ds, run_id, ".log"));
    const std::string dev = env_or_default("ANNFS_INTERNAL_BIOLATENCY_DEVICE", basename_device(pc.dev_path));
    const int interval_sec = std::max(1, env_int_or_default("ANNFS_INTERNAL_BIOLATENCY_INTERVAL_SEC", 1));
    const int max_sec = env_int_or_default("ANNFS_INTERNAL_BIOLATENCY_MAX_SEC", 0);
    const std::string run_expr_disk = run_first_available({
        unbuffered_cmd("/usr/sbin/biolatency-bpfcc -d " + dev + " " + std::to_string(interval_sec)),
        unbuffered_cmd("biolatency-bpfcc -d " + dev + " " + std::to_string(interval_sec)),
        unbuffered_cmd("/usr/share/bcc/tools/biolatency -d " + dev + " " + std::to_string(interval_sec)),
        unbuffered_cmd("biolatency -d " + dev + " " + std::to_string(interval_sec)),
    });
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/biolatency-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("biolatency-bpfcc " + std::to_string(interval_sec)),
        unbuffered_cmd("/usr/share/bcc/tools/biolatency " + std::to_string(interval_sec)),
        unbuffered_cmd("biolatency " + std::to_string(interval_sec)),
    });
    const bool prefer_disk = prefer_disk_scope();
    const bool allow_global_fallback = fallback_to_global_scope();
    const bool strict_scope = strict_disk_scope();
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (prefer_disk) {
        cmd += "dev='" + dev + "'; dev=${dev##*/}; "
               "if [ -e \"/sys/class/block/${dev}/stat\" ]; then "
               "  ";
        if (max_sec > 0) {
            cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
        }
        cmd += "((" + run_expr_disk + " || true) > '" + report + "' 2>&1); "
               "if [ ! -s '" + report + "' ]; then ";
        if (allow_global_fallback && !strict_scope) {
            cmd += "echo \"# scope=disk produced empty output for biolatency device=$dev; fallback=global\" > '" +
                   report + "'; ";
            if (max_sec > 0) {
                cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
            }
            cmd += "((" + run_expr + " || true) >> '" + report + "' 2>&1); ";
        } else {
            cmd += "echo \"# scope=disk produced empty output for biolatency device=$dev; strict mode\" > '" +
                   report + "'; exit 1; ";
        }
        cmd += "fi; "
               "else ";
        if (allow_global_fallback && !strict_scope) {
            cmd += "echo \"# scope=disk unavailable for biolatency device=$dev; fallback=global\" > '" + report +
                   "'; ";
            if (max_sec > 0) {
                cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
            }
            cmd += "((" + run_expr + " || true) >> '" + report + "' 2>&1); ";
        } else {
            cmd += "echo \"# scope=disk unavailable for biolatency device=$dev; strict mode\" > '" + report +
                   "'; exit 1; ";
        }
        cmd += "fi";
        return std::make_unique<ShellCommandMonitor>("biolatency", cmd, report);
    }
    if (max_sec > 0) {
        cmd += "timeout --signal=INT " + std::to_string(max_sec) + "s ";
    }
    cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    return std::make_unique<ShellCommandMonitor>("biolatency", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_ext4slower_monitor(const std::string& policy_name,
                                                           const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_EXT4SLOWER_REPORT",
        monitor_report_path("./ext4slower_report", "ext4slower", st, ds, run_id, ".log"));
    const int threshold_ms = std::max(1, env_int_or_default("ANNFS_INTERNAL_EXT4SLOWER_THRESHOLD_MS", 10));
    const long target_pid = static_cast<long>(std::max(
        1, env_int_or_default("ANNFS_INTERNAL_TARGET_PID", static_cast<int>(::getpid()))));
    const bool use_pid_filter = env_flag_or_default(
        "ANNFS_INTERNAL_EXT4SLOWER_USE_PID_FILTER",
        env_flag_or_default("ANNFS_INTERNAL_EXT4_USE_PID_FILTER", true));
    const bool read_only = env_int_or_default("ANNFS_INTERNAL_EXT4SLOWER_READ_ONLY", 0) != 0;
    const std::string pid_opt = use_pid_filter ? (" -p " + std::to_string(target_pid)) : "";
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/ext4slower-bpfcc" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("ext4slower-bpfcc" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("/usr/share/bcc/tools/ext4slower" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("ext4slower" + pid_opt + " " + std::to_string(threshold_ms)),
    });
    const std::string run_expr_csv = run_first_available({
        unbuffered_cmd("/usr/sbin/ext4slower-bpfcc -j" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("ext4slower-bpfcc -j" + pid_opt + " " + std::to_string(threshold_ms)),
    });
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (read_only) {
        cmd += "((" + run_expr_csv +
               " || true) 2>&1 | awk -F',' 'NR==1 || $4==\"R\" {print}' > '" + report + "')";
    } else {
        cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    }
    return std::make_unique<ShellCommandMonitor>("ext4slower", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_ext4dist_monitor(const std::string& policy_name,
                                                         const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_EXT4DIST_REPORT",
        monitor_report_path("./ext4dist_report", "ext4dist", st, ds, run_id, ".log"));
    const long target_pid = static_cast<long>(std::max(
        1, env_int_or_default("ANNFS_INTERNAL_TARGET_PID", static_cast<int>(::getpid()))));
    const bool use_pid_filter = env_flag_or_default(
        "ANNFS_INTERNAL_EXT4DIST_USE_PID_FILTER",
        env_flag_or_default("ANNFS_INTERNAL_EXT4_USE_PID_FILTER", true));
    const bool read_only = env_int_or_default("ANNFS_INTERNAL_EXT4DIST_READ_ONLY", 0) != 0;
    const std::string pid_opt = use_pid_filter ? (" -p " + std::to_string(target_pid)) : "";
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/ext4dist-bpfcc" + pid_opt),
        unbuffered_cmd("ext4dist-bpfcc" + pid_opt),
        unbuffered_cmd("/usr/share/bcc/tools/ext4dist" + pid_opt),
        unbuffered_cmd("ext4dist" + pid_opt),
    });
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (read_only) {
        cmd += "((" + run_expr + " || true) 2>&1 | "
               "awk 'BEGIN{keep=0} "
               "/^Tracing ext4/ {print; next} "
               "/^operation[[:space:]]*=[[:space:]]*read[[:space:]]*$/ {keep=1; print; next} "
               "/^operation[[:space:]]*=/ {keep=0} "
               "{if(keep) print}' > '" + report + "')";
    } else {
        cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    }
    return std::make_unique<ShellCommandMonitor>("ext4dist", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_xfsslower_monitor(const std::string& policy_name,
                                                          const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_XFSSLOWER_REPORT",
        monitor_report_path("./xfsslower_report", "xfsslower", st, ds, run_id, ".log"));
    const int threshold_ms = std::max(1, env_int_or_default("ANNFS_INTERNAL_XFSSLOWER_THRESHOLD_MS", 10));
    const long target_pid = static_cast<long>(std::max(
        1, env_int_or_default("ANNFS_INTERNAL_TARGET_PID", static_cast<int>(::getpid()))));
    const bool use_pid_filter = env_flag_or_default(
        "ANNFS_INTERNAL_XFSSLOWER_USE_PID_FILTER",
        env_flag_or_default("ANNFS_INTERNAL_XFS_USE_PID_FILTER", true));
    const bool read_only = env_int_or_default("ANNFS_INTERNAL_XFSSLOWER_READ_ONLY", 0) != 0;
    const std::string pid_opt = use_pid_filter ? (" -p " + std::to_string(target_pid)) : "";
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/xfsslower-bpfcc" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("xfsslower-bpfcc" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("/usr/share/bcc/tools/xfsslower" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("xfsslower" + pid_opt + " " + std::to_string(threshold_ms)),
    });
    const std::string run_expr_csv = run_first_available({
        unbuffered_cmd("/usr/sbin/xfsslower-bpfcc -j" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("xfsslower-bpfcc -j" + pid_opt + " " + std::to_string(threshold_ms)),
    });
    std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() + "' && ";
    if (read_only) {
        cmd += "((" + run_expr_csv +
               " || true) 2>&1 | awk -F',' 'NR==1 || $4==\"R\" {print}' > '" + report + "')";
    } else {
        cmd += "((" + run_expr + " || true) > '" + report + "' 2>&1)";
    }
    return std::make_unique<ShellCommandMonitor>("xfsslower", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_xfsdist_monitor(const std::string& policy_name,
                                                        const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_XFSDIST_REPORT",
        monitor_report_path("./xfsdist_report", "xfsdist", st, ds, run_id, ".log"));
    const long target_pid = static_cast<long>(std::max(
        1, env_int_or_default("ANNFS_INTERNAL_TARGET_PID", static_cast<int>(::getpid()))));
    const bool use_pid_filter = env_flag_or_default(
        "ANNFS_INTERNAL_XFSDIST_USE_PID_FILTER",
        env_flag_or_default("ANNFS_INTERNAL_XFS_USE_PID_FILTER", true));
    const std::string pid_opt = use_pid_filter ? (" -p " + std::to_string(target_pid)) : "";
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/xfsdist-bpfcc" + pid_opt),
        unbuffered_cmd("xfsdist-bpfcc" + pid_opt),
        unbuffered_cmd("/usr/share/bcc/tools/xfsdist" + pid_opt),
        unbuffered_cmd("xfsdist" + pid_opt),
    });
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && ((" + run_expr + " || true) > '" + report + "' 2>&1)";
    return std::make_unique<ShellCommandMonitor>("xfsdist", cmd, report);
}

std::unique_ptr<IInternalMonitor> build_xfssync_monitor(const std::string& policy_name,
                                                        const HnswIoCliArgs& args)
{
    const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
    const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
    const std::string run_id = now_compact_utc();
    const std::string report = env_or_default(
        "ANNFS_INTERNAL_XFSSYNC_REPORT",
        monitor_report_path("./xfssync_report", "xfssync", st, ds, run_id, ".log"));
    const int threshold_ms = std::max(1, env_int_or_default("ANNFS_INTERNAL_XFSSYNC_THRESHOLD_MS", 1));
    const long target_pid = static_cast<long>(std::max(
        1, env_int_or_default("ANNFS_INTERNAL_TARGET_PID", static_cast<int>(::getpid()))));
    const bool use_pid_filter = env_flag_or_default(
        "ANNFS_INTERNAL_XFSSYNC_USE_PID_FILTER",
        env_flag_or_default("ANNFS_INTERNAL_XFS_USE_PID_FILTER", true));
    const std::string pid_opt = use_pid_filter ? (" -p " + std::to_string(target_pid)) : "";
    const std::string run_expr_csv = run_first_available({
        unbuffered_cmd("/usr/sbin/xfsslower-bpfcc -j" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("xfsslower-bpfcc -j" + pid_opt + " " + std::to_string(threshold_ms)),
    });
    const std::string run_expr = run_first_available({
        unbuffered_cmd("/usr/sbin/xfsslower-bpfcc" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("xfsslower-bpfcc" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("/usr/share/bcc/tools/xfsslower" + pid_opt + " " + std::to_string(threshold_ms)),
        unbuffered_cmd("xfsslower" + pid_opt + " " + std::to_string(threshold_ms)),
    });
    const std::string cmd = "mkdir -p '" + fs::path(report).parent_path().string() +
                            "' && ((((" + run_expr_csv + ") || (" + run_expr + ")) || true) 2>&1 | "
                            "awk 'NR==1 || $4==\"S\" || $4==\"F\" {print}' > '" + report + "')";
    return std::make_unique<ShellCommandMonitor>("xfssync", cmd, report);
}

class InternalTraceCmdExt4Monitor : public IInternalMonitor {
public:
    InternalTraceCmdExt4Monitor(const HnswIoCliArgs& args,
                                const ParamConfig& pc,
                                const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_EXT4_REPORT",
            monitor_report_path("./trace_cmd_report", "trace_ext4", st, ds, run_id, ".metadata.txt"));
        trace_dat_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_EXT4_DAT",
            monitor_report_path("./trace_cmd_report", "trace_ext4", st, ds, run_id, ".dat"));
        trace_txt_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_EXT4_TXT",
            monitor_report_path("./trace_cmd_report", "trace_ext4", st, ds, run_id, ".report.txt"));
        record_log_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_EXT4_RECORD_LOG",
            monitor_report_path("./trace_cmd_report", "trace_ext4", st, ds, run_id, ".record.log"));
        report_log_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_EXT4_REPORT_LOG",
            monitor_report_path("./trace_cmd_report", "trace_ext4", st, ds, run_id, ".report.log"));
        device_path_guess_ = normalize_blktrace_device(
            env_or_default("ANNFS_INTERNAL_TRACE_EXT4_DEVICE", basename_device(pc.dev_path)));
    }

    const char* Name() const override { return "trace_ext4"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        fs::create_directories(fs::path(trace_dat_path_).parent_path(), ec);
        fs::remove(trace_dat_path_, ec);
        fs::remove(trace_txt_path_, ec);
        fs::remove(report_path_, ec);

        const std::string start_cmd =
            "(/usr/bin/trace-cmd reset >/dev/null 2>&1 || true) ; "
            "/usr/bin/trace-cmd start -e ext4 > '" +
            record_log_path_ + "' 2>&1";
        const int rc =
            ::system(("/bin/bash -lc \"" + start_cmd + "\"").c_str());
        if (rc != 0) {
            std::cerr << "[INTERNAL_TRACE_EXT4] trace-cmd start failed rc=" << rc << "\n";
            return false;
        }
        running_ = true;
        std::cout << "[INTERNAL_TRACE_EXT4] started dat=" << trace_dat_path_ << "\n";
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        const std::string stop_extract_cmd =
            "/usr/bin/trace-cmd stop >> '" + record_log_path_ + "' 2>&1 || true; "
            "/usr/bin/trace-cmd extract -o '" + trace_dat_path_ +
            "' >> '" + record_log_path_ + "' 2>&1 || true; "
            "/usr/bin/trace-cmd reset >> '" + record_log_path_ + "' 2>&1 || true";
        const int stop_rc =
            ::system(("/bin/bash -lc \"" + stop_extract_cmd + "\"").c_str());
        (void)stop_rc;
        running_ = false;

        const std::string convert_cmd =
            "/usr/bin/trace-cmd report -i '" + trace_dat_path_ + "' > '" + trace_txt_path_ +
            "' 2> '" + report_log_path_ + "' || true";
        const int convert_rc =
            ::system(("/bin/bash -lc \"" + convert_cmd + "\"").c_str());
        (void)convert_rc;
        const bool prefer_disk = prefer_disk_scope();
        const bool allow_global_fallback = fallback_to_global_scope();
        const bool strict_scope = strict_disk_scope();
        std::string grep_cmd =
            "prefer_disk=\"" + std::string(prefer_disk ? "1" : "0") + "\"; "
            "allow_fb=\"" + std::string(allow_global_fallback ? "1" : "0") + "\"; "
            "strict_scope=\"" + std::string(strict_scope ? "1" : "0") + "\"; "
            "target_mm=\"${ANNFS_INTERNAL_TARGET_DEV_MM:-}\"; "
            "if [ -z \"$target_mm\" ]; then "
            "  dev_path=\"${ANNFS_INTERNAL_TRACE_EXT4_DEVICE:-" + device_path_guess_ + "}\"; "
            "  dev_name=\"${dev_path##*/}\"; "
            "  if [ -n \"$dev_name\" ] && [ -r \"/sys/class/block/$dev_name/dev\" ]; then "
            "    target_mm=\"$(cat /sys/class/block/$dev_name/dev 2>/dev/null | tr -d [:space:])\"; "
            "  fi; "
            "fi; "
            "if [ -z \"$target_mm\" ]; then "
            "  target_mm=\"$(findmnt -no MAJ:MIN --target ${ANNFS_INTERNAL_TARGET_MOUNT:-/mnt/nvme0n1} 2>/dev/null | tr -d [:space:])\"; "
            "fi; "
            "target_dev=\"${target_mm//:/,}\"; "
            "if [ \"$prefer_disk\" = \"1\" ] && [ -n \"$target_dev\" ]; then "
            "  grep -E \"ext4_es_lookup_extent_enter|ext4_ext_map_blocks_enter\" \"" + trace_txt_path_ +
            "\" | grep -E \"dev[[:space:]]+$target_dev([[:space:]]|$)\" > \"" + report_path_ + "\" || true; "
            "  if [ ! -s \"" + report_path_ + "\" ]; then "
            "    if [ \"$allow_fb\" = \"1\" ] && [ \"$strict_scope\" != \"1\" ]; then "
            "      echo \"# scope=disk filter empty target_dev=$target_dev; fallback=global\" > \"" + report_path_ +
            "\"; "
            "      grep -E \"ext4_es_lookup_extent_enter|ext4_ext_map_blocks_enter\" \"" + trace_txt_path_ +
            "\" >> \"" + report_path_ + "\" || true; "
            "    else "
            "      echo \"# scope=disk filter empty target_dev=$target_dev; strict mode\" > \"" + report_path_ +
            "\"; "
            "    fi; "
            "  fi; "
            "elif [ \"$prefer_disk\" = \"1\" ]; then "
            "  if [ \"$allow_fb\" = \"1\" ] && [ \"$strict_scope\" != \"1\" ]; then "
            "    echo \"# scope=disk target unresolved; fallback=global\" > \"" + report_path_ + "\"; "
            "    grep -E \"ext4_es_lookup_extent_enter|ext4_ext_map_blocks_enter\" \"" + trace_txt_path_ +
            "\" >> \"" + report_path_ + "\" || true; "
            "  else "
            "    echo \"# scope=disk target unresolved; strict mode\" > \"" + report_path_ + "\"; "
            "  fi; "
            "else "
            "  grep -E \"ext4_es_lookup_extent_enter|ext4_ext_map_blocks_enter\" \"" + trace_txt_path_ +
            "\" > \"" + report_path_ + "\" || true; "
            "fi";
        const int grep_rc =
            ::system(("/bin/bash -lc '" + grep_cmd + "'").c_str());
        (void)grep_rc;
        if (!fs::exists(report_path_)) {
            std::ofstream out(report_path_, std::ios::out | std::ios::trunc);
            out << "# trace_ext4 metadata event extract unavailable\n";
        }
        std::cout << "[INTERNAL_TRACE_EXT4] REPORT_FILE=" << report_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_EXT4] REPORT_FILE=" << trace_dat_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_EXT4] REPORT_FILE=" << trace_txt_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_EXT4] REPORT_FILE=" << record_log_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_EXT4] REPORT_FILE=" << report_log_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_trace_ext4_report"] = report_path_;
        io_meta["internal_trace_ext4_dat"] = trace_dat_path_;
        io_meta["internal_trace_ext4_txt"] = trace_txt_path_;
        io_meta["internal_trace_ext4_record_log"] = record_log_path_;
        io_meta["internal_trace_ext4_report_log"] = report_log_path_;
    }

private:
    bool enabled_ = false;
    bool running_ = false;
    std::string device_path_guess_;
    std::string report_path_;
    std::string trace_dat_path_;
    std::string trace_txt_path_;
    std::string record_log_path_;
    std::string report_log_path_;
};

class InternalTraceCmdXfsMonitor : public IInternalMonitor {
public:
    InternalTraceCmdXfsMonitor(const HnswIoCliArgs& args,
                               const ParamConfig& pc,
                               const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_XFS_REPORT",
            monitor_report_path("./trace_cmd_report", "trace_xfs", st, ds, run_id, ".metadata.txt"));
        trace_dat_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_XFS_DAT",
            monitor_report_path("./trace_cmd_report", "trace_xfs", st, ds, run_id, ".dat"));
        trace_txt_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_XFS_TXT",
            monitor_report_path("./trace_cmd_report", "trace_xfs", st, ds, run_id, ".report.txt"));
        record_log_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_XFS_RECORD_LOG",
            monitor_report_path("./trace_cmd_report", "trace_xfs", st, ds, run_id, ".record.log"));
        report_log_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_XFS_REPORT_LOG",
            monitor_report_path("./trace_cmd_report", "trace_xfs", st, ds, run_id, ".report.log"));
        device_path_guess_ = normalize_blktrace_device(
            env_or_default("ANNFS_INTERNAL_TRACE_XFS_DEVICE", basename_device(pc.dev_path)));
    }

    const char* Name() const override { return "trace_xfs"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        fs::create_directories(fs::path(trace_dat_path_).parent_path(), ec);
        fs::remove(trace_dat_path_, ec);
        fs::remove(trace_txt_path_, ec);
        fs::remove(report_path_, ec);

        const std::string start_cmd =
            "(/usr/bin/trace-cmd reset >/dev/null 2>&1 || true) ; "
            "/usr/bin/trace-cmd start -e xfs > '" +
            record_log_path_ + "' 2>&1";
        const int rc =
            ::system(("/bin/bash -lc \"" + start_cmd + "\"").c_str());
        if (rc != 0) {
            std::cerr << "[INTERNAL_TRACE_XFS] trace-cmd start failed rc=" << rc << "\n";
            return false;
        }
        running_ = true;
        std::cout << "[INTERNAL_TRACE_XFS] started dat=" << trace_dat_path_ << "\n";
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        const std::string stop_extract_cmd =
            "/usr/bin/trace-cmd stop >> '" + record_log_path_ + "' 2>&1 || true; "
            "/usr/bin/trace-cmd extract -o '" + trace_dat_path_ +
            "' >> '" + record_log_path_ + "' 2>&1 || true; "
            "/usr/bin/trace-cmd reset >> '" + record_log_path_ + "' 2>&1 || true";
        const int stop_rc =
            ::system(("/bin/bash -lc \"" + stop_extract_cmd + "\"").c_str());
        (void)stop_rc;
        running_ = false;

        const std::string convert_cmd =
            "/usr/bin/trace-cmd report -i '" + trace_dat_path_ + "' > '" + trace_txt_path_ +
            "' 2> '" + report_log_path_ + "' || true";
        const int convert_rc =
            ::system(("/bin/bash -lc \"" + convert_cmd + "\"").c_str());
        (void)convert_rc;
        const bool prefer_disk = prefer_disk_scope();
        const bool allow_global_fallback = fallback_to_global_scope();
        const bool strict_scope = strict_disk_scope();
        std::string grep_cmd =
            "prefer_disk=\"" + std::string(prefer_disk ? "1" : "0") + "\"; "
            "allow_fb=\"" + std::string(allow_global_fallback ? "1" : "0") + "\"; "
            "strict_scope=\"" + std::string(strict_scope ? "1" : "0") + "\"; "
            "target_mm=\"${ANNFS_INTERNAL_TARGET_DEV_MM:-}\"; "
            "if [ -z \"$target_mm\" ]; then "
            "  dev_path=\"${ANNFS_INTERNAL_TRACE_XFS_DEVICE:-" + device_path_guess_ + "}\"; "
            "  dev_name=\"${dev_path##*/}\"; "
            "  if [ -n \"$dev_name\" ] && [ -r \"/sys/class/block/$dev_name/dev\" ]; then "
            "    target_mm=\"$(cat /sys/class/block/$dev_name/dev 2>/dev/null | tr -d [:space:])\"; "
            "  fi; "
            "fi; "
            "if [ -z \"$target_mm\" ]; then "
            "  target_mm=\"$(findmnt -no MAJ:MIN --target ${ANNFS_INTERNAL_TARGET_MOUNT:-/mnt/nvme0n1} 2>/dev/null | tr -d [:space:])\"; "
            "fi; "
            "target_dev=\"${target_mm//:/,}\"; "
            "if [ \"$prefer_disk\" = \"1\" ] && [ -n \"$target_dev\" ]; then "
            "  grep -E \"xfs_\" \"" + trace_txt_path_ +
            "\" | grep -E \"dev[[:space:]]+$target_dev([[:space:]]|$)\" > \"" + report_path_ + "\" || true; "
            "  if [ ! -s \"" + report_path_ + "\" ]; then "
            "    if [ \"$allow_fb\" = \"1\" ] && [ \"$strict_scope\" != \"1\" ]; then "
            "      echo \"# scope=disk filter empty target_dev=$target_dev; fallback=global\" > \"" + report_path_ +
            "\"; "
            "      grep -E \"xfs_\" \"" + trace_txt_path_ + "\" >> \"" + report_path_ + "\" || true; "
            "    else "
            "      echo \"# scope=disk filter empty target_dev=$target_dev; strict mode\" > \"" + report_path_ +
            "\"; "
            "    fi; "
            "  fi; "
            "elif [ \"$prefer_disk\" = \"1\" ]; then "
            "  if [ \"$allow_fb\" = \"1\" ] && [ \"$strict_scope\" != \"1\" ]; then "
            "    echo \"# scope=disk target unresolved; fallback=global\" > \"" + report_path_ + "\"; "
            "    grep -E \"xfs_\" \"" + trace_txt_path_ + "\" >> \"" + report_path_ + "\" || true; "
            "  else "
            "    echo \"# scope=disk target unresolved; strict mode\" > \"" + report_path_ + "\"; "
            "  fi; "
            "else "
            "  grep -E \"xfs_\" \"" + trace_txt_path_ + "\" > \"" + report_path_ + "\" || true; "
            "fi";
        const int grep_rc =
            ::system(("/bin/bash -lc '" + grep_cmd + "'").c_str());
        (void)grep_rc;
        if (!fs::exists(report_path_)) {
            std::ofstream out(report_path_, std::ios::out | std::ios::trunc);
            out << "# trace_xfs metadata event extract unavailable\n";
        }
        std::cout << "[INTERNAL_TRACE_XFS] REPORT_FILE=" << report_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_XFS] REPORT_FILE=" << trace_dat_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_XFS] REPORT_FILE=" << trace_txt_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_XFS] REPORT_FILE=" << record_log_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_XFS] REPORT_FILE=" << report_log_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_trace_xfs_report"] = report_path_;
        io_meta["internal_trace_xfs_dat"] = trace_dat_path_;
        io_meta["internal_trace_xfs_txt"] = trace_txt_path_;
        io_meta["internal_trace_xfs_record_log"] = record_log_path_;
        io_meta["internal_trace_xfs_report_log"] = report_log_path_;
    }

private:
    bool enabled_ = false;
    bool running_ = false;
    std::string device_path_guess_;
    std::string report_path_;
    std::string trace_dat_path_;
    std::string trace_txt_path_;
    std::string record_log_path_;
    std::string report_log_path_;
};

class InternalTraceMmPageCacheMonitor : public IInternalMonitor {
public:
    InternalTraceMmPageCacheMonitor(const HnswIoCliArgs& args,
                                    const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_MM_PAGECACHE_REPORT",
            monitor_report_path("./trace_cmd_report", "trace_mm_pagecache", st, ds, run_id, ".metadata.txt"));
        trace_dat_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_MM_PAGECACHE_DAT",
            monitor_report_path("./trace_cmd_report", "trace_mm_pagecache", st, ds, run_id, ".dat"));
        trace_txt_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_MM_PAGECACHE_TXT",
            monitor_report_path("./trace_cmd_report", "trace_mm_pagecache", st, ds, run_id, ".report.txt"));
        record_log_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_MM_PAGECACHE_RECORD_LOG",
            monitor_report_path("./trace_cmd_report", "trace_mm_pagecache", st, ds, run_id, ".record.log"));
        report_log_path_ = env_or_default(
            "ANNFS_INTERNAL_TRACE_MM_PAGECACHE_REPORT_LOG",
            monitor_report_path("./trace_cmd_report", "trace_mm_pagecache", st, ds, run_id, ".report.log"));
    }

    const char* Name() const override { return "trace_mm_pagecache"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        fs::create_directories(fs::path(trace_dat_path_).parent_path(), ec);
        fs::remove(trace_dat_path_, ec);
        fs::remove(trace_txt_path_, ec);
        fs::remove(report_path_, ec);

        const std::string start_cmd =
            "(/usr/bin/trace-cmd reset >/dev/null 2>&1 || true) ; "
            "(/usr/bin/trace-cmd start "
            "-e filemap:mm_filemap_add_to_page_cache "
            "-e filemap:mm_filemap_delete_from_page_cache "
            "-e vmscan > '" + record_log_path_ + "' 2>&1 "
            "|| /usr/bin/trace-cmd start "
            "-e mm_filemap_add_to_page_cache "
            "-e mm_filemap_delete_from_page_cache "
            "-e vmscan > '" + record_log_path_ + "' 2>&1)";
        const int rc =
            ::system(("/bin/bash -lc \"" + start_cmd + "\"").c_str());
        if (rc != 0) {
            std::cerr << "[INTERNAL_TRACE_MM_PAGECACHE] trace-cmd start failed rc=" << rc << "\n";
            return false;
        }
        running_ = true;
        std::cout << "[INTERNAL_TRACE_MM_PAGECACHE] started dat=" << trace_dat_path_ << "\n";
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        const std::string stop_extract_cmd =
            "/usr/bin/trace-cmd stop >> '" + record_log_path_ + "' 2>&1 || true; "
            "/usr/bin/trace-cmd extract -o '" + trace_dat_path_ +
            "' >> '" + record_log_path_ + "' 2>&1 || true; "
            "/usr/bin/trace-cmd reset >> '" + record_log_path_ + "' 2>&1 || true";
        const int stop_rc =
            ::system(("/bin/bash -lc \"" + stop_extract_cmd + "\"").c_str());
        (void)stop_rc;
        running_ = false;

        const std::string convert_cmd =
            "/usr/bin/trace-cmd report -i '" + trace_dat_path_ + "' > '" + trace_txt_path_ +
            "' 2> '" + report_log_path_ + "' || true";
        const int convert_rc =
            ::system(("/bin/bash -lc \"" + convert_cmd + "\"").c_str());
        (void)convert_rc;

        const auto filter_tids =
            parse_tid_filter_env("ANNFS_INTERNAL_TRACE_MM_PAGECACHE_FILTER_TIDS");
        {
            std::ifstream in(trace_txt_path_);
            std::ofstream out(report_path_, std::ios::out | std::ios::trunc);
            if (in.is_open() && out.is_open()) {
                std::string line;
                while (std::getline(in, line)) {
                    if (!is_trace_mm_pagecache_event_line(line)) {
                        continue;
                    }
                    if (!filter_tids.empty()) {
                        const long tid = extract_trace_tid_from_line(line);
                        if (tid <= 0 || filter_tids.find(tid) == filter_tids.end()) {
                            continue;
                        }
                    }
                    out << line << '\n';
                }
            }
        }
        if (!fs::exists(report_path_)) {
            std::ofstream out(report_path_, std::ios::out | std::ios::trunc);
            out << "# trace_mm_pagecache metadata event extract unavailable\n";
        }
        std::cout << "[INTERNAL_TRACE_MM_PAGECACHE] REPORT_FILE=" << report_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_MM_PAGECACHE] REPORT_FILE=" << trace_dat_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_MM_PAGECACHE] REPORT_FILE=" << trace_txt_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_MM_PAGECACHE] REPORT_FILE=" << record_log_path_ << "\n";
        std::cout << "[INTERNAL_TRACE_MM_PAGECACHE] REPORT_FILE=" << report_log_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_trace_mm_pagecache_report"] = report_path_;
        io_meta["internal_trace_mm_pagecache_dat"] = trace_dat_path_;
        io_meta["internal_trace_mm_pagecache_txt"] = trace_txt_path_;
        io_meta["internal_trace_mm_pagecache_record_log"] = record_log_path_;
        io_meta["internal_trace_mm_pagecache_report_log"] = report_log_path_;
    }

private:
    bool enabled_ = false;
    bool running_ = false;
    std::string report_path_;
    std::string trace_dat_path_;
    std::string trace_txt_path_;
    std::string record_log_path_;
    std::string report_log_path_;
};

class InternalMemInfoMonitor : public IInternalMonitor {
public:
    InternalMemInfoMonitor(const HnswIoCliArgs& args,
                           const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_MEMINFO_REPORT",
            monitor_report_path("./meminfo_report", "meminfo", st, ds, run_id, ".tsv"));
        interval_ms_ = std::max(10, env_int_or_default("ANNFS_INTERNAL_MEMINFO_INTERVAL_MS", 1000));
    }

    const char* Name() const override { return "meminfo"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        out_.open(report_path_, std::ios::out | std::ios::trunc);
        if (!out_.is_open()) {
            std::cerr << "[INTERNAL_MEMINFO] failed to open report: " << report_path_ << "\n";
            return false;
        }
        out_ << "ts_ms\tmem_total_kib\tmem_avail_kib\tcached_kib\tactive_file_kib\tinactive_file_kib\n";
        sample_once();
        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
                if (!running_) {
                    break;
                }
                sample_once();
            }
        });
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        sample_once();
        out_.flush();
        out_.close();
        std::cout << "[INTERNAL_MEMINFO] REPORT_FILE=" << report_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_meminfo_report"] = report_path_;
        io_meta["internal_meminfo_interval_ms"] = interval_ms_;
    }

private:
    void sample_once()
    {
        std::ifstream in("/proc/meminfo");
        if (!in.is_open()) {
            return;
        }
        long long total = 0;
        long long avail = 0;
        long long cached = 0;
        long long af = 0;
        long long inf = 0;
        std::string key;
        long long value = 0;
        std::string unit;
        while (in >> key >> value >> unit) {
            if (key == "MemTotal:") {
                total = value;
            } else if (key == "MemAvailable:") {
                avail = value;
            } else if (key == "Cached:") {
                cached = value;
            } else if (key == "Active(file):") {
                af = value;
            } else if (key == "Inactive(file):") {
                inf = value;
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (out_.is_open()) {
            out_ << fsrunner::epoch_ms() << "\t" << total << "\t" << avail << "\t" << cached << "\t" << af << "\t" << inf << "\n";
        }
    }

    bool enabled_ = false;
    std::atomic<bool> running_ {false};
    int interval_ms_ = 1000;
    std::string report_path_;
    std::ofstream out_;
    std::thread worker_;
    std::mutex mu_;
};

class InternalVmStatMonitor : public IInternalMonitor {
public:
    InternalVmStatMonitor(const HnswIoCliArgs& args,
                          const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_VMSTAT_REPORT",
            monitor_report_path("./vmstat_report", "vmstat", st, ds, run_id, ".tsv"));
        interval_ms_ = std::max(10, env_int_or_default("ANNFS_INTERNAL_VMSTAT_INTERVAL_MS", 1000));
    }

    const char* Name() const override { return "vmstat"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        out_.open(report_path_, std::ios::out | std::ios::trunc);
        if (!out_.is_open()) {
            std::cerr << "[INTERNAL_VMSTAT] failed to open report: " << report_path_ << "\n";
            return false;
        }
        out_ << "ts_ms\tpgscan_total\tpgsteal_total\tworkingset_refault_file\tpgmajfault\n";
        sample_once();
        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
                if (!running_) {
                    break;
                }
                sample_once();
            }
        });
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        sample_once();
        out_.flush();
        out_.close();
        std::cout << "[INTERNAL_VMSTAT] REPORT_FILE=" << report_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_vmstat_report"] = report_path_;
        io_meta["internal_vmstat_interval_ms"] = interval_ms_;
    }

private:
    void sample_once()
    {
        std::ifstream in("/proc/vmstat");
        if (!in.is_open()) {
            return;
        }
        long long pgscan = 0;
        long long pgsteal = 0;
        long long refault = 0;
        long long majflt = 0;
        std::string key;
        long long value = 0;
        while (in >> key >> value) {
            if (key.rfind("pgscan", 0) == 0) {
                pgscan += value;
            } else if (key.rfind("pgsteal", 0) == 0) {
                pgsteal += value;
            } else if (key == "workingset_refault_file") {
                refault = value;
            } else if (key == "pgmajfault") {
                majflt = value;
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (out_.is_open()) {
            out_ << fsrunner::epoch_ms() << "\t" << pgscan << "\t" << pgsteal
                 << "\t" << refault << "\t" << majflt << "\n";
        }
    }

    bool enabled_ = false;
    std::atomic<bool> running_ {false};
    int interval_ms_ = 1000;
    std::string report_path_;
    std::ofstream out_;
    std::thread worker_;
    std::mutex mu_;
};

class InternalPsiMonitor : public IInternalMonitor {
public:
    InternalPsiMonitor(const HnswIoCliArgs& args,
                       const std::string& policy_name)
    {
        enabled_ = true;
        const std::string ds = args.dataset_name.empty() ? "dataset" : args.dataset_name;
        const std::string st = policy_name.empty() ? "FsRunner" : policy_name;
        const std::string run_id = now_compact_utc();
        report_path_ = env_or_default(
            "ANNFS_INTERNAL_PSI_REPORT",
            monitor_report_path("./psi_report", "psi", st, ds, run_id, ".tsv"));
        interval_ms_ = std::max(100, env_int_or_default("ANNFS_INTERNAL_PSI_INTERVAL_MS", 1000));
    }

    const char* Name() const override { return "psi"; }
    bool Enabled() const override { return enabled_; }

    bool Start() override
    {
        if (!enabled_ || running_) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(fs::path(report_path_).parent_path(), ec);
        out_.open(report_path_, std::ios::out | std::ios::trunc);
        if (!out_.is_open()) {
            std::cerr << "[INTERNAL_PSI] failed to open report: " << report_path_ << "\n";
            return false;
        }
        out_ << "ts_ms\tmem_some_avg10\tmem_full_avg10\tio_some_avg10\tio_full_avg10\n";
        sample_once();
        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
                if (!running_) {
                    break;
                }
                sample_once();
            }
        });
        return true;
    }

    bool Stop() override
    {
        if (!enabled_ || !running_) {
            return true;
        }
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        sample_once();
        out_.flush();
        out_.close();
        std::cout << "[INTERNAL_PSI] REPORT_FILE=" << report_path_ << "\n";
        return true;
    }

    void EmitMeta(nlohmann::json& io_meta) const override
    {
        io_meta["internal_psi_report"] = report_path_;
        io_meta["internal_psi_interval_ms"] = interval_ms_;
    }

private:
    static double parse_avg10(const std::string& line)
    {
        const std::string key = "avg10=";
        auto pos = line.find(key);
        if (pos == std::string::npos) {
            return 0.0;
        }
        pos += key.size();
        auto end = line.find(' ', pos);
        std::string token = line.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        try {
            return std::stod(token);
        } catch (...) {
            return 0.0;
        }
    }

    static std::pair<double, double> read_psi_pair(const char* path)
    {
        std::ifstream in(path);
        if (!in.is_open()) {
            return {0.0, 0.0};
        }
        std::string line;
        double some = 0.0;
        double full = 0.0;
        while (std::getline(in, line)) {
            if (line.rfind("some ", 0) == 0) {
                some = parse_avg10(line);
            } else if (line.rfind("full ", 0) == 0) {
                full = parse_avg10(line);
            }
        }
        return {some, full};
    }

    void sample_once()
    {
        auto [mem_some, mem_full] = read_psi_pair("/proc/pressure/memory");
        auto [io_some, io_full] = read_psi_pair("/proc/pressure/io");
        std::lock_guard<std::mutex> lk(mu_);
        if (out_.is_open()) {
            out_ << fsrunner::epoch_ms() << "\t"
                 << mem_some << "\t" << mem_full << "\t"
                 << io_some << "\t" << io_full << "\n";
        }
    }

    bool enabled_ = false;
    std::atomic<bool> running_ {false};
    int interval_ms_ = 1000;
    std::string report_path_;
    std::ofstream out_;
    std::thread worker_;
    std::mutex mu_;
};

} // namespace

bool HasInternalMonitorSelectionEnv()
{
    return !parse_monitor_kinds().empty();
}

std::vector<std::unique_ptr<IInternalMonitor>> BuildInternalMonitors(
    const HnswIoCliArgs& args,
    const ParamConfig& pc,
    const std::string& policy_name)
{
    std::vector<std::unique_ptr<IInternalMonitor>> monitors;
    const auto kinds = parse_monitor_kinds();
    if (kinds.empty()) {
        return monitors;
    }

    if (kind_selected(kinds, "blktrace")) {
        monitors.emplace_back(std::make_unique<InternalFsBlktraceMonitor>(args, pc, policy_name));
    }
    if (kind_selected(kinds, "rss")) {
        monitors.emplace_back(std::make_unique<InternalRssMonitor>(args, policy_name));
    }
    if (kind_selected(kinds, "perf")) {
        monitors.emplace_back(build_perf_monitor(args, policy_name));
    }
    if (kind_selected(kinds, "mem_access")) {
        monitors.emplace_back(build_mem_access_monitor(args, policy_name));
    }
    if (kind_selected(kinds, "mem_imc")) {
        monitors.emplace_back(build_mem_imc_monitor(args, policy_name));
    }
    if (kind_selected(kinds, "mem_imc_0")) {
        monitors.emplace_back(build_mem_imc0_monitor(args, policy_name));
    }
    if (kind_selected(kinds, "numa_node")) {
        monitors.emplace_back(build_numa_node_monitor(args, policy_name));
    }
    if (kind_selected(kinds, "pcm_memory")) {
        monitors.emplace_back(build_pcm_memory_monitor(args, policy_name));
    }
    if (kind_selected(kinds, "pidstat")) {
        monitors.emplace_back(build_pidstat_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "pidstat_r")) {
        monitors.emplace_back(build_pidstat_r_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "pidstat_d")) {
        monitors.emplace_back(build_pidstat_d_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "iostat")) {
        monitors.emplace_back(build_iostat_monitor(policy_name, args, pc));
    }
    if (kind_selected(kinds, "meminfo")) {
        monitors.emplace_back(std::make_unique<InternalMemInfoMonitor>(args, policy_name));
    }
    if (kind_selected(kinds, "vmstat")) {
        monitors.emplace_back(std::make_unique<InternalVmStatMonitor>(args, policy_name));
    }
    if (kind_selected(kinds, "psi")) {
        monitors.emplace_back(std::make_unique<InternalPsiMonitor>(args, policy_name));
    }
    if (kind_selected(kinds, "strace")) {
        monitors.emplace_back(build_strace_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "perf_fs")) {
        monitors.emplace_back(build_perf_fs_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "perf_sched")) {
        monitors.emplace_back(build_perf_sched_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "numastat")) {
        monitors.emplace_back(build_numastat_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "dcstat")) {
        monitors.emplace_back(build_dcstat_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "cachestat")) {
        monitors.emplace_back(build_cachestat_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "vfsstat")) {
        monitors.emplace_back(build_vfsstat_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "biosnoop")) {
        monitors.emplace_back(build_biosnoop_monitor(policy_name, args, pc));
    }
    if (kind_selected(kinds, "biolatency")) {
        monitors.emplace_back(build_biolatency_monitor(policy_name, args, pc));
    }
    if (kind_selected(kinds, "ext4slower")) {
        monitors.emplace_back(build_ext4slower_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "ext4dist")) {
        monitors.emplace_back(build_ext4dist_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "xfsslower")) {
        monitors.emplace_back(build_xfsslower_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "xfsdist")) {
        monitors.emplace_back(build_xfsdist_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "xfssync")) {
        monitors.emplace_back(build_xfssync_monitor(policy_name, args));
    }
    if (kind_selected(kinds, "trace_ext4")) {
        monitors.emplace_back(std::make_unique<InternalTraceCmdExt4Monitor>(args, pc, policy_name));
    }
    if (kind_selected(kinds, "trace_xfs")) {
        monitors.emplace_back(std::make_unique<InternalTraceCmdXfsMonitor>(args, pc, policy_name));
    }
    if (kind_selected(kinds, "trace_mm_pagecache")) {
        monitors.emplace_back(std::make_unique<InternalTraceMmPageCacheMonitor>(args, policy_name));
    }

    return monitors;
}

} // namespace annfs::tests
