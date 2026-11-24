#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../io/raw_trace_block_writer.hpp"

namespace {

struct Args {
    std::string dev_path;
    std::string manifest_path;
    std::size_t verify_count = 64;  // 0 => all
};

Args parse_args(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error("Usage: <dev_path> <manifest_path> [verify_count]");
    }
    Args a;
    a.dev_path = argv[1];
    a.manifest_path = argv[2];
    if (argc >= 4) a.verify_count = static_cast<std::size_t>(std::stoull(argv[3]));
    return a;
}

std::vector<annfs::io::TraceWriteRecord> load_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open manifest: " + path);
    }

    std::vector<annfs::io::TraceWriteRecord> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        annfs::io::TraceWriteRecord r;
        unsigned crossed = 0;
        if (!(iss >> r.source_label
                  >> r.start_byte
                  >> r.payload_size
                  >> r.end_offset_in_block
                  >> crossed
                  >> r.packed_label)) {
            throw std::runtime_error("bad manifest line: " + line);
        }
        r.crossed_blocks = static_cast<std::uint8_t>(crossed);
        out.push_back(r);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto records = load_manifest(args.manifest_path);
        if (records.empty()) {
            std::cout << "Manifest has no records.\n";
            return 0;
        }

        const auto res = annfs::io::verify_trace_blocks_from_device(
            args.dev_path, records, args.verify_count);

        std::cout << "Device        : " << args.dev_path << "\n";
        std::cout << "Manifest      : " << args.manifest_path << "\n";
        std::cout << "Checked       : " << res.checked << "\n";
        std::cout << "Passed        : " << res.passed << "\n";
        std::cout << "Failed        : " << res.failed << "\n";
        return (res.failed == 0) ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

