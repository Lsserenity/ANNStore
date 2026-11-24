#pragma once

#include <memory>
#include <string>
#include <vector>

#include <json.hpp>

#include "tests/HnswIoCommon.hpp"

namespace annfs::tests {

class IInternalMonitor {
public:
    virtual ~IInternalMonitor() = default;
    virtual const char* Name() const = 0;
    virtual bool Enabled() const = 0;
    virtual bool Start() = 0;
    virtual bool Stop() = 0;
    virtual void EmitMeta(nlohmann::json& io_meta) const = 0;
};

bool HasInternalMonitorSelectionEnv();

std::vector<std::unique_ptr<IInternalMonitor>> BuildInternalMonitors(
    const HnswIoCliArgs& args,
    const ParamConfig& pc,
    const std::string& policy_name);

} // namespace annfs::tests
