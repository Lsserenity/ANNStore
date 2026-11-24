#pragma once

#include <string>

#include <json.hpp>

#include "tests/HnswExperimentBatch.hpp"
#include "tests/HnswFsDataset.hpp"
#include "tests/HnswIoCommon.hpp"
#include "tests/HnswFsIoPolicies.hpp"

namespace annfs::tests {

struct FsUnifiedRunOptions {
    bool enable_flops_count = false;
};

int RunFsUnifiedWithResources(FsDatasetResources& resources,
                              const HnswIoCliArgs& args,
                              const ExperimentBatchRunSpec* spec,
                              const nlohmann::json& root,
                              IoExecutionPolicy& policy,
                              const FsUnifiedRunOptions& options);

} // namespace annfs::tests
