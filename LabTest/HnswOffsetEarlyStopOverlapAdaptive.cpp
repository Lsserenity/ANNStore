#include <string>

#include "tests/HnswAdaptiveOverlap.hpp"
#include "tests/HnswExperimentRunner.hpp"

int main(int argc, char** argv)
{
    auto args_opt = annfs::tests::ParseCliArgs(argc, argv);
    if (!args_opt) {
        annfs::tests::PrintAdaptiveOverlapUsage(argv[0]);
        return 1;
    }
    annfs::tests::AdaptiveEarlyStopOverlapIOStrategy strategy;
    return annfs::tests::RunHnswExperiment(*args_opt, strategy);
}
