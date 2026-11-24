#include <iostream>
#include <string>

#include "tests/HnswExperimentRunner.hpp"
#include "tests/HnswExperimentStrategies.hpp"

int main(int argc, char** argv)
{

    auto args_opt = annfs::tests::ParseCliArgs(argc, argv);
    if (!args_opt) {
        return 1;
    }
    annfs::tests::OffsetIOStrategy strategy;
    return annfs::tests::RunHnswExperiment(*args_opt, strategy);
}
