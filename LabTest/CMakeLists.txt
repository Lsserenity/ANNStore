# Build only programs whose source is part of the anonymous artifact.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/execs)

set(HNSW_EXPERIMENT_COMMON
    ../core/tests/HnswExperimentRunner.cpp
    ../core/tests/HnswExperimentStrategies.cpp
    ../core/tests/HnswExperimentBatch.cpp
    ../core/tests/HnswInternalMonitors.cpp
    ../core/tests/HnswIoCommon.cpp
    ../core/config/param_config.cpp
)

set(HNSW_FS_COMMON
    ../core/tests/HnswFsDataset.cpp
    ../core/tests/HnswFsIoPolicies.cpp
    ../core/tests/HnswFsUnifiedRunner.cpp
    ../core/tests/HnswFsEntryPoints.cpp
)

function(add_annfs_executable target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE annfs)
endfunction()

function(add_hnsw_experiment target source)
    add_annfs_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE hnsw_experiment_support)
endfunction()

add_library(hnsw_experiment_support STATIC ${HNSW_EXPERIMENT_COMMON})
target_link_libraries(hnsw_experiment_support PUBLIC annfs)

add_library(hnsw_fs_support STATIC ${HNSW_FS_COMMON})
target_link_libraries(hnsw_fs_support PUBLIC hnsw_experiment_support)

# Raw-device ANNStore experiments.
add_hnsw_experiment(HnswOffsetIO HnswOffsetIO.cpp)
add_hnsw_experiment(HnswOffsetEarlyStop HnswOffsetEarlyStop.cpp)
add_hnsw_experiment(HnswOffsetOverlapAdaptive HnswOffsetOverlapAdaptive.cpp)
add_hnsw_experiment(HnswOffsetEarlyStopOverlapAdaptive HnswOffsetEarlyStopOverlapAdaptive.cpp)
add_hnsw_experiment(HnswMappedOffsetIO ../core/impl/HnswMappedOffsetOverlapAdaptiveMonitor.cpp)

# File-system baselines.
add_annfs_executable(HnswFsIO ../FSLabTest/HnswFsIo.cpp)
target_link_libraries(HnswFsIO PRIVATE hnsw_fs_support)
add_annfs_executable(HnswFsOpenOnly ../FSLabTest/HnswFsOpenOnly.cpp)
target_link_libraries(HnswFsOpenOnly PRIVATE hnsw_fs_support)

# Index, layout, and verification utilities.
add_annfs_executable(IndexBuild ../core/impl/IndexBuild.cpp)
add_annfs_executable(FsPrepareFilesTrace ../core/impl/FsPrepareFilesTrace.cpp)
add_annfs_executable(HnswRelabelIndex ../core/impl/HnswRelabelIndex.cpp ../core/config/param_config.cpp)
add_annfs_executable(HnswTraceBlockVerify ../core/impl/HnswTraceBlockVerify.cpp)
add_annfs_executable(HnswL1LayoutBase ../core/impl/HnswL1LayoutBase.cpp ../core/config/param_config.cpp)
add_annfs_executable(HnswL1LayoutPacked ../core/impl/HnswL1LayoutPacked.cpp ../core/config/param_config.cpp)

