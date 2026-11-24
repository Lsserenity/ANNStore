# Include from the end of LabTest/CMakeLists.txt after defining add_annfs_executable.
add_annfs_executable(HnswL1LayoutPacked_repair
    ../core/impl/HnswL1LayoutPacked_repair.cpp ../core/config/param_config.cpp)
add_executable(TailPacking_repair_test ../core/tests/TailPacking_repair_test.cpp)