# --- Root CMakeLists.txt ---
cmake_minimum_required(VERSION 3.22)
project(ANNFS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native") # 统一优化参数

include(CTest)
option(ANNFS_PACKING_TESTS_ONLY "Build dependency-free packing tests only" OFF)
if(BUILD_TESTING)
    add_executable(TailPacking_test core/tests/TailPacking_test.cpp)
    add_test(NAME TailPacking COMMAND TailPacking_test)
endif()
if(ANNFS_PACKING_TESTS_ONLY)
    return()
endif()
# 1. 查找系统依赖库
find_library(PTHREAD pthread)
find_library(URING uring)
find_library(NUMA numa)
find_library(RT rt)

if(NOT PTHREAD OR NOT URING OR NOT NUMA)
    message(FATAL_ERROR "Missing core dependencies (pthread, uring, or numa)")
endif()

# 2. 定义全局变量方便子目录使用
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/external)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/core)

# 3. 添加子目录
add_subdirectory(core)  # 先编译公共库
add_subdirectory(LabTest) # 再编译实验可执行文件
