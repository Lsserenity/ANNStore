# --- core/CMakeLists.txt ---

add_library(annfs STATIC
    common/fs_path_mapper.cpp
)

target_include_directories(annfs PUBLIC
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/core/common
    ${CMAKE_SOURCE_DIR}/core/config
    ${CMAKE_SOURCE_DIR}/core/hnsw
    ${CMAKE_SOURCE_DIR}/core/tests
    ${CMAKE_SOURCE_DIR}/core/io
    ${THIRD_PARTY_DIR}
    ${THIRD_PARTY_DIR}/hnswlib
    ${THIRD_PARTY_DIR}/spdlog/include
)

target_link_libraries(annfs PUBLIC
    ${PTHREAD}
    ${URING}
    ${NUMA}
    ${RT}
)

if(SPDLOG_BUILD_EXAMPLE_HO)
    target_link_libraries(annfs PUBLIC spdlog::spdlog_header_only)
endif()
