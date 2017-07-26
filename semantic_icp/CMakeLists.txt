set (LIB_NAME semantic_icp)

file (GLOB HEADER_FILES *.h *.hpp)

add_library(semantic_icp_lib INTERFACE)
target_include_directories(semantic_icp_lib INTERFACE
    ${PROJECT_SOURCE_DIR}/semantic_icp)
