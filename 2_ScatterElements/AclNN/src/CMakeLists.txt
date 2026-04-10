cmake_minimum_required(VERSION 3.5.1)

project(scatter_elements_acl_runner)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../output")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "DDK_PATH is not set, use default: ${INC_PATH}")
else ()
    message(STATUS "Use DDK_PATH from env: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "${INC_PATH}/acllib/lib64/stub")
    set(LIB_PATH1 "${INC_PATH}/atc/lib64/stub")
    message(STATUS "NPU_HOST_LIB is not set, use default stub libs")
else ()
    message(STATUS "Use NPU_HOST_LIB from env: ${LIB_PATH}")
endif ()

# Prefer this operator vendor package path, fallback to generic customize path.
set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/scatter_elements/op_api")
if (NOT EXISTS "${CUST_PKG_PATH}")
    set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
endif ()

set(ACL_DEMO_REQUIRED_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/../inc/op_runner.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../inc/common.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../inc/operator_desc.h"
)

set(ACL_DEMO_REQUIRED_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/scatter_elements_main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/operator_desc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common.cpp"
)

foreach(file_path IN LISTS ACL_DEMO_REQUIRED_HEADERS ACL_DEMO_REQUIRED_SOURCES)
    if (NOT EXISTS "${file_path}")
        message(WARNING "Skip ACL demo build: missing file ${file_path}")
        return()
    endif ()
endforeach()

include_directories(
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CUST_PKG_PATH}/include
)

link_directories(
    ${LIB_PATH}
    ${LIB_PATH1}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_scatter_elements_acl
    scatter_elements_main.cpp
    operator_desc.cpp
    op_runner.cpp
    common.cpp
)

target_link_libraries(execute_scatter_elements_acl
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_scatter_elements_acl DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
