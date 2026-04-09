cmake_minimum_required(VERSION 3.5.1)

project(softmax_aclnn_runner)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "DDK_PATH is not set, use default: ${INC_PATH}")
else ()
    message(STATUS "Use DDK_PATH from env: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "${INC_PATH}/aarch64-linux/lib64")
    set(LIB_PATH1 "${INC_PATH}/aarch64-linux/devlib/linux/x86_64")
    message(STATUS "NPU_HOST_LIB is not set, use default: ${LIB_PATH}")
else ()
    message(STATUS "Use NPU_HOST_LIB from env: ${LIB_PATH}")
endif ()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/softmax/op_api")
if (NOT EXISTS "${CUST_PKG_PATH}")
    set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
endif ()

include_directories(
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUST_PKG_PATH}/include
)

link_directories(
    ${LIB_PATH}
    ${LIB_PATH1}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_softmax_aclnn
    src/main.cpp
    src/operator_desc.cpp
    src/op_runner.cpp
    src/common.cpp
)

target_link_libraries(execute_softmax_aclnn
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_softmax_aclnn DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
