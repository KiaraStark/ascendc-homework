cmake_minimum_required(VERSION 3.16.0)
project(softmax_ascendc)

# AscendC package path - can be overridden
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.0.0-beta.2)
endif()

# Compiler - use AscendC compiler if available, otherwise use system g++
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_C_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler/bin/ccec)
    set(CMAKE_CXX_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler/bin/ccec)
endif()

# Add include directories for AscendC
set(ASCEND_INC
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/basic_api/interface
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/impl/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/impl/utils
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include
)

include_directories(${ASCEND_INC})

# Compiler flags for AscendC
if(CMAKE_CXX_COMPILER MATCHES "ccec")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cce-aicore-lang -std=c++17")
endif()

# Kernel source files
set(KERNEL_SOURCES
    softmax_kernel.cpp
)

# Create static library for kernel
add_library(softmax_kernel_lib STATIC ${KERNEL_SOURCES})

# Also build simplified version for testing
set(SIMPLIFIED_SOURCES
    softmax_kernel_simple.cpp
)

add_library(softmax_kernel_simple_lib STATIC ${SIMPLIFIED_SOURCES})

# Install targets
install(TARGETS softmax_kernel_lib softmax_kernel_simple_lib
        ARCHIVE DESTINATION lib)
install(FILES
        softmax_kernel.h
        softmax_kernel_simple.h
        DESTINATION include)

# Print configuration info
message(STATUS "AscendC CANN Package Path: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
