cmake_minimum_required(VERSION 3.16.0)
project(softmax_ascendc)

set(ASCEND_HOME_PATH /usr/local/Ascend/cann-9.0.0-beta.2)
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
set(ASCEND_PYTHON_EXECUTABLE python3)

# Include AscendC cmake modules
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/config.cmake)
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake)
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/intf.cmake)

# Include ASC modules
list(APPEND CMAKE_MODULE_PATH ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc_modules)
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc_modules/FindASC.cmake)

# Set build mode
set(BUILD_MODE ascend910b)

# Add include directories
set(INCLUDES
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api/tiling/platform
)

# Kernel source files
set(DEVICE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/device/softmax_kernel.cpp)
set(HOST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/host/softmax_host.cpp)

# Create ascendc library for device code
ascendc_library(softmax_kernel STATIC
    ${DEVICE_FILES}
)

ascendc_include_directories(softmax_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device
    ${INCLUDES}
)

# Host library
set_source_files_properties(${HOST_FILES} PROPERTIES LANGUAGE ASC)

add_library(softmax_ascendc SHARED
    ${HOST_FILES}
)

target_include_directories(softmax_ascendc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/include/experiment
    ${ASCEND_HOME_PATH}/aarch64-linux/include
)

target_link_libraries(softmax_ascendc PRIVATE
    -Wl,--whole-archive softmax_kernel
    -Wl,--no-whole-archive
    c_sec
    ascendalog
    platform
    tiling_api
)

target_compile_options(softmax_ascendc PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=ascend910b> -O2 -fno-common -Wextra -Wfloat-equal
    -fvisibility=default -DLOG_CPP -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=
)

target_compile_options(softmax_kernel PRIVATE
    -DHAVE_TILING
    -DAUTO_FUSE_DEVICE=1
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=ascend910b>
)
