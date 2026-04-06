set(CANN_ROOT ${ASCEND_CANN_PACKAGE_PATH})
set(CANN_AARCH64 ${CANN_ROOT}/aarch64-linux)
set(OPP_VENDOR_LIB_DIR ${CANN_ROOT}/opp/vendors/${vendor_name}/op_api/lib)
set(OPP_VENDOR_INC_DIR ${CANN_ROOT}/opp/vendors/${vendor_name}/op_api/include)

add_executable(run_softmax_custom aclnn_softmax.cpp)

target_include_directories(run_softmax_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANN_AARCH64}/include
    ${OPP_VENDOR_INC_DIR}
)

target_link_directories(run_softmax_custom PRIVATE
    ${CANN_ROOT}/lib64
    ${CANN_AARCH64}/lib64
    ${OPP_VENDOR_LIB_DIR}
)

target_link_libraries(run_softmax_custom PRIVATE
    acl_rt
    acl_op_compiler
    acl_op_executor
    nnopbase
    cust_opapi
)

target_link_options(run_softmax_custom PRIVATE
    -Wl,--disable-new-dtags
)

set(_RUNPATHS
    ${CANN_ROOT}/lib64
    ${CANN_AARCH64}/lib64
    ${OPP_VENDOR_LIB_DIR}
)

set_target_properties(run_softmax_custom PROPERTIES
    BUILD_RPATH "${_RUNPATHS}"
    INSTALL_RPATH "${_RUNPATHS}"
    # Put caller binary in source acl/ for direct ./run_softmax_custom usage.
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/acl
)
