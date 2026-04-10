/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void scatter_sub_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* var, uint8_t* indices, uint8_t* updates);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void scatter_sub( GM_ADDR var, GM_ADDR indices, GM_ADDR updates);
#endif

int32_t main(int32_t argc, char* argv[])
{
    uint32_t blockDim = 1;
    size_t varByteSize = 3 * 4 * 24 * 24 * sizeof(uint32_t) / 4;
    size_t indicesByteSize = 3 * sizeof(uint32_t);
    size_t updatesByteSize = 3 * 4 * 24 * 24 * sizeof(uint32_t) / 4;;
    
#ifdef ASCENDC_CPU_DEBUG
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);

    ReadFile("./input/input_var.bin", varByteSize, var, varByteSize);
    ReadFile("./input/input_indices.bin", indicesByteSize, indices, indicesByteSize);
    ReadFile("./input/input_updates.bin", updatesByteSize, updates, updatesByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF(scatter_sub, blockDim, var, indices, updates); // use this macro for cpu debug

    WriteFile("./output/output_y.bin", var, varByteSize);

    AscendC::GmFree((void *)var);
    AscendC::GmFree((void *)indices);
    AscendC::GmFree((void *)updates);
#else
    
    CHECK_ACL(aclInit("./scripts/acl.json"));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // SpenceTilingData* tiling;
    uint8_t *varHost, *indicesHost, *updatesHost;
    uint8_t *varDevice, *indicesDevice, *updatesDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&varHost), varByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&indicesHost), indicesByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&updatesHost), updatesByteSize));

    CHECK_ACL(aclrtMalloc((void**)&varDevice, varByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&indicesDevice, indicesByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&updatesDevice, updatesByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_var.bin", varByteSize, varHost, varByteSize);
    ReadFile("./input/input_indices.bin", indicesByteSize, indicesHost, indicesByteSize);
    ReadFile("./input/input_updates.bin", updatesByteSize, updatesHost, updatesByteSize);

    CHECK_ACL(aclrtMemcpy(varDevice, varByteSize, varHost, varByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(indicesDevice, indicesByteSize, indicesHost, indicesByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(updatesDevice, updatesByteSize, updatesHost, updatesByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    
    scatter_sub_do(blockDim, nullptr, stream, varDevice, indicesDevice, updatesDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(varHost, varByteSize, varDevice, varByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", varHost, varByteSize);

    CHECK_ACL(aclrtFree(varDevice));
    CHECK_ACL(aclrtFree(indicesDevice));
    CHECK_ACL(aclrtFree(updatesDevice));
    CHECK_ACL(aclrtFreeHost(varHost));
    CHECK_ACL(aclrtFreeHost(indicesHost));
    CHECK_ACL(aclrtFreeHost(updatesHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
