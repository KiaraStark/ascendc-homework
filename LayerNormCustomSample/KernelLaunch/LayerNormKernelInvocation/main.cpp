/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void layer_norm_custom_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* gamma, uint8_t* beta, uint8_t* z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void layer_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,GM_ADDR z);
#endif


int32_t main(int32_t argc, char* argv[])
{
    uint32_t blockDim = 48;
    size_t inputSize_x = 16384 * 1024 * sizeof(float);
    size_t outputSize_z = 16384 * 1024 * sizeof(float);
    size_t inputSize_gamma = 1024 * sizeof(float);
    size_t inputSize_beta = 1024 * sizeof(float);
    
#ifdef ASCENDC_CPU_DEBUG
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize_x);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputSize_gamma);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputSize_beta);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputSize_z);

    ReadFile("./input/input_x.bin", inputSize_x, x, inputSize_x);
    ReadFile("./input/input_gamma.bin", inputSize_gamma, gamma, inputSize_gamma);
    ReadFile("./input/input_beta.bin", inputSize_beta, beta, inputSize_beta);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(layer_norm_custom, blockDim, x, gamma, beta, z); // use this macro for cpu debug

    WriteFile("./output/output_z.bin", z, outputSize_z);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)beta);
    AscendC::GmFree((void *)z);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *zHost, *gammaHost, *betaHost;
    uint8_t *xDevice, *zDevice, *gammaDevice, *betaDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputSize_x));
    CHECK_ACL(aclrtMallocHost((void**)(&gammaHost), inputSize_gamma));
    CHECK_ACL(aclrtMallocHost((void**)(&betaHost), inputSize_beta));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputSize_z));
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputSize_x, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&gammaDevice, inputSize_gamma, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&betaDevice, inputSize_beta, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputSize_z, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputSize_x, xHost, inputSize_x);
    ReadFile("./input/input_gamma.bin", inputSize_gamma, gammaHost, inputSize_gamma);
    ReadFile("./input/input_beta.bin", inputSize_beta, betaHost, inputSize_beta);

    CHECK_ACL(aclrtMemcpy(xDevice, inputSize_x, xHost, inputSize_x, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(gammaDevice, inputSize_gamma, gammaHost, inputSize_gamma, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(betaDevice, inputSize_beta, betaHost, inputSize_beta, ACL_MEMCPY_HOST_TO_DEVICE));

    layer_norm_custom_do(blockDim, nullptr, stream, xDevice, gammaDevice, betaDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputSize_z, zDevice, outputSize_z, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputSize_z);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(gammaDevice));
    CHECK_ACL(aclrtFree(betaDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(gammaHost));
    CHECK_ACL(aclrtFreeHost(betaHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
