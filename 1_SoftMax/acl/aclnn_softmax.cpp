#include "aclnn_softmax_custom.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#include <stdio.h>
#include <stdlib.h>
#include "data_utils.h"
#include <unistd.h>
#include <string>

aclrtStream CreateStream(const int32_t device)
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        printf("acl init failed\n");
        return nullptr;
    }
    if (aclrtSetDevice(device) != ACL_SUCCESS) {
        printf("Set device failed\n");
        CHECK_ACL(aclFinalize());
        return nullptr;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        printf("Create stream failed\n");
        CHECK_ACL(aclFinalize());
        return nullptr;
    }
    return stream;
}

void DestroyStream(aclrtStream stream, const int32_t device)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    if (aclrtResetDevice(device) != ACL_SUCCESS) {
        printf("Reset device failed\n");
    }
    if (aclFinalize() != ACL_SUCCESS) {
        printf("Finalize acl failed\n");
    }
}

void DestroyTensor(aclTensor* tensors[], void* devMem[], const int32_t tensorCount)
{
    for (auto i = 0; i < tensorCount; i++) {
        if (!tensors[i]) {
            continue;
        }
        if (devMem[i]) {
            CHECK_ACL(aclrtFree(devMem[i]));
        }
        CHECK_ACL(aclDestroyTensor(tensors[i]));
    }
}

struct tensorInfo {
    int64_t* dims;
    int64_t dimCnt;
    aclDataType dtype;
    aclFormat fmt;
};

int64_t GetDataSize(const struct tensorInfo* desc)
{
    if (!desc->dims) {
        return 0;
    }
    int64_t size = 1;
    for (auto i = 0; i < desc->dimCnt; i++) {
        size *= desc->dims[i];
    }
    return size * sizeof(float);
}

static std::string SelectPath(const char* primary, const char* fallback)
{
    return (access(primary, F_OK) == 0) ? std::string(primary) : std::string(fallback);
}

static std::string SelectOutputPath(const char* fileName)
{
    const char* primaryDir = "../test/output";
    const char* fallbackDir = "../output";
    const char* outDir = (access(primaryDir, F_OK) == 0) ? primaryDir : fallbackDir;
    return std::string(outDir) + "/" + fileName;
}

static int64_t CompareResult(void* outputData, const int64_t outSize)
{
    void* goldenData;
    CHECK_ACL(aclrtMallocHost((void**)(&goldenData), outSize));
    size_t goldenSize = outSize;
    std::string goldenPath = SelectPath("../test/output/golden.bin", "../output/golden.bin");
    bool ret = ReadFile(goldenPath, goldenSize, goldenData, goldenSize);
    if (ret) {
        printf("ReadFile golden success!\n");
    } else {
        CHECK_ACL(aclrtFreeHost(goldenData));
        return -1;
    }
    constexpr float EPS = 1e-5;
    int64_t wrongNum = 0;

    for (auto i = 0; i < outSize / sizeof(float); i++) {
        float a = ((float*)outputData)[i];
        float b = ((float*)goldenData)[i];
        float ae = std::abs(a - b);
        float re = ae / abs(b);
        if (ae > EPS && re > EPS) {
            printf("CompareResult failed output is %lf, golden is %lf\n", a, b);
            wrongNum++;
        }
    }
    CHECK_ACL(aclrtFreeHost(goldenData));
    return wrongNum;
}

int32_t main(void)
{
    int64_t inputX[] = { 960, 960 };
    int64_t outputMax[] = { 960, 8 };
    int64_t outputSum[] = { 960, 8 };
    int64_t outputZ[] = { 960, 960 };
    const struct tensorInfo tensorDesc[] = {
        { inputX, 2, ACL_FLOAT, ACL_FORMAT_ND },
        { outputMax, 2, ACL_FLOAT, ACL_FORMAT_ND },
        { outputSum, 2, ACL_FLOAT, ACL_FORMAT_ND },
        { outputZ, 2, ACL_FLOAT, ACL_FORMAT_ND },
    };
    aclrtStream stream = CreateStream(0);
    if (stream == nullptr) {
        return -1;
    }
    const int32_t tensorCount = sizeof(tensorDesc) / sizeof(struct tensorInfo);
    aclTensor* tensors[tensorCount];
    void* devMem[tensorCount];
    for (auto i = 0; i < tensorCount; i++) {
        void* data;
        const struct tensorInfo* info = &(tensorDesc[i]);
        int64_t size = GetDataSize(info);
        if (size == 0) {
            tensors[i] = nullptr;
            devMem[i] = nullptr;
            continue;
        }
        CHECK_ACL(aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST));
        // Allocate host memory and read input data, then copy to device memory
        if (i == 0) {
            size_t inputSize = size;
            void* dataHost;
            CHECK_ACL(aclrtMallocHost((void**)(&dataHost), inputSize));
            std::string inputPath = SelectPath("../test/input/input_x.bin", "../input/input_x.bin");
              ReadFile(inputPath, inputSize, dataHost, inputSize);
            CHECK_ACL(aclrtMemcpy(data, size, dataHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
            CHECK_ACL(aclrtFreeHost(dataHost));
        }
        devMem[i] = data;
        tensors[i] = aclCreateTensor(info->dims, info->dimCnt, info->dtype, nullptr, 0, info->fmt, info->dims,
                                     info->dimCnt, data);
    }
    size_t workspaceSize = 0;
    aclOpExecutor* handle;
    // Tensor order is input0, output0, output1, output2
    int32_t ret =
        aclnnSoftmaxCustomGetWorkspaceSize(tensors[0], tensors[1], tensors[2], tensors[3], &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        printf("aclnnSoftmaxCustomGetWorkspaceSize failed. error code is %d\n", ret);
        DestroyTensor(tensors, devMem, tensorCount);
        DestroyStream(stream, 0);
        return ret;
    }
    printf("aclnnSoftmaxCustomGetWorkspaceSize ret %d workspace size %lu\n", ret, workspaceSize);
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ret = aclnnSoftmaxCustom(workspace, workspaceSize, handle, stream);
    printf("aclnnSoftmaxCustom ret %d\n", ret);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Allocate host memory, copy result to host memory and write to output file
    uint8_t *zHost, *maxHost, *sumHost;
    int64_t maxHostSize = GetDataSize(&(tensorDesc[1]));
    int64_t sumHostSize = GetDataSize(&(tensorDesc[2]));
    int64_t zHostSize = GetDataSize(&(tensorDesc[3]));

    CHECK_ACL(aclrtMallocHost((void**)(&maxHost), maxHostSize));
    CHECK_ACL(aclrtMallocHost((void**)(&sumHost), sumHostSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), zHostSize));

    CHECK_ACL(aclrtMemcpy(maxHost, maxHostSize, devMem[1], maxHostSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(sumHost, sumHostSize, devMem[2], sumHostSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(zHost, zHostSize, devMem[3], zHostSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string outZ = SelectOutputPath("output_z.bin");
    WriteFile(outZ, zHost, zHostSize);
    std::string outMax = SelectOutputPath("output_max.bin");
    WriteFile(outMax, maxHost, maxHostSize);
    std::string outSum = SelectOutputPath("output_sum.bin");
    WriteFile(outSum, sumHost, sumHostSize);

    int64_t wrongNum = CompareResult(zHost, zHostSize);
    if (wrongNum != 0) {
        printf("test failed!\n");
    } else {
        printf("test pass!\n");
    }
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(maxHost));
    CHECK_ACL(aclrtFreeHost(sumHost));

    DestroyTensor(tensors, devMem, tensorCount);
    DestroyStream(stream, 0);
    return 0;
}