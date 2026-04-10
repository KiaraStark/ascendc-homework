/**
* @file main.cpp
*/
#include <cstdint>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

bool g_isDevice = false;
int deviceId = 0;

static aclDataType ParseDataType(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dtype" && (i + 1) < argc) {
            std::string dtype = argv[i + 1];
            if (dtype == "fp16") {
                return ACL_FLOAT16;
            }
            if (dtype == "fp32") {
                return ACL_FLOAT;
            }
            ERROR_LOG("Unsupported dtype: %s", dtype.c_str());
            return ACL_DT_UNDEFINED;
        }
    }
    return ACL_FLOAT;
}

static OperatorDesc CreateOpDesc(aclDataType dataType)
{
    // Match AclNN/scripts/gen_softmax_data.py defaults: shape=(64, 256).
    const std::vector<int64_t> shapeX {64, 256};
    const std::vector<int64_t> shapeMS {64, 8};
    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataType, shapeX.size(), shapeX.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shapeMS.size(), shapeMS.data(), format); // max
    opDesc.AddOutputTensorDesc(dataType, shapeMS.size(), shapeMS.data(), format); // sum
    opDesc.AddOutputTensorDesc(dataType, shapeX.size(), shapeX.data(), format);   // z
    return opDesc;
}

static bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    if (!ReadFile("../scripts/input/input_x.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0))) {
        ERROR_LOG("Read input data failed");
        return false;
    }
    INFO_LOG("Set input success");
    return true;
}

static bool ProcessOutputData(OpRunner &runner)
{
    bool ok = true;
    ok = ok && WriteFile("../scripts/output/output_max.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    ok = ok && WriteFile("../scripts/output/output_sum.bin", runner.GetOutputBuffer<void>(1), runner.GetOutputSize(1));
    ok = ok && WriteFile("../scripts/output/output.bin", runner.GetOutputBuffer<void>(2), runner.GetOutputSize(2));
    if (!ok) {
        ERROR_LOG("Write output failed");
        return false;
    }
    INFO_LOG("Write output success");
    return true;
}

static void DestroyResource()
{
    bool failed = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        failed = true;
    }

    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        failed = true;
    }

    if (failed) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

static bool InitResource()
{
    std::string outputDir = "../scripts/output";
    if (access(outputDir.c_str(), 0) == -1) {
        if (mkdir(outputDir.c_str(), 0700) != 0) {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);
    return true;
}

static bool RunOp(aclDataType dataType)
{
    OperatorDesc opDesc = CreateOpDesc(dataType);
    OpRunner runner(&opDesc);

    if (!runner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    if (!SetInputData(runner)) {
        return false;
    }

    if (!runner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    if (!ProcessOutputData(runner)) {
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    aclDataType dataType = ParseDataType(argc, argv);
    if (dataType == ACL_DT_UNDEFINED) {
        return FAILED;
    }

    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }

    if (!RunOp(dataType)) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();
    return SUCCESS;
}
