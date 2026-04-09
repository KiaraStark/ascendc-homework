#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    // Match 2_ScatterElements/gen_scatter_elements_data.py defaults.
    std::vector<int64_t> shapeVar {8, 16, 32};
    std::vector<int64_t> shapeIndices {8, 16, 32};

    aclDataType dataType = ACL_FLOAT16;
    aclDataType indicesType = ACL_INT32;
    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.axis = 1;
    opDesc.reduce = (char*)"None";
    opDesc.AddInputTensorDesc(dataType, shapeVar.size(), shapeVar.data(), format);
    opDesc.AddInputTensorDesc(indicesType, shapeIndices.size(), shapeIndices.data(), format);
    opDesc.AddInputTensorDesc(dataType, shapeIndices.size(), shapeIndices.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shapeVar.size(), shapeVar.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("./2_ScatterElements/input/var.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("./2_ScatterElements/input/indices.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("./2_ScatterElements/input/updates.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("./2_ScatterElements/output/output.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "./2_ScatterElements/output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
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
    INFO_LOG("Set device[%d] success", deviceId);

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    OperatorDesc opDesc = CreateOpDesc();

    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();
    return SUCCESS;
}
