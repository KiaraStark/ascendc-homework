#include <cstdint>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

struct RunConfig {
    aclDataType dataType = ACL_FLOAT16;
    int64_t axis = 1;
    std::string reduce = "None";
    std::vector<int64_t> shapeVar {8, 16, 32};
};

static bool ParseArgs(int argc, char **argv, RunConfig &cfg)
{
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--dtype" && (i + 1) < argc) {
            std::string dtype = argv[++i];
            if (dtype == "fp16") {
                cfg.dataType = ACL_FLOAT16;
            } else if (dtype == "fp32") {
                cfg.dataType = ACL_FLOAT;
            } else if (dtype == "int32") {
                cfg.dataType = ACL_INT32;
            } else if (dtype == "uint8") {
                cfg.dataType = ACL_UINT8;
            } else {
                ERROR_LOG("Unsupported dtype: %s", dtype.c_str());
                return false;
            }
        } else if (key == "--axis" && (i + 1) < argc) {
            cfg.axis = std::stoll(argv[++i]);
        } else if (key == "--reduce" && (i + 1) < argc) {
            std::string reduce = argv[++i];
            if (reduce == "none" || reduce == "None") {
                cfg.reduce = "None";
            } else if (reduce == "add") {
                cfg.reduce = "add";
            } else if (reduce == "multiply") {
                cfg.reduce = "multiply";
            } else {
                ERROR_LOG("Unsupported reduce: %s", reduce.c_str());
                return false;
            }
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: ./execute_op [--dtype fp16|fp32|int32|uint8] [--axis int] [--reduce none|add|multiply]" << std::endl;
            return false;
        } else {
            ERROR_LOG("Unknown argument: %s", key.c_str());
            return false;
        }
    }
    return true;
}

OperatorDesc CreateOpDesc(const RunConfig &cfg)
{
    std::vector<int64_t> shapeIndices = cfg.shapeVar;
    aclDataType indicesType = ACL_INT32;
    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.axis = cfg.axis;
    opDesc.reduce = const_cast<char *>(cfg.reduce.c_str());
    opDesc.AddInputTensorDesc(cfg.dataType, cfg.shapeVar.size(), cfg.shapeVar.data(), format);
    opDesc.AddInputTensorDesc(indicesType, shapeIndices.size(), shapeIndices.data(), format);
    opDesc.AddInputTensorDesc(cfg.dataType, shapeIndices.size(), shapeIndices.data(), format);
    // ScatterElements is in-place on varRef; no extra output tensor is required.
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    if (!ReadFile("./input/var.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0))) {
        return false;
    }
    if (!ReadFile("./input/indices.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1))) {
        return false;
    }
    if (!ReadFile("./input/updates.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2))) {
        return false;
    }
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    // Updated varRef is copied back into input buffer 0 by the runner.
    if (!WriteFile("./output/output.bin", runner.GetInputBuffer<void>(0), runner.GetInputSize(0))) {
        return false;
    }
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
    std::string output = "./output";
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
        ERROR_LOG("Get RunMode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp(const RunConfig &cfg)
{
    OperatorDesc opDesc = CreateOpDesc(cfg);

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
    RunConfig cfg;
    if (!ParseArgs(argc, argv, cfg)) {
        return FAILED;
    }

    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp(cfg)) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();
    return SUCCESS;
}
