#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

struct RunConfig {
    aclDataType dataType = ACL_FLOAT16;
    std::vector<int64_t> shape {8, 16, 32};
    std::vector<int64_t> outShape {16, 32};
    std::vector<int64_t> dim {0};
    bool keepdim = false;
};

static bool ParseBool(const std::string &text, bool &value)
{
    if (text == "1" || text == "true" || text == "True" || text == "TRUE") {
        value = true;
        return true;
    }
    if (text == "0" || text == "false" || text == "False" || text == "FALSE") {
        value = false;
        return true;
    }
    return false;
}

static bool NormalizeDims(const std::vector<int64_t> &dims, int64_t rank, std::vector<int64_t> &normalized)
{
    normalized.clear();
    if (dims.empty()) {
        ERROR_LOG("--dim cannot be empty");
        return false;
    }

    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t d = dims[i];
        if (d < -rank || d >= rank) {
            ERROR_LOG("dim out of range: %ld (rank=%ld)", static_cast<long>(d), static_cast<long>(rank));
            return false;
        }
        if (d < 0) {
            d += rank;
        }
        if (std::find(normalized.begin(), normalized.end(), d) == normalized.end()) {
            normalized.push_back(d);
        }
    }

    std::sort(normalized.begin(), normalized.end());
    return true;
}

static std::vector<int64_t> BuildOutputShape(const std::vector<int64_t> &inShape,
                                             const std::vector<int64_t> &reduceDims,
                                             bool keepdim)
{
    std::vector<int64_t> out;
    const int64_t rank = static_cast<int64_t>(inShape.size());
    if (keepdim) {
        out = inShape;
        for (size_t i = 0; i < reduceDims.size(); ++i) {
            out[reduceDims[i]] = 1;
        }
        return out;
    }

    out.reserve(inShape.size());
    for (int64_t i = 0; i < rank; ++i) {
        if (std::find(reduceDims.begin(), reduceDims.end(), i) == reduceDims.end()) {
            out.push_back(inShape[i]);
        }
    }
    if (out.empty()) {
        out.push_back(1);
    }
    return out;
}

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
            } else {
                ERROR_LOG("Unsupported dtype: %s", dtype.c_str());
                return false;
            }
        } else if (key == "--dim") {
            cfg.dim.clear();
            while ((i + 1) < argc) {
                std::string maybe = argv[i + 1];
                if (!maybe.empty() && maybe[0] == '-') {
                    break;
                }
                cfg.dim.push_back(std::stoll(argv[++i]));
            }
            if (cfg.dim.empty()) {
                ERROR_LOG("--dim requires at least one int64 value");
                return false;
            }
        } else if (key == "--keep_dim") {
            cfg.keepdim = true;
            if ((i + 1) < argc) {
                std::string maybe = argv[i + 1];
                if (!maybe.empty() && maybe[0] != '-') {
                    bool keep = false;
                    if (!ParseBool(maybe, keep)) {
                        ERROR_LOG("Invalid --keep_dim value: %s", maybe.c_str());
                        return false;
                    }
                    cfg.keepdim = keep;
                    ++i;
                }
            }
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: ./execute_log_sum_exp_acl [--dtype fp16|fp32] [--dim d0 d1 ...] [--keep_dim [true|false]]" << std::endl;
            return false;
        } else {
            ERROR_LOG("Unknown argument: %s", key.c_str());
            return false;
        }
    }

    std::vector<int64_t> normalized;
    if (!NormalizeDims(cfg.dim, static_cast<int64_t>(cfg.shape.size()), normalized)) {
        return false;
    }
    cfg.dim = normalized;
    cfg.outShape = BuildOutputShape(cfg.shape, cfg.dim, cfg.keepdim);
    return true;
}

OperatorDesc CreateOpDesc(const RunConfig &cfg)
{
    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.dim = cfg.dim;
    opDesc.keepdim = cfg.keepdim;
    opDesc.AddInputTensorDesc(cfg.dataType, cfg.shape.size(), cfg.shape.data(), format);
    opDesc.AddOutputTensorDesc(cfg.dataType, cfg.outShape.size(), cfg.outShape.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    if (!ReadFile("./script/input/input_x.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0))) {
        return false;
    }
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    if (!WriteFile("./script/output/output.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0))) {
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
    std::string output = "./script/output";
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
