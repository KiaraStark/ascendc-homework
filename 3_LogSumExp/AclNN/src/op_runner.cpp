/**
* @file op_runner.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "op_runner.h"

#include <cassert>
#include <iomanip>

#include "aclnn_log_sum_exp.h"

extern bool g_isDevice;

OpRunner::OpRunner(OperatorDesc *opDesc) : numInputs_(0), numOutputs_(0), opDesc_(opDesc)
{
    if (opDesc_ != nullptr) {
        numInputs_ = opDesc_->inputDesc.size();
        numOutputs_ = opDesc_->outputDesc.size();
    }
}

OpRunner::~OpRunner()
{
    for (size_t i = 0; i < inputTensor_.size(); ++i) {
        (void)aclDestroyTensor(inputTensor_[i]);
    }
    for (size_t i = 0; i < outputTensor_.size(); ++i) {
        (void)aclDestroyTensor(outputTensor_[i]);
    }
    for (size_t i = 0; i < inputBuffers_.size(); ++i) {
        (void)aclDestroyDataBuffer(inputBuffers_[i]);
    }
    for (size_t i = 0; i < outputBuffers_.size(); ++i) {
        (void)aclDestroyDataBuffer(outputBuffers_[i]);
    }
    for (size_t i = 0; i < devInputs_.size(); ++i) {
        (void)aclrtFree(devInputs_[i]);
    }
    for (size_t i = 0; i < devOutputs_.size(); ++i) {
        (void)aclrtFree(devOutputs_[i]);
    }
    for (size_t i = 0; i < hostInputs_.size(); ++i) {
        if (g_isDevice) {
            (void)aclrtFree(hostInputs_[i]);
        } else {
            (void)aclrtFreeHost(hostInputs_[i]);
        }
    }
    for (size_t i = 0; i < hostOutputs_.size(); ++i) {
        if (g_isDevice) {
            (void)aclrtFree(hostOutputs_[i]);
        } else {
            (void)aclrtFreeHost(hostOutputs_[i]);
        }
    }
}

bool OpRunner::Init()
{
    if (opDesc_ == nullptr) {
        ERROR_LOG("opDesc is null");
        return false;
    }

    for (size_t i = 0; i < numInputs_; ++i) {
        const auto size = GetInputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return false;
        }
        devInputs_.emplace_back(devMem);

        aclDataBuffer *inputBuf = aclCreateDataBuffer(devMem, size);
        if (inputBuf == nullptr) {
            ERROR_LOG("Create input data buffer[%zu] failed", i);
            return false;
        }
        inputBuffers_.emplace_back(inputBuf);

        void *hostInput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostInput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc host(device) memory for input[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostInput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc host memory for input[%zu] failed", i);
                return false;
            }
        }
        hostInputs_.emplace_back(hostInput);

        std::vector<int64_t> shape = GetInputShape(i);
        aclTensor *inputTensor = aclCreateTensor(shape.data(), GetInputNumDims(i), GetInputDataType(i),
            nullptr, 0, GetInputFormat(i), shape.data(), GetInputNumDims(i), devInputs_[i]);
        if (inputTensor == nullptr) {
            ERROR_LOG("Create tensor for input[%zu] failed", i);
            return false;
        }
        inputTensor_.emplace_back(inputTensor);
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        const auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return false;
        }
        devOutputs_.emplace_back(devMem);

        aclDataBuffer *outputBuf = aclCreateDataBuffer(devMem, size);
        if (outputBuf == nullptr) {
            ERROR_LOG("Create output data buffer[%zu] failed", i);
            return false;
        }
        outputBuffers_.emplace_back(outputBuf);

        void *hostOutput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostOutput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc host(device) memory for output[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostOutput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc host memory for output[%zu] failed", i);
                return false;
            }
        }
        hostOutputs_.emplace_back(hostOutput);

        std::vector<int64_t> shape = GetOutputShape(i);
        aclTensor *outputTensor = aclCreateTensor(shape.data(), GetOutputNumDims(i), GetOutputDataType(i),
            nullptr, 0, GetOutputFormat(i), shape.data(), GetOutputNumDims(i), devOutputs_[i]);
        if (outputTensor == nullptr) {
            ERROR_LOG("Create tensor for output[%zu] failed", i);
            return false;
        }
        outputTensor_.emplace_back(outputTensor);
    }

    return true;
}

const size_t OpRunner::NumInputs() { return numInputs_; }
const size_t OpRunner::NumOutputs() { return numOutputs_; }

const size_t OpRunner::GetInputSize(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }
    return aclGetTensorDescSize(opDesc_->inputDesc[index]);
}

const size_t OpRunner::GetInputNumDims(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }
    return aclGetTensorDescNumDims(opDesc_->inputDesc[index]);
}

aclDataType OpRunner::GetInputDataType(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_DT_UNDEFINED;
    }
    return aclGetTensorDescType(opDesc_->inputDesc[index]);
}

aclFormat OpRunner::GetInputFormat(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_FORMAT_UNDEFINED;
    }
    return aclGetTensorDescFormat(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputSize(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }
    return aclGetTensorDescSize(opDesc_->outputDesc[index]);
}

const size_t OpRunner::GetOutputNumDims(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }
    return aclGetTensorDescNumDims(opDesc_->outputDesc[index]);
}

aclDataType OpRunner::GetOutputDataType(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_DT_UNDEFINED;
    }
    return aclGetTensorDescType(opDesc_->outputDesc[index]);
}

aclFormat OpRunner::GetOutputFormat(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_FORMAT_UNDEFINED;
    }
    return aclGetTensorDescFormat(opDesc_->outputDesc[index]);
}

size_t OpRunner::GetInputElementCount(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }
    return aclGetTensorDescElementCount(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputElementCount(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }
    return aclGetTensorDescElementCount(opDesc_->outputDesc[index]);
}

std::vector<int64_t> OpRunner::GetInputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ret;
    }

    aclTensorDesc *desc = opDesc_->inputDesc[index];
    const size_t numDims = aclGetTensorDescNumDims(desc);
    ret.reserve(numDims);
    for (size_t i = 0; i < numDims; ++i) {
        int64_t dimSize = 0;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get input dim failed. dim index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }
    return ret;
}

std::vector<int64_t> OpRunner::GetOutputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ret;
    }

    aclTensorDesc *desc = opDesc_->outputDesc[index];
    const size_t numDims = aclGetTensorDescNumDims(desc);
    ret.reserve(numDims);
    for (size_t i = 0; i < numDims; ++i) {
        int64_t dimSize = 0;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get output dim failed. dim index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }
    return ret;
}

bool OpRunner::CompileStaticOp()
{
    WARN_LOG("CompileStaticOp is not used in aclnn demo");
    return true;
}

bool OpRunner::CompileDynamicOp()
{
    WARN_LOG("CompileDynamicOp is not used in aclnn demo");
    return true;
}

bool OpRunner::RunOp()
{
    if (numInputs_ != 1 || numOutputs_ != 1) {
        ERROR_LOG("LogSumExp expects 1 input and 1 output, got %zu inputs, %zu outputs", numInputs_, numOutputs_);
        return false;
    }

    for (size_t i = 0; i < numInputs_; ++i) {
        const auto size = GetInputSize(i);
        aclrtMemcpyKind kind = g_isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE;
        if (aclrtMemcpy(devInputs_[i], size, hostInputs_[i], size, kind) != ACL_SUCCESS) {
            ERROR_LOG("copy input[%zu] failed", i);
            return false;
        }
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("create stream failed");
        return false;
    }

    aclIntArray *dimArray = nullptr;
    if (!opDesc_->dim.empty()) {
        dimArray = aclCreateIntArray(opDesc_->dim.data(), opDesc_->dim.size());
        if (dimArray == nullptr) {
            ERROR_LOG("create dim array failed");
            (void)aclrtDestroyStream(stream);
            return false;
        }
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnLogSumExpGetWorkspaceSize(
        inputTensor_[0], dimArray, opDesc_->keepdim, outputTensor_[0], &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnLogSumExpGetWorkspaceSize failed. ret=%d", static_cast<int32_t>(ret));
        if (dimArray != nullptr) {
            aclDestroyIntArray(dimArray);
        }
        (void)aclrtDestroyStream(stream);
        return false;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("malloc workspace failed, size=%lu", workspaceSize);
            if (dimArray != nullptr) {
                aclDestroyIntArray(dimArray);
            }
            (void)aclrtDestroyStream(stream);
            return false;
        }
    }

    ret = aclnnLogSumExp(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnLogSumExp failed. ret=%d", static_cast<int32_t>(ret));
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
        }
        if (dimArray != nullptr) {
            aclDestroyIntArray(dimArray);
        }
        (void)aclrtDestroyStream(stream);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("synchronize stream failed. ret=%d", static_cast<int32_t>(ret));
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
        }
        if (dimArray != nullptr) {
            aclDestroyIntArray(dimArray);
        }
        (void)aclrtDestroyStream(stream);
        return false;
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        const auto size = GetOutputSize(i);
        aclrtMemcpyKind kind = g_isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
        if (aclrtMemcpy(hostOutputs_[i], size, devOutputs_[i], size, kind) != ACL_SUCCESS) {
            ERROR_LOG("copy output[%zu] failed", i);
            if (workspace != nullptr) {
                (void)aclrtFree(workspace);
            }
            if (dimArray != nullptr) {
                aclDestroyIntArray(dimArray);
            }
            (void)aclrtDestroyStream(stream);
            return false;
        }
    }

    if (workspace != nullptr) {
        (void)aclrtFree(workspace);
    }
    if (dimArray != nullptr) {
        aclDestroyIntArray(dimArray);
    }
    (void)aclrtDestroyStream(stream);
    return true;
}

template<typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void DoPrintFp16Data(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << std::setprecision(4) << aclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void PrintData(const void *data, size_t count, aclDataType dataType, size_t elementsPerRow)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case ACL_BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case ACL_INT8:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT8:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT16:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT16:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT32:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT32:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT64:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT64:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT16:
            DoPrintFp16Data(reinterpret_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case ACL_DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
            break;
    }
}

void OpRunner::PrintInput(size_t index, size_t numElementsPerRow)
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numInputs_);
        return;
    }

    auto desc = opDesc_->inputDesc[index];
    PrintData(hostInputs_[index], GetInputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}

void OpRunner::PrintOutput(size_t index, size_t numElementsPerRow)
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return;
    }

    auto desc = opDesc_->outputDesc[index];
    PrintData(hostOutputs_[index], GetOutputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}
