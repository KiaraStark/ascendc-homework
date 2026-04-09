/**
* @file operator_desc.cpp
*/
#include "common.h"
#include "operator_desc.h"

OperatorDesc::OperatorDesc() : dim(-1) {}

OperatorDesc::~OperatorDesc()
{
    for (auto *desc : inputDesc) {
        (void)aclDestroyTensorDesc(desc);
    }

    for (auto *desc : outputDesc) {
        (void)aclDestroyTensorDesc(desc);
    }
}

OperatorDesc &OperatorDesc::AddInputTensorDesc(aclDataType dataType,
                                               int numDims,
                                               const int64_t *dims,
                                               aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create input tensor desc failed");
        return *this;
    }
    inputDesc.emplace_back(desc);
    return *this;
}

OperatorDesc &OperatorDesc::AddOutputTensorDesc(aclDataType dataType,
                                                int numDims,
                                                const int64_t *dims,
                                                aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create output tensor desc failed");
        return *this;
    }
    outputDesc.emplace_back(desc);
    return *this;
}
