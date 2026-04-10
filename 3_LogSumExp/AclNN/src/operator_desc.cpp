/**
* @file operator_desc.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "common.h"
#include "operator_desc.h"

OperatorDesc::OperatorDesc() : keepdim(false) {}

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
