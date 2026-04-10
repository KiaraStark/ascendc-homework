/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "kernel_operator.h"
#include <cmath>

constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

class KernelScatterSubInt8Aligned {
public:
    __aicore__ inline KernelScatterSubInt8Aligned() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, uint32_t alignNum,
        uint32_t lastDim, uint32_t indicesLength, uint32_t var1stDim, uint32_t firstTiling)
    {        
        this->tileNum = indicesLength;
        this->indicesLength = indicesLength;
        this->lastDim = lastDim;
        this->var1stDim = var1stDim;
        this->firstTiling = firstTiling;

        varGm.SetGlobalBuffer((__gm__ int8_t*)var, var1stDim * lastDim);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, indicesLength);
        updatesGm.SetGlobalBuffer((__gm__ int8_t*)updates , indicesLength * lastDim);

        pipe.InitBuffer(inQueueVar, BUFFER_NUM, firstTiling * sizeof(int8_t));
        pipe.InitBuffer(varBuf, firstTiling * sizeof(half));
        pipe.InitBuffer(inQueueUpdates, BUFFER_NUM,  firstTiling * sizeof(int8_t));
        pipe.InitBuffer(updatesBuf, firstTiling * sizeof(half));
        pipe.InitBuffer(outQueueVar, BUFFER_NUM, firstTiling * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t total = lastDim;
            uint32_t count = lastDim / firstTiling;
            uint32_t left = lastDim % firstTiling;
            for (int ii = 0; ii < count ;ii++) {
                CopyIn(i, ii, firstTiling);
                Compute(i, ii, firstTiling);
                CopyOut(i, ii, firstTiling);
            }
            if (left !=0) {
                CopyIn(i, count, left);
                Compute(i, count, left);
                CopyOut(i, count, left);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t lastDimProgress, uint32_t tilingNum)
    {
        AscendC::LocalTensor<int8_t> updatesLocal = inQueueUpdates.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> varLocal = inQueueVar.AllocTensor<int8_t>();

        AscendC::DataCopy(updatesLocal, updatesGm[progress * lastDim + lastDimProgress * firstTiling], tilingNum);
        int32_t indice = indicesGm.GetValue(progress);
        AscendC::DataCopy(varLocal, varGm[indice * lastDim + lastDimProgress * firstTiling], tilingNum);

        inQueueUpdates.EnQue(updatesLocal);
        inQueueVar.EnQue(varLocal);
    }
    __aicore__ inline void Compute(int32_t progress, int32_t lastDimProgress, uint32_t tilingNum)
    {
        AscendC::LocalTensor<int8_t> updatesLocal = inQueueUpdates.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> varLocal = inQueueVar.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> outVarLocal = outQueueVar.AllocTensor<int8_t>();
       
        AscendC::LocalTensor<half> varBufLocal = varBuf.Get<half>();
        AscendC::LocalTensor<half> updatesBufLocal = updatesBuf.Get<half>();
        
        AscendC::Cast(varBufLocal, varLocal, AscendC::RoundMode::CAST_NONE, tilingNum);
        AscendC::Cast(updatesBufLocal, updatesLocal, AscendC::RoundMode::CAST_NONE, tilingNum);

        AscendC::Sub(varBufLocal, varBufLocal, updatesBufLocal, tilingNum);
        AscendC::Cast(outVarLocal, varBufLocal, AscendC::RoundMode::CAST_ROUND, tilingNum);
        
        inQueueUpdates.FreeTensor(updatesLocal);
        inQueueVar.FreeTensor(varLocal);
        outQueueVar.EnQue(outVarLocal);
    }


    __aicore__ inline void CopyOut(int32_t progress, int32_t lastDimProgress, uint32_t tilingNum)
    {
        AscendC::LocalTensor<int8_t> outVarLocal = outQueueVar.DeQue<int8_t>();
        int32_t indice = indicesGm.GetValue(progress);
        AscendC::DataCopy(varGm[indice * lastDim + lastDimProgress * firstTiling], outVarLocal, tilingNum);
        outQueueVar.FreeTensor(outVarLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueUpdates;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueVar;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueVar;

    AscendC::TBuf<AscendC::TPosition::VECCALC> varBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> updatesBuf;

    AscendC::GlobalTensor<int8_t> varGm;
    AscendC::GlobalTensor<int8_t> updatesGm;
    AscendC::GlobalTensor<int32_t> indicesGm;

    uint32_t tileNum;

    uint32_t indicesLength;
    uint32_t lastDim;
    uint32_t var1stDim;
    uint32_t firstTiling;
};


extern "C" __global__ __aicore__ void scatter_sub( GM_ADDR var, GM_ADDR indices, GM_ADDR updates)
{
    KernelScatterSubInt8Aligned op;
    uint32_t alignNum = 32;
    uint32_t lastDim = 2304;
    uint32_t indicesLength = 3;
    uint32_t var1stDim = 3;
    uint32_t firstTiling = 2304;
    op.Init(var, indices, updates, alignNum,
        lastDim, indicesLength, var1stDim, firstTiling);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void scatter_sub_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* var, uint8_t* indices, uint8_t* updates)
{
    scatter_sub<<<coreDim, l2ctrl, stream>>>(var, indices, updates);
}
#endif