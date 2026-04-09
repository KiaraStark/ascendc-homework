#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

template<typename TYPE_X, typename TYPE_Y> class KernelAsinh {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, 
                                uint32_t ALIGN_NUM, uint32_t block_size, 
                                uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<TYPE_X, half>){
            pipe.InitBuffer(QueueTmp1, this->tileLength * sizeof(float));
            pipe.InitBuffer(QueueTmp2, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<TYPE_X, half>){
            auto tmp = QueueTmp1.Get<float>();
            auto tmpx = QueueTmp2.Get<float>();
            float c1 = 1.0;
            Cast(tmpx, xLocal, RoundMode::CAST_NONE, length);
            Mul(tmp, tmpx, tmpx, length);
            Adds(tmp, tmp, c1, length);
            Sqrt(tmp, tmp, length);
            Add(tmp, tmp, tmpx, length);
            Ln(tmp, tmp, length);
            Cast(yLocal, tmp, RoundMode::CAST_NONE, length);
        }
        else{
            TYPE_Y c1 = 1.0;
            Mul(yLocal, xLocal, xLocal, length);
            Adds(yLocal, yLocal, c1, length);
            Sqrt(yLocal, yLocal, length);
            Add(yLocal, xLocal, yLocal, length);
            Ln(yLocal, yLocal, length);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> QueueTmp1, QueueTmp2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinh<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, 
            tiling_data.block_size, tiling_data.core_size, 
            tiling_data.core_remain);
    op.Process();
}
