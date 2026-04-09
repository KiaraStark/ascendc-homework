#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  

class KernelSoftmax_Float {
public:
    __aicore__ inline KernelSoftmax_Float() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t width, uint32_t height)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->height = height;
        this->width = width;
        this->blockLength = this->height * width;
        this->rightpadding = ((width + 8) / 8 * 8) - width;

        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ float*)x, bufferlength);
        yGm.SetGlobalBuffer((__gm__ float*)y, bufferlength);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->width * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->width * sizeof(float));
        pipe.InitBuffer(maxQueue, 1, this->width * sizeof(float));
        pipe.InitBuffer(sumQueue, 1, this->width * sizeof(float));
        pipe.InitBuffer(workQueue, 1, this->width * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->height;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->width], this->width);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> sumTempLocal = sumQueue.AllocTensor<float>();
        LocalTensor<float> maxTempLocal = maxQueue.AllocTensor<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> workLocal = workQueue.AllocTensor<float>();

        float  c1 = -1.0; 
       
        ReduceMax(maxTempLocal, xLocal, workLocal, this->width, 0);
        Muls(maxTempLocal, maxTempLocal, c1, 1);
        auto max = maxTempLocal.GetValue(0);
        Duplicate(maxTempLocal, max, this->width);
        Add(xLocal, xLocal, maxTempLocal, this->width);

        Exp(xLocal, xLocal, this->width);
        ReduceSum(sumTempLocal, xLocal, workLocal, this->width);
        auto sum = sumTempLocal.GetValue(0);
        Duplicate(sumTempLocal, sum, this->width);
        Div(yLocal, xLocal, sumTempLocal, this->width);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        maxQueue.FreeTensor(maxTempLocal);
        sumQueue.FreeTensor(sumTempLocal);
        workQueue.FreeTensor(workLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) 
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->width], yLocal, this->width);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECIN, 1> maxQueue, sumQueue, workQueue;
    TBuf<QuePosition::VECCALC> tmp;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t width;
    uint32_t height;
    uint32_t blockLength;
    uint32_t rightpadding;
};



class KernelSoftmax_Half {
public:
    __aicore__ inline KernelSoftmax_Half() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t width, uint32_t height)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->height = height;
        this->width = width;
        this->blockLength = this->height * width;
        this->rightpadding = ((width + 16) / 16 * 16) - width;

        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ half*)x, bufferlength);
        yGm.SetGlobalBuffer((__gm__ half*)y, bufferlength);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->width * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->width * sizeof(half));
        pipe.InitBuffer(maxQueue, 1, this->width * sizeof(float));
        pipe.InitBuffer(sumQueue, 1, this->width * sizeof(float));
        pipe.InitBuffer(workQueue, 1, this->width * sizeof(float));
        pipe.InitBuffer(tmp, this->width * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->height;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->width], this->width);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<float> sumTempLocal = sumQueue.AllocTensor<float>();
        LocalTensor<float> maxTempLocal = maxQueue.AllocTensor<float>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<float> workLocal = workQueue.AllocTensor<float>();

        float  c1 = -1.0; 
        auto float_x = tmp.Get<float>();
        
        Cast(float_x, xLocal, RoundMode::CAST_NONE, this->width);

        Exp(float_x, float_x, this->width);
        ReduceSum(sumTempLocal, float_x, workLocal, this->width);
        auto sum = sumTempLocal.GetValue(0);
        Duplicate(sumTempLocal, sum, this->width);
        Div(float_x, float_x, sumTempLocal, this->width);

        Cast(yLocal, float_x, RoundMode::CAST_ROUND, this->width);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        maxQueue.FreeTensor(maxTempLocal);
        sumQueue.FreeTensor(sumTempLocal);
        workQueue.FreeTensor(workLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) 
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * this->width], yLocal, this->width);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECIN, 1> maxQueue, sumQueue, workQueue;
    TBuf<QuePosition::VECCALC> tmp;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    uint32_t width;
    uint32_t height;
    uint32_t blockLength;
    uint32_t rightpadding;
};



extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) {
        KernelSoftmax_Half op;
        op.Init(x, y, tiling_data.width, tiling_data.height);
        op.Process();
    }
    else if (TILING_KEY_IS(2)){
        KernelSoftmax_Float op;
        op.Init(x, y, tiling_data.width, tiling_data.height);
        op.Process();
    }
}