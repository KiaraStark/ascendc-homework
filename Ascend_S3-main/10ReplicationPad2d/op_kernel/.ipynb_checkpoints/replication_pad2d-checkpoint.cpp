#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

template<typename T> class KernelReplicationPad2d {
public:
    __aicore__ inline KernelReplicationPad2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, uint32_t loopCount, uint32_t core_size, uint32_t block_size)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        paddingsGm.SetGlobalBuffer((__gm__ int32_t*)paddings, 4);
        this->left_pad = paddingsGm.GetValue(0);
        this->right_pad = paddingsGm.GetValue(1);
        this->top_pad = paddingsGm.GetValue(2);
        this->bottom_pad = paddingsGm.GetValue(3);

        this->align_left_pad = ((this->left_pad + 31) / 32) * 32;
        this->align_right_pad = ((this->right_pad + 31) / 32) * 32;

        this->outerloop = core_size;
        this->innerloop = loopCount;
        this->block_size = block_size;
        this->repeatTimes = (block_size * sizeof(T) + 255) / 256;
        this->mask = block_size * sizeof(T) > 256 ? (256 / sizeof(T)) : block_size;

        auto startPointer = core_size * loopCount * block_size * GetBlockIdx();
        this->out_block_size = (block_size + this->left_pad + this->right_pad);
        auto endPointer = core_size * (loopCount + this->top_pad + this->bottom_pad) * this->out_block_size * GetBlockIdx();
        auto startbufferlength = this->outerloop * block_size * loopCount;
        auto endbufferlength = this->outerloop * (loopCount + this->top_pad + this->bottom_pad) * this->out_block_size;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, startbufferlength);
        yGm.SetGlobalBuffer((__gm__ T*)y + endPointer, endbufferlength);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(outQueuehead, BUFFER_NUM, this->align_left_pad * sizeof(T));
        pipe.InitBuffer(outQueuetail, BUFFER_NUM, this->align_right_pad * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        for(int32_t i = 0; i < this->outerloop; i++){
            CopyIn(i * this->innerloop);
            Compute();
            CopyOut(i * (this->innerloop + this->top_pad + this->bottom_pad) * this->out_block_size, this->top_pad + 1);
        
            for(int32_t j = 1; j < this->innerloop - 1; j++){
                CopyIn(i * this->innerloop + j);
                Compute();
                CopyOut((i * (this->innerloop + this->top_pad + this->bottom_pad) + this->top_pad + j) * this->out_block_size, 1);
            }

            CopyIn((i + 1) * this->innerloop - 1);
            Compute();
            CopyOut(((i + 1) * (this->innerloop + this->top_pad + this->bottom_pad) - this->bottom_pad - 1) * this->out_block_size, this->bottom_pad + 1);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->block_size * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[progress * this->block_size], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> headLocal = outQueuehead.AllocTensor<T>();
        LocalTensor<T> tailLocal = outQueuetail.AllocTensor<T>();

        T headnumber = xLocal.GetValue(0);
        T tailnumber = xLocal.GetValue(this->block_size - 1);

        Duplicate(headLocal, headnumber, this->align_left_pad);
        Duplicate(tailLocal, tailnumber, this->align_right_pad);
        Copy(yLocal, xLocal, this->mask, this->repeatTimes, {1,1,8,8});
        
        outQueuehead.EnQue<T>(headLocal);
        outQueueY.EnQue<T>(yLocal);
        outQueuetail.EnQue<T>(tailLocal);
        inQueueX.FreeTensor(xLocal);
    }
     __aicore__ inline void CopyOut(int32_t address, int32_t loop)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        LocalTensor<T> headLocal = outQueuehead.DeQue<T>();
        LocalTensor<T> tailLocal = outQueuetail.DeQue<T>();

        DataCopyExtParams copyParamshead{1, static_cast<uint32_t>(this->left_pad * sizeof(T)), 0, 0, 0};
        DataCopyExtParams copyParamsblock{1, static_cast<uint32_t>(this->block_size * sizeof(T)), 0, 0, 0};
        DataCopyExtParams copyParamstail{1, static_cast<uint32_t>(this->right_pad * sizeof(T)), 0, 0, 0};

        for(int32_t i = 0; i < loop; i++){
            DataCopyPad(yGm[address + i * this->out_block_size], headLocal, copyParamshead);
            DataCopyPad(yGm[address + this->left_pad + i * this->out_block_size], yLocal, copyParamsblock);
            DataCopyPad(yGm[address + this->left_pad + this->block_size + i * this->out_block_size], headLocal, copyParamstail);
        }
        outQueueY.FreeTensor(yLocal);
        outQueuehead.FreeTensor(headLocal);
        outQueuetail.FreeTensor(tailLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuehead;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuetail;
    GlobalTensor<T> xGm;
    GlobalTensor<int32_t> paddingsGm;
    GlobalTensor<T> yGm;
    uint32_t left_pad;
    uint32_t align_left_pad;
    uint32_t right_pad;
    uint32_t align_right_pad;
    uint32_t top_pad;
    uint32_t bottom_pad;
    uint32_t outerloop;
    uint32_t innerloop;
    uint32_t block_size;
    uint32_t out_block_size;
    uint32_t repeatTimes;
    uint32_t mask;
};


extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelReplicationPad2d<DTYPE_X> op;
    op.Init(x, paddings, y, tiling_data.loopCount, tiling_data.core_size, tiling_data.block_size);
    op.Process();
}