#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

template<typename TYPE_Y, typename TYPE_DY, typename TYPE_Z> class KernelAsinhGrad {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z,  uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);
        Gm_dy.SetGlobalBuffer((__gm__ TYPE_DY*)dy + startPointer, bufferlength);
        Gm_z.SetGlobalBuffer((__gm__ TYPE_Z*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe = pipeIn;
        pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe->InitBuffer(Q_dy, BUFFER_NUM, this->tileLength * sizeof(TYPE_DY));
        pipe->InitBuffer(Q_z, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z));

        if constexpr (std::is_same_v<T, half>) {
            pipe->InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe->InitBuffer(B_x2, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        LocalTensor<TYPE_DY> dy = Q_dy.AllocTensor<TYPE_DY>();
        DataCopy(y, Gm_y[progress * this->tileLength], length);
        DataCopy(dy, Gm_dy[progress * this->tileLength], length);
        Q_y.EnQue(y);
        Q_dy.EnQue(dy);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        LocalTensor<TYPE_DY> dy = Q_dy.DeQue<TYPE_DY>();
        LocalTensor<TYPE_Z> z = Q_z.AllocTensor<TYPE_Z>();

        TYPE_Y c1 = -1.0, c2 = 0.5;
        if constexpr (std::is_same_v<T, half>) {
            auto float_y1 = B_x1.Get<float>();
            auto float_y2 = B_x2.Get<float>();
            Cast(float_y1, y, RoundMode::CAST_NONE, length);
            Muls(y, y, c1, length);
            Cast(float_y2, y, RoundMode::CAST_NONE, length);
            Exp(float_y1, float_y1, length);
            Exp(float_y2, float_y2, length);
            Add(float_y1, float_y1, float_y2, length);
            Cast(y, float_y1, RoundMode::CAST_NONE, length);

            Muls(y, y, c2, length);
            Div(z, dy, y, length);
        }
        else{
            Muls(z, y, c1, length);
            Exp(y, y, length);
            Exp(z, z, length);
            Add(y, y, z, length);
            Muls(y, y, c2, length);
            Div(z, dy, y, length);
        }
        
        Q_y.FreeTensor(y);
        Q_dy.FreeTensor(dy);
        Q_z.EnQue<TYPE_Z>(z);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Z> z = Q_z.DeQue<TYPE_Z>();
        DataCopy(Gm_z[progress * this->tileLength], z, length);
        Q_z.FreeTensor(z);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_y, Q_dy;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_z;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    GlobalTensor<TYPE_DY> Gm_dy;
    GlobalTensor<TYPE_Z> Gm_z;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelAsinhGrad<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;
    op.Init(y, dy, z, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, &pipe);
    op.Process();
}