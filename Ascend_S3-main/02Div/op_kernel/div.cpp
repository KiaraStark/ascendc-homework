#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelDiv {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,  uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe = pipeIn;
        pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe->InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<T, int8_t> ) {
            pipe->InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe->InitBuffer(B_x2, this->tileLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int32_t>) {
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
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<T, int8_t>) {
            auto half_x1 = B_x1.Get<half>();
            auto half_x2 = B_x2.Get<half>();
            Cast(half_x1, x1, RoundMode::CAST_NONE, length);
            Cast(half_x2, x2, RoundMode::CAST_NONE, length);
            Div(half_x1, half_x1, half_x2, length);
            //Cast(y, half_x1, RoundMode::CAST_FLOOR, length);
            
            Cast(half_x2.ReinterpretCast<int16_t>(), half_x1, RoundMode::CAST_FLOOR, length);
            ShiftLeft(half_x2.ReinterpretCast<int16_t>(), half_x2.ReinterpretCast<int16_t>(), int16_t(8), length);
            ShiftRight(half_x2.ReinterpretCast<int16_t>(), half_x2.ReinterpretCast<int16_t>(), int16_t(8), length);
            Cast(half_x1, half_x2.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, length);
            Cast(y, half_x1, RoundMode::CAST_ROUND, length);
        }
        else if constexpr (std::is_same_v<T, half>){
            auto float_x1 = B_x1.Get<float>();
            auto float_x2 = B_x2.Get<float>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Div(float_x1, float_x1, float_x2, length);
            Cast(y, float_x1, RoundMode::CAST_ROUND, length);
        }
        else if constexpr (std::is_same_v<T, int32_t>) {
            auto float_x1 = B_x1.Get<float>();
            auto float_x2 = B_x2.Get<float>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Div(float_x1, float_x1, float_x2, length);
            Cast(y, float_x1, RoundMode::CAST_FLOOR, length);
        }
        else{
            Div(y, x1, x2, length);
        }
        
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelDiv_Broadcast {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelDiv_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t x1_length, uint32_t x2_length, uint32_t total_length, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->x1Length = x1_length;
        this->x2Length = x2_length;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->startPointer = core_size * GetBlockIdx();

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, total_length);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, total_length);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, total_length);

        pipe = pipeIn;
        pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe->InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        if constexpr (std::is_same_v<T, int8_t> ) {
            pipe->InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe->InitBuffer(B_x2, this->tileLength * sizeof(half));
        }
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int32_t>) {
            pipe->InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe->InitBuffer(B_x2, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        for (int32_t i = 0; i < loopCount-1; i++) {
            uint32_t position = startPointer + i * this->tileLength;
            CopyIn(position, this->tileLength);
            Compute(this->tileLength);
            CopyOut(position, this->tileLength);
        }
        uint32_t position = startPointer + (loopCount - 1) * this->tileLength;
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(position, length);
        Compute(length);
        CopyOut(position, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t position, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[position % x1Length], length);
        DataCopy(x2, Gm_x2[position % x2Length], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<T, int8_t>) {
            auto half_x1 = B_x1.Get<half>();
            auto half_x2 = B_x2.Get<half>();
            Cast(half_x1, x1, RoundMode::CAST_NONE, length);
            Cast(half_x2, x2, RoundMode::CAST_NONE, length);
            Div(half_x1, half_x1, half_x2, length);
            //Cast(y, half_x1, RoundMode::CAST_FLOOR, length);
            
            Cast(half_x2.ReinterpretCast<int16_t>(), half_x1, RoundMode::CAST_FLOOR, length);
            ShiftLeft(half_x2.ReinterpretCast<int16_t>(), half_x2.ReinterpretCast<int16_t>(), int16_t(8), length);
            ShiftRight(half_x2.ReinterpretCast<int16_t>(), half_x2.ReinterpretCast<int16_t>(), int16_t(8), length);
            Cast(half_x1, half_x2.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, length);
            Cast(y, half_x1, RoundMode::CAST_ROUND, length);
        }
        else if constexpr (std::is_same_v<T, half>){
            auto float_x1 = B_x1.Get<float>();
            auto float_x2 = B_x2.Get<float>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Div(float_x1, float_x1, float_x2, length);
            Cast(y, float_x1, RoundMode::CAST_ROUND, length);
        }
        else if constexpr (std::is_same_v<T, int32_t>) {
            auto float_x1 = B_x1.Get<float>();
            auto float_x2 = B_x2.Get<float>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Div(float_x1, float_x1, float_x2, length);
            Cast(y, float_x1, RoundMode::CAST_FLOOR, length);
        }
        else{
            Div(y, x1, x2, length);
        }

        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t position, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[position], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startPointer;
    uint32_t x1Length;
    uint32_t x2Length;
};


extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

    if (tiling_data.x1_length == tiling_data.total_length && tiling_data.x2_length == tiling_data.total_length) {
        KernelDiv<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x1, x2, y, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, &pipe);
        op.Process();
    }
    else {
        KernelDiv_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x1, x2, y, tiling_data.x1_length, tiling_data.x2_length, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, &pipe);
        op.Process();
    }
}