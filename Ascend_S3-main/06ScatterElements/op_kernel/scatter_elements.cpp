#include "kernel_operator.h"
using namespace AscendC;


class KernelScatterElements_UINT8{
public:
    __aicore__ inline KernelScatterElements_UINT8() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, 
                                uint32_t var_M, uint32_t var_N, uint32_t var_K, 
                                uint32_t index_M, uint32_t index_N, uint32_t index_K, 
                                uint32_t updates_M, uint32_t updates_N, uint32_t updates_K, 
                                uint32_t axis, uint32_t reduce){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->var_M = var_M;
        this->var_N = var_N;
        this->var_K = var_K;

        this->index_M = index_M;
        this->index_N = index_N;
        this->index_K = index_K;

        this->updates_M = updates_M;
        this->updates_N = updates_N;
        this->updates_K = updates_K;

        this->axis = axis;
        this->reduce = reduce;
        auto varlength = var_M * var_N * var_K;
        auto indexlength = index_M * index_N * index_K;
        auto updateslength = updates_M * updates_N * updates_K;

        varGm.SetGlobalBuffer((__gm__ uint8_t*)var, varlength);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, indexlength);
        updatesGm.SetGlobalBuffer((__gm__ uint8_t*)updates, updateslength);

        pipe.InitBuffer(tmp1, 32);
        pipe.InitBuffer(tmp2, 32);
    }
    __aicore__ inline void Process() {
        uint8_t var_value, updates_value;
        int32_t indices_value;
        int32_t var_index, indices_index, updates_index;
        auto p1 = tmp1.Get<uint8_t>();
        auto p2 = tmp2.Get<uint8_t>();
        for(int32_t i = 0; i < this->index_M; i++){
            for(int32_t j = 0; j < this->index_N; j++){
                for(int32_t k = 0; k < this->index_K; k++){
                    // indices_index = i * this->index_N * this->index_K + j * this->index_K + k;
                    //printf("1: %d\n", indices_index);
                    indices_value = indicesGm.GetValue(i * this->index_N * this->index_K + j * this->index_K + k);
                    //printf("2: %d\n", indices_value);

                    //updates_index = i * this->updates_N * this->updates_K + j * this->updates_K + k;
                    updates_value = updatesGm.GetValue(i * this->updates_N * this->updates_K + j * this->updates_K + k);
                    //printf("3: %f\n", (float)updates_value);

                    if(this->axis == 0){
                        var_index = indices_value * this->var_N * this->var_K + j * this->var_K + k;
                    }
                    else if(this->axis == 1){
                        var_index = i * this->var_N * this->var_K + indices_value * this->var_K + k;
                    }
                    else{
                        var_index = i * this->var_N * this->var_K + j * this->var_K + indices_value;
                    }
                    
                    if(this->reduce == 0){
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                        varGm.SetValue(var_index, updates_value);
                    }
                    else if(this->reduce == 1){
                        var_value = varGm.GetValue(var_index);
                        //printf("%d  %f\n",  var_index, (float)var_value);
                        
                        auto p3 = tmp3.Get<half>();
                        auto p4 = tmp4.Get<half>();
                        p1.SetValue(0, var_value);
                        p2.SetValue(0, updates_value);
                        Cast(p3, p1, RoundMode::CAST_NONE, 32);
                        Cast(p4, p2, RoundMode::CAST_NONE, 32);
                        half value1 = p3.GetValue(0);
                        half value2 = p4.GetValue(0);
                        int16_t varInt16 = *(int16_t*)&value1;
                        int32_t fltInt32    =  ((varInt16 & 0x8000) << 16);
                        fltInt32        |= ((varInt16 & 0x7fff) << 13) + 0x38000000;
                        float varFp32 = *(float*)&fltInt32;

                        int16_t updateInt16 = *(int16_t*)&value2;
                        fltInt32    =  ((updateInt16 & 0x8000) << 16);
                        fltInt32        |= ((updateInt16 & 0x7fff) << 13) + 0x38000000;
                        float updateFp32 = *(float*)&fltInt32;

                        float res = varFp32 + updateFp32;

                        int16_t fltInt16;
                        fltInt32 = *(int32_t*)&res;
                        fltInt16    =  ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
                        fltInt16    |= ((fltInt32 & 0x80000000) >> 16);
                        value1 = *(half*)&fltInt16;
                        p3.SetValue(0, value1);
                        Cast(p1, p3, RoundMode::CAST_ROUND, 32);
                        updates_value = p1.GetValue(0);
                        varGm.SetValue(var_index, updates_value);
                    }
                    else{
                        var_value = varGm.GetValue(var_index);

                        auto p3 = tmp3.Get<half>();
                        auto p4 = tmp4.Get<half>();
                        p1.SetValue(0, var_value);
                        p2.SetValue(0, updates_value);
                        Cast(p3, p1, RoundMode::CAST_NONE, 32);
                        Cast(p4, p2, RoundMode::CAST_NONE, 32);
                        half value1 = p3.GetValue(0);
                        half value2 = p4.GetValue(0);
                        int16_t varInt16 = *(int16_t*)&value1;
                        int32_t fltInt32    =  ((varInt16 & 0x8000) << 16);
                        fltInt32        |= ((varInt16 & 0x7fff) << 13) + 0x38000000;
                        float varFp32 = *(float*)&fltInt32;

                        int16_t updateInt16 = *(int16_t*)&value2;
                        fltInt32    =  ((updateInt16 & 0x8000) << 16);
                        fltInt32        |= ((updateInt16 & 0x7fff) << 13) + 0x38000000;
                        float updateFp32 = *(float*)&fltInt32;

                        float res = varFp32 * updateFp32;

                        int16_t fltInt16;
                        fltInt32 = *(int32_t*)&res;
                        fltInt16    =  ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
                        fltInt16    |= ((fltInt32 & 0x80000000) >> 16);
                        value1 = *(half*)&fltInt16;
                        p3.SetValue(0, value1);
                        Cast(p1, p3, RoundMode::CAST_ROUND, 32);
                        updates_value = p1.GetValue(0);
                        varGm.SetValue(var_index, updates_value);
                    }
                }
            }
        }
    }
private:
    TPipe pipe;
    GlobalTensor<uint8_t> varGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<uint8_t> updatesGm;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4;

    uint32_t var_M;
    uint32_t var_N;
    uint32_t var_K;

    uint32_t index_M;
    uint32_t index_N;
    uint32_t index_K; 
    
    uint32_t updates_M;
    uint32_t updates_N;
    uint32_t updates_K;

    uint32_t axis;
    uint32_t reduce;
};


class KernelScatterElements_Half{
public:
    __aicore__ inline KernelScatterElements_Half() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, 
                                uint32_t var_M, uint32_t var_N, uint32_t var_K, 
                                uint32_t index_M, uint32_t index_N, uint32_t index_K, 
                                uint32_t updates_M, uint32_t updates_N, uint32_t updates_K, 
                                uint32_t axis, uint32_t reduce){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->var_M = var_M;
        this->var_N = var_N;
        this->var_K = var_K;

        this->index_M = index_M;
        this->index_N = index_N;
        this->index_K = index_K;

        this->updates_M = updates_M;
        this->updates_N = updates_N;
        this->updates_K = updates_K;

        this->axis = axis;
        this->reduce = reduce;
        auto varlength = var_M * var_N * var_K;
        auto indexlength = index_M * index_N * index_K;
        auto updateslength = updates_M * updates_N * updates_K;

        varGm.SetGlobalBuffer((__gm__ half*)var, varlength);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, indexlength);
        updatesGm.SetGlobalBuffer((__gm__ half*)updates, updateslength);

        pipe.InitBuffer(tmp1, 64);
        pipe.InitBuffer(tmp2, 64);
    }
    __aicore__ inline void Process() {
        half var_value, updates_value;
        int32_t indices_value;
        int32_t var_index, indices_index, updates_index;
        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        for(int32_t i = 0; i < this->index_M; i++){
            for(int32_t j = 0; j < this->index_N; j++){
                for(int32_t k = 0; k < this->index_K; k++){
                    // indices_index = i * this->index_N * this->index_K + j * this->index_K + k;
                    //printf("1: %d\n", indices_index);
                    indices_value = indicesGm.GetValue(i * this->index_N * this->index_K + j * this->index_K + k);
                    //printf("2: %d\n", indices_value);

                    //updates_index = i * this->updates_N * this->updates_K + j * this->updates_K + k;
                    updates_value = updatesGm.GetValue(i * this->updates_N * this->updates_K + j * this->updates_K + k);
                    //printf("3: %f\n", (float)updates_value);

                    if(this->axis == 0){
                        var_index = indices_value * this->var_N * this->var_K + j * this->var_K + k;
                    }
                    else if(this->axis == 1){
                        var_index = i * this->var_N * this->var_K + indices_value * this->var_K + k;
                    }
                    else{
                        var_index = i * this->var_N * this->var_K + j * this->var_K + indices_value;
                    }
                    
                    if(this->reduce == 0){
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                        varGm.SetValue(var_index, updates_value);
                    }
                    else if(this->reduce == 1){
                        var_value = varGm.GetValue(var_index);
                        //printf("%d  %f\n",  var_index, (float)var_value);

                        //int16_t varInt16 = *(int16_t*)&var_value;
                        //int32_t fltInt32    =  ((varInt16 & 0x8000) << 16);
                        //fltInt32        |= ((varInt16 & 0x7fff) << 13) + 0x38000000;
                        //float varFp32 = *(float*)&fltInt32;

                        //int16_t updateInt16 = *(int16_t*)&updates_value;
                        //fltInt32    =  ((updateInt16 & 0x8000) << 16);
                        //fltInt32        |= ((updateInt16 & 0x7fff) << 13) + 0x38000000;
                        //float updateFp32 = *(float*)&fltInt32;

                        //float res = varFp32 + updateFp32;
                        //int16_t fltInt16;
                        //fltInt32 = *(int32_t*)&res;
                        //fltInt16    =  ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
                        //fltInt16    |= ((fltInt32 & 0x80000000) >> 16);
                        //updates_value = *(half*)&fltInt16;
                        
                        updates_value = (half)((float)updates_value + (float)var_value);
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                        varGm.SetValue(var_index, updates_value);
                    }
                    else{
                        var_value = varGm.GetValue(var_index);
                        int16_t varInt16 = *(int16_t*)&var_value;
                        int32_t fltInt32    =  ((varInt16 & 0x8000) << 16);
                        fltInt32        |= ((varInt16 & 0x7fff) << 13) + 0x38000000;
                        float varFp32 = *(float*)&fltInt32;

                        int16_t updateInt16 = *(int16_t*)&updates_value;
                        fltInt32    =  ((updateInt16 & 0x8000) << 16);
                        fltInt32        |= ((updateInt16 & 0x7fff) << 13) + 0x38000000;
                        float updateFp32 = *(float*)&fltInt32;

                        float res = varFp32 * updateFp32;

                        int16_t fltInt16;
                        fltInt32 = *(int32_t*)&res;
                        fltInt16    =  ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
                        fltInt16    |= ((fltInt32 & 0x80000000) >> 16);
                        half value1 = *(half*)&fltInt16;
                        varGm.SetValue(var_index, value1);
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                        //updates_value = (half)((float)updates_value * (float)var_value);
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                    }
                }
            }
        }
    }
private:
    TPipe pipe;
    GlobalTensor<half> varGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<half> updatesGm;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4;

    uint32_t var_M;
    uint32_t var_N;
    uint32_t var_K;

    uint32_t index_M;
    uint32_t index_N;
    uint32_t index_K; 
    
    uint32_t updates_M;
    uint32_t updates_N;
    uint32_t updates_K;

    uint32_t axis;
    uint32_t reduce;
};


class KernelScatterElements_FLOAT{
public:
    __aicore__ inline KernelScatterElements_FLOAT() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, 
                                uint32_t var_M, uint32_t var_N, uint32_t var_K, 
                                uint32_t index_M, uint32_t index_N, uint32_t index_K, 
                                uint32_t updates_M, uint32_t updates_N, uint32_t updates_K, 
                                uint32_t axis, uint32_t reduce){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->var_M = var_M;
        this->var_N = var_N;
        this->var_K = var_K;

        this->index_M = index_M;
        this->index_N = index_N;
        this->index_K = index_K;

        this->updates_M = updates_M;
        this->updates_N = updates_N;
        this->updates_K = updates_K;

        this->axis = axis;
        this->reduce = reduce;
        auto varlength = var_M * var_N * var_K;
        auto indexlength = index_M * index_N * index_K;
        auto updateslength = updates_M * updates_N * updates_K;

        varGm.SetGlobalBuffer((__gm__ float*)var, varlength);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, indexlength);
        updatesGm.SetGlobalBuffer((__gm__ float*)updates, updateslength);

        pipe.InitBuffer(tmp1, 32);
        pipe.InitBuffer(tmp2, 32);
    }
    __aicore__ inline void Process() {
        float var_value, updates_value;
        int32_t indices_value;
        int32_t var_index, indices_index, updates_index;
        auto p1 = tmp1.Get<float>();
        auto p2 = tmp2.Get<float>();
        for(int32_t i = 0; i < this->index_M; i++){
            for(int32_t j = 0; j < this->index_N; j++){
                for(int32_t k = 0; k < this->index_K; k++){
                    // indices_index = i * this->index_N * this->index_K + j * this->index_K + k;
                    //printf("1: %d\n", indices_index);
                    indices_value = indicesGm.GetValue(i * this->index_N * this->index_K + j * this->index_K + k);
                    //printf("2: %d\n", indices_value);

                    //updates_index = i * this->updates_N * this->updates_K + j * this->updates_K + k;
                    updates_value = updatesGm.GetValue(i * this->updates_N * this->updates_K + j * this->updates_K + k);
                    //printf("3: %f\n", (float)updates_value);

                    if(this->axis == 0){
                        var_index = indices_value * this->var_N * this->var_K + j * this->var_K + k;
                    }
                    else if(this->axis == 1){
                        var_index = i * this->var_N * this->var_K + indices_value * this->var_K + k;
                    }
                    else{
                        var_index = i * this->var_N * this->var_K + j * this->var_K + indices_value;
                    }
                    
                    if(this->reduce == 0){
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                        varGm.SetValue(var_index, updates_value);
                    }
                    else if(this->reduce == 1){
                        var_value = varGm.GetValue(var_index);
                        //printf("%d  %f\n",  var_index, (float)var_value);
                        
                        //p1.SetValue(0, var_value);
                        //p2.SetValue(0, updates_value);
                        //Add(p1, p1, p2, 16);
                        //updates_value = p1.GetValue(0);
                        updates_value = updates_value + var_value;
                        varGm.SetValue(var_index, updates_value);
                        //updates_value = (half)((float)updates_value + (float)var_value);
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                    }
                    else{
                        var_value = varGm.GetValue(var_index);
                       
                        //p1.SetValue(0, var_value);
                        //p2.SetValue(0, updates_value);
                        //Mul(p1, p1, p2, 16);
                        //updates_value = p1.GetValue(0);
                        updates_value = updates_value + var_value;
                        varGm.SetValue(var_index, updates_value);
                    }
                }
            }
        }
    }
private:
    TPipe pipe;
    GlobalTensor<float> varGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<float> updatesGm;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;

    uint32_t var_M;
    uint32_t var_N;
    uint32_t var_K;

    uint32_t index_M;
    uint32_t index_N;
    uint32_t index_K; 
    
    uint32_t updates_M;
    uint32_t updates_N;
    uint32_t updates_K;

    uint32_t axis;
    uint32_t reduce;
};



class KernelScatterElements_INT{
public:
    __aicore__ inline KernelScatterElements_INT() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, 
                                uint32_t var_M, uint32_t var_N, uint32_t var_K, 
                                uint32_t index_M, uint32_t index_N, uint32_t index_K, 
                                uint32_t updates_M, uint32_t updates_N, uint32_t updates_K, 
                                uint32_t axis, uint32_t reduce){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->var_M = var_M;
        this->var_N = var_N;
        this->var_K = var_K;

        this->index_M = index_M;
        this->index_N = index_N;
        this->index_K = index_K;

        this->updates_M = updates_M;
        this->updates_N = updates_N;
        this->updates_K = updates_K;

        this->axis = axis;
        this->reduce = reduce;
        auto varlength = var_M * var_N * var_K;
        auto indexlength = index_M * index_N * index_K;
        auto updateslength = updates_M * updates_N * updates_K;

        varGm.SetGlobalBuffer((__gm__ int32_t*)var, varlength);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, indexlength);
        updatesGm.SetGlobalBuffer((__gm__ int32_t*)updates, updateslength);

        pipe.InitBuffer(tmp1, 32);
        pipe.InitBuffer(tmp2, 32);
    }
    __aicore__ inline void Process() {
        int32_t var_value, updates_value;
        int32_t indices_value;
        int32_t var_index, indices_index, updates_index;
        auto p1 = tmp1.Get<int32_t>();
        auto p2 = tmp2.Get<int32_t>();
        for(int32_t i = 0; i < this->index_M; i++){
            for(int32_t j = 0; j < this->index_N; j++){
                for(int32_t k = 0; k < this->index_K; k++){
                    // indices_index = i * this->index_N * this->index_K + j * this->index_K + k;
                    //printf("1: %d\n", indices_index);
                    indices_value = indicesGm.GetValue(i * this->index_N * this->index_K + j * this->index_K + k);
                    //printf("2: %d\n", indices_value);

                    //updates_index = i * this->updates_N * this->updates_K + j * this->updates_K + k;
                    updates_value = updatesGm.GetValue(i * this->updates_N * this->updates_K + j * this->updates_K + k);
                    //printf("3: %f\n", (float)updates_value);

                    if(this->axis == 0){
                        var_index = indices_value * this->var_N * this->var_K + j * this->var_K + k;
                    }
                    else if(this->axis == 1){
                        var_index = i * this->var_N * this->var_K + indices_value * this->var_K + k;
                    }
                    else{
                        var_index = i * this->var_N * this->var_K + j * this->var_K + indices_value;
                    }
                    
                    if(this->reduce == 0){
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                        varGm.SetValue(var_index, updates_value);
                    }
                    else if(this->reduce == 1){
                        var_value = varGm.GetValue(var_index);
                        //printf("%d  %f\n",  var_index, (float)var_value);
                        
                        //p1.SetValue(0, var_value);
                        //p2.SetValue(0, updates_value);
                        //Add(p1, p1, p2, 16);
                        //updates_value = p1.GetValue(0);
                        updates_value = updates_value + var_value;
                        varGm.SetValue(var_index, updates_value);
                        //updates_value = (half)((float)updates_value + (float)var_value);
                        //printf("%d  %f\n",  var_index, (float)updates_value);
                    }
                    else{
                        var_value = varGm.GetValue(var_index);
                       
                        //p1.SetValue(0, var_value);
                        //p2.SetValue(0, updates_value);
                        //Mul(p1, p1, p2, 16);
                        //updates_value = p1.GetValue(0);
                        updates_value = updates_value + var_value;
                        varGm.SetValue(var_index, updates_value);
                    }
                }
            }
        }
    }
private:
    TPipe pipe;
    GlobalTensor<int32_t> varGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<int32_t> updatesGm;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;

    uint32_t var_M;
    uint32_t var_N;
    uint32_t var_K;

    uint32_t index_M;
    uint32_t index_N;
    uint32_t index_K; 
    
    uint32_t updates_M;
    uint32_t updates_N;
    uint32_t updates_K;

    uint32_t axis;
    uint32_t reduce;
};


extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl

    if (TILING_KEY_IS(1)) {
        KernelScatterElements_UINT8 op;
        op.Init(var, indices, updates, tiling_data.var_M, tiling_data.var_N, tiling_data.var_K,
            tiling_data.index_M, tiling_data.index_N, tiling_data.index_K, 
            tiling_data.updates_M, tiling_data.updates_N, tiling_data.updates_K, 
            tiling_data.axis, tiling_data.reduce);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) {
        KernelScatterElements_Half op;
        op.Init(var, indices, updates, tiling_data.var_M, tiling_data.var_N, tiling_data.var_K,
            tiling_data.index_M, tiling_data.index_N, tiling_data.index_K, 
            tiling_data.updates_M, tiling_data.updates_N, tiling_data.updates_K, 
            tiling_data.axis, tiling_data.reduce);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) {
        KernelScatterElements_FLOAT op;
        op.Init(var, indices, updates, tiling_data.var_M, tiling_data.var_N, tiling_data.var_K,
            tiling_data.index_M, tiling_data.index_N, tiling_data.index_K, 
            tiling_data.updates_M, tiling_data.updates_N, tiling_data.updates_K, 
            tiling_data.axis, tiling_data.reduce);
        op.Process();
    }
    else{
        KernelScatterElements_INT op;
        op.Init(var, indices, updates, tiling_data.var_M, tiling_data.var_N, tiling_data.var_K,
            tiling_data.index_M, tiling_data.index_N, tiling_data.index_K, 
            tiling_data.updates_M, tiling_data.updates_N, tiling_data.updates_K, 
            tiling_data.axis, tiling_data.reduce);
        op.Process();
    }
    
}