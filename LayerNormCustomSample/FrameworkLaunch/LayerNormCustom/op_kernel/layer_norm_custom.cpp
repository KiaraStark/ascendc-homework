#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;

class KernelLayerNorm {
 public:
  __aicore__ inline KernelLayerNorm() {}
  __aicore__ inline void InitTiling(GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    rowNum = tiling_data.rowNum;
    rowNumSp = tiling_data.rowNumSp;
    rowLength = tiling_data.rowLength;
    blockPivot = tiling_data.blockPivot;
    tileLoop = tiling_data.tileLoop;
    tileLength = tiling_data.tileLength;
    loopCount = tiling_data.loopCount;
    factor = tiling_data.factor;
    mfactor = tiling_data.mfactor;
    eps = tiling_data.eps;
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR z,
                              GM_ADDR tiling) {
    InitTiling(tiling);

    this->leftRow = this->rowNum % this->tileLoop;
    if (AscendC::GetBlockIdx() < this->blockPivot) {
      this->rowNum = this->rowNumSp;
      this->leftRow += 1;
    }

    this->blockLength = this->rowNum * this->rowLength;
    uint32_t offset = 0;
    if (AscendC::GetBlockIdx() < this->blockPivot) {
      offset = this->blockLength * AscendC::GetBlockIdx();
    } else {
      offset = this->blockLength * AscendC::GetBlockIdx() +
               this->rowLength * this->blockPivot;
    }

    xGm.SetGlobalBuffer((__gm__ float *)x + offset, this->blockLength);
    zGm.SetGlobalBuffer((__gm__ float *)z + offset, this->blockLength);

    gammaGm.SetGlobalBuffer((__gm__ float *)gamma, this->rowLength);
    betaGm.SetGlobalBuffer((__gm__ float *)beta, this->rowLength);

    pipe.InitBuffer(queueX, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(queueZ, BUFFER_NUM, this->tileLength * sizeof(float));

    pipe.InitBuffer(tmpBuffer1, 64 * sizeof(float));
    pipe.InitBuffer(tmpBuffer2, 64 * sizeof(float));
    pipe.InitBuffer(onesBuffer, 64 * sizeof(float));

    pipe.InitBuffer(queueGamma, 1, this->rowLength * sizeof(float));
    pipe.InitBuffer(queueBeta, 1, this->rowLength * sizeof(float));
  }
  __aicore__ inline void Process() {
    for (int32_t i = 0; i < this->loopCount; i++) {
      CopyIn(i, this->tileLoop);
      Compute(i, this->tileLoop);
      CopyOut(i, this->tileLoop);
    }
    if (this->leftRow > 0) {
      CopyIn(this->loopCount, this->leftRow);
      Compute(this->loopCount, this->leftRow);
      CopyOut(this->loopCount, this->leftRow);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress, int32_t rowNum) {
    AscendC::LocalTensor<float> xLocal = queueX.AllocTensor<float>();
    AscendC::LocalTensor<float> gammaLocal = queueGamma.AllocTensor<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength],
             this->rowLength * rowNum);
    AscendC::DataCopy(gammaLocal, gammaGm[0], this->rowLength);
    AscendC::DataCopy(betaLocal, betaGm[0], this->rowLength);
    queueX.EnQue(xLocal);
    queueGamma.EnQue(gammaLocal);
    queueBeta.EnQue(betaLocal);
  }

  __aicore__ inline void Compute(int32_t progress, int32_t rowNum) {
    AscendC::LocalTensor<float> xLocal = queueX.DeQue<float>();
    AscendC::LocalTensor<float> gammaLocal = queueGamma.DeQue<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.DeQue<float>();

    AscendC::LocalTensor<float> tmpTensor1 = tmpBuffer1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuffer2.Get<float>();
    AscendC::LocalTensor<float> onesLocal = onesBuffer.Get<float>();
    AscendC::LocalTensor<float> zLocal = queueZ.AllocTensor<float>();
    AscendC::Duplicate<float>(onesLocal, 1.0f, this->tileLoop);

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::ReduceSum<float>(tmpTensor2[j], xLocal[buffIndex], tmpTensor1,
                       this->rowLength);
    }

    AscendC::Muls(zLocal, tmpTensor2, this->mfactor, rowNum);

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Adds(xLocal[buffIndex], xLocal[buffIndex], zLocal.GetValue(j),
           this->rowLength);
    }

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Mul(zLocal[buffIndex], xLocal[buffIndex], xLocal[buffIndex],
          this->rowLength);
    }
    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::ReduceSum<float>(tmpTensor2[j], zLocal[buffIndex], tmpTensor1,
                       this->rowLength);
    }
    AscendC::Muls(tmpTensor2, tmpTensor2, this->factor, rowNum);
    AscendC::Adds(tmpTensor2, tmpTensor2, this->eps, rowNum);
    AscendC::Sqrt(tmpTensor2, tmpTensor2, rowNum);
    AscendC::Div(tmpTensor2, onesLocal, tmpTensor2, rowNum);

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Muls(zLocal[buffIndex], xLocal[buffIndex], tmpTensor2.GetValue(j),
           this->rowLength);
    }

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Mul(zLocal[buffIndex], zLocal[buffIndex], gammaLocal, this->rowLength);
      AscendC::Add(zLocal[buffIndex], zLocal[buffIndex], betaLocal, this->rowLength);
    }

    queueZ.EnQue<float>(zLocal);
    queueGamma.FreeTensor(gammaLocal);
    queueBeta.FreeTensor(betaLocal);
    queueX.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress, int32_t rowNum) {
    AscendC::LocalTensor<float> zLocal = queueZ.DeQue<float>();

    AscendC::DataCopy(zGm[progress * this->tileLength], zLocal,
             rowNum * this->rowLength);

    queueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, onesBuffer;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> queueX;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> queueGamma, queueBeta;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> queueZ;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> gammaGm;
  AscendC::GlobalTensor<float> betaGm;
  AscendC::GlobalTensor<float> zGm;

  uint32_t blockLength = 0;
  uint32_t leftRow = 0;
  uint32_t rowNum = 341;
  uint32_t rowNumSp = 342;
  uint32_t rowLength = 1024;
  uint32_t blockPivot = 16;
  uint32_t tileLoop = 8;
  uint32_t tileLength = 8 * 1024;
  uint32_t loopCount = 42;
  float factor = 0.0009765625;
  float mfactor = -0.0009765625;
  float eps = 1e-5;
};
extern "C" __global__ __aicore__ void layer_norm_custom(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR res_out, GM_ADDR workspace,
    GM_ADDR tiling) {
  KernelLayerNorm op;
  op.Init(x, gamma, beta, res_out, tiling);
  if (TILING_KEY_IS(1)) {
    op.Process();
  }
}