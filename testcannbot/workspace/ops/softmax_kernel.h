/**
 * Softmax Kernel Definition
 * AscendC implementation for Softmax operator with dim=-1
 */

#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tensor.h"

constexpr uint32_t BLOCK_SIZE = 256;

class SoftmaxKernel {
public:
    __aicore__ SoftmaxKernel() {}
    ~SoftmaxKernel() {}

    /**
     * @brief Initialize the Softmax kernel
     */
    __aicore__ inline void Init(uint32_t* shape, uint32_t shapeSize, int32_t dim, int32_t dtype) {
        kernelParams.shapeSize = shapeSize;
        for (uint32_t i = 0; i < shapeSize; ++i) {
            kernelParams.shape[i] = shape[i];
        }
        kernelParams.dim = dim;
        kernelParams.dtype = dtype;
    }

    /**
     * @brief Main compute function
     */
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output) {
        uint32_t totalElements = 1;
        for (uint32_t i = 0; i < kernelParams.shapeSize; ++i) {
            totalElements *= kernelParams.shape[i];
        }

        const uint32_t lastDimSize = kernelParams.shape[kernelParams.shapeSize - 1];
        const uint32_t outerSize = totalElements / lastDimSize;

        constexpr uint32_t blockSize = BLOCK_SIZE;
        const uint32_t loopCount = (lastDimSize + blockSize - 1) / blockSize;

        if (kernelParams.dtype == ACL_FLOAT16) {
            ProcessFp16(input, output, lastDimSize, outerSize, blockSize, loopCount);
        } else {
            ProcessFp32(input, output, lastDimSize, outerSize, blockSize, loopCount);
        }
    }

private:
    struct {
        uint32_t shape[MAX_SHAPE_TENSOR_SIZE];
        uint32_t shapeSize;
        int32_t dim;
        int32_t dtype;
    } kernelParams;

    /**
     * @brief FP32 process
     */
    __aicore__ inline void ProcessFp32(
        GM_ADDR input, GM_ADDR output,
        uint32_t lastDimSize, uint32_t outerSize,
        uint32_t blockSize, uint32_t loopCount
    ) {
        GlobalTensor<float> inputGlobal;
        GlobalTensor<float> outputGlobal;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));

        LocalTensor<float> inputLocal = inputGlobal.GetLocalTensor();
        LocalTensor<float> outputLocal = outputGlobal.GetLocalTensor();

        for (uint32_t batchIdx = 0; batchIdx < outerSize; ++batchIdx) {
            for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
                ComputeBlockFp32(inputLocal, outputLocal, batchIdx, tileIdx, lastDimSize, blockSize);
            }
        }
    }

    /**
     * @brief FP16 process
     */
    __aicore__ inline void ProcessFp16(
        GM_ADDR input, GM_ADDR output,
        uint32_t lastDimSize, uint32_t outerSize,
        uint32_t blockSize, uint32_t loopCount
    ) {
        GlobalTensor<half> inputGlobal;
        GlobalTensor<half> outputGlobal;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));

        LocalTensor<half> inputLocal = inputGlobal.GetLocalTensor();
        LocalTensor<half> outputLocal = outputGlobal.GetLocalTensor();

        for (uint32_t batchIdx = 0; batchIdx < outerSize; ++batchIdx) {
            for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
                ComputeBlockFp16(inputLocal, outputLocal, batchIdx, tileIdx, lastDimSize, blockSize);
            }
        }
    }

    /**
     * @brief Compute a single block of softmax - FP32
     */
    __aicore__ inline void ComputeBlockFp32(
        LocalTensor<float>& inputLocal, LocalTensor<float>& outputLocal,
        uint32_t batchIdx, uint32_t tileIdx,
        uint32_t lastDimSize, uint32_t blockSize
    ) {
        const uint32_t offset = batchIdx * lastDimSize;
        const uint32_t startIdx = tileIdx * blockSize;
        const uint32_t actualBlockSize = (startIdx + blockSize > lastDimSize)
                                          ? (lastDimSize - startIdx)
                                          : blockSize;

        // Find max value
        float maxVal = -3.4028235e38f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            maxVal = max(maxVal, inputLocal[idx]);
        }

        // Compute exp sum
        float expSum = 0.0f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float expVal = Exp(inputLocal[idx] - maxVal);
            expSum += expVal;
        }

        // Compute softmax
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float expVal = Exp(inputLocal[idx] - maxVal);
            outputLocal[idx] = expVal / expSum;
        }
    }

    /**
     * @brief Compute a single block of softmax - FP16
     */
    __aicore__ inline void ComputeBlockFp16(
        LocalTensor<half>& inputLocal, LocalTensor<half>& outputLocal,
        uint32_t batchIdx, uint32_t tileIdx,
        uint32_t lastDimSize, uint32_t blockSize
    ) {
        const uint32_t offset = batchIdx * lastDimSize;
        const uint32_t startIdx = tileIdx * blockSize;
        const uint32_t actualBlockSize = (startIdx + blockSize > lastDimSize)
                                          ? (lastDimSize - startIdx)
                                          : blockSize;

        // Find max value (convert to float for comparison)
        float maxVal = -65504.0f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float val = static_cast<float>(inputLocal[idx]);
            maxVal = max(maxVal, val);
        }

        // Compute exp sum
        float expSum = 0.0f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float val = static_cast<float>(inputLocal[idx]);
            float expVal = Exp(val - maxVal);
            expSum += expVal;
        }

        // Compute softmax
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float val = static_cast<float>(inputLocal[idx]);
            float expVal = Exp(val - maxVal);
            outputLocal[idx] = static_cast<half>(expVal / expSum);
        }
    }
};

#endif // SOFTMAX_KERNEL_H
