/**
 * Softmax Kernel Definition - Simplified Version
 * This is a simplified AscendC-style implementation for compilation testing
 */

#ifndef SOFTMAX_KERNEL_SIMPLE_H
#define SOFTMAX_KERNEL_SIMPLE_H

#include <cstdint>
#include <cmath>

// Simplified AscendC types for compilation
using GM_ADDR = void*;
constexpr int ACL_FLOAT16 = 1;
constexpr int ACL_FLOAT = 2;
constexpr uint32_t MAX_SHAPE_TENSOR_SIZE = 16;

constexpr uint32_t BLOCK_SIZE = 256;

class SoftmaxKernelSimple {
public:
    SoftmaxKernelSimple() {}
    ~SoftmaxKernelSimple() {}

    inline void Init(uint32_t* shape, uint32_t shapeSize, int32_t dim, int32_t dtype) {
        kernelParams.shapeSize = shapeSize;
        for (uint32_t i = 0; i < shapeSize; ++i) {
            kernelParams.shape[i] = shape[i];
        }
        kernelParams.dim = dim;
        kernelParams.dtype = dtype;
    }

    inline void Process(GM_ADDR input, GM_ADDR output) {
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

    inline void ProcessFp32(
        GM_ADDR input, GM_ADDR output,
        uint32_t lastDimSize, uint32_t outerSize,
        uint32_t blockSize, uint32_t loopCount
    ) {
        float* inputPtr = reinterpret_cast<float*>(input);
        float* outputPtr = reinterpret_cast<float*>(output);

        for (uint32_t batchIdx = 0; batchIdx < outerSize; ++batchIdx) {
            for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
                ComputeBlockFp32(inputPtr, outputPtr, batchIdx, tileIdx, lastDimSize, blockSize);
            }
        }
    }

    inline void ProcessFp16(
        GM_ADDR input, GM_ADDR output,
        uint32_t lastDimSize, uint32_t outerSize,
        uint32_t blockSize, uint32_t loopCount
    ) {
        uint16_t* inputPtr = reinterpret_cast<uint16_t*>(input);
        uint16_t* outputPtr = reinterpret_cast<uint16_t*>(output);

        for (uint32_t batchIdx = 0; batchIdx < outerSize; ++batchIdx) {
            for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
                ComputeBlockFp16(inputPtr, outputPtr, batchIdx, tileIdx, lastDimSize, blockSize);
            }
        }
    }

    inline void ComputeBlockFp32(
        float* input, float* output,
        uint32_t batchIdx, uint32_t tileIdx,
        uint32_t lastDimSize, uint32_t blockSize
    ) {
        const uint32_t offset = batchIdx * lastDimSize;
        const uint32_t startIdx = tileIdx * blockSize;
        const uint32_t actualBlockSize = (startIdx + blockSize > lastDimSize)
                                          ? (lastDimSize - startIdx)
                                          : blockSize;

        float maxVal = -3.4028235e38f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            maxVal = (input[idx] > maxVal) ? input[idx] : maxVal;
        }

        float expSum = 0.0f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float expVal = expf(input[idx] - maxVal);
            expSum += expVal;
        }

        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float expVal = expf(input[idx] - maxVal);
            output[idx] = expVal / expSum;
        }
    }

    inline void ComputeBlockFp16(
        uint16_t* input, uint16_t* output,
        uint32_t batchIdx, uint32_t tileIdx,
        uint32_t lastDimSize, uint32_t blockSize
    ) {
        const uint32_t offset = batchIdx * lastDimSize;
        const uint32_t startIdx = tileIdx * blockSize;
        const uint32_t actualBlockSize = (startIdx + blockSize > lastDimSize)
                                          ? (lastDimSize - startIdx)
                                          : blockSize;

        float maxVal = -65504.0f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float val = half_to_float(input[idx]);
            maxVal = (val > maxVal) ? val : maxVal;
        }

        float expSum = 0.0f;
        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float val = half_to_float(input[idx]);
            float expVal = expf(val - maxVal);
            expSum += expVal;
        }

        for (uint32_t i = 0; i < actualBlockSize; ++i) {
            uint32_t idx = offset + startIdx + i;
            float val = half_to_float(input[idx]);
            float expVal = expf(val - maxVal);
            output[idx] = float_to_half(expVal / expSum);
        }
    }

    // IEEE 754 half-precision float conversion
    inline float half_to_float(uint16_t h) {
        unsigned int sign = (h >> 15) & 0x1;
        unsigned int exponent = (h >> 10) & 0x1f;
        unsigned int mantissa = h & 0x3ff;

        float result;
        if (exponent == 0) {
            result = ldexpf((float)mantissa, -24);
        } else if (exponent == 31) {
            result = (mantissa == 0) ? INFINITY : NAN;
        } else {
            result = ldexpf((float)(mantissa | 0x400), (int)exponent - 25);
        }

        return sign ? -result : result;
    }

    inline uint16_t float_to_half(float f) {
        unsigned int bits = *((unsigned int*)&f);
        unsigned int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xff) - 127;
        unsigned int mantissa = bits & 0x7fffff;

        if (exponent > 15) return sign | 0x7c00;  // Overflow -> Infinity
        if (exponent < -24) return sign;  // Underflow -> Zero

        uint16_t result;
        if (exponent >= -14) {
            int adjusted_exp = exponent + 15;
            result = sign | ((adjusted_exp << 10) & 0x7c00) | ((mantissa >> 13) & 0x3ff);
        } else {
            result = sign | ((mantissa + 0x800000) >> (14 - exponent));
        }

        return result;
    }
};

#endif // SOFTMAX_KERNEL_SIMPLE_H
