/**
 * Softmax Kernel Implementation
 * AscendC implementation for Softmax operator with dim=-1
 * Supports fp16 and fp32 precision
 */

#include "softmax_kernel.h"

extern "C" __global__ __ascend__ void SoftmaxKernel(
    __gm__ void* input,
    __gm__ void* output,
    uint32_t* shape,
    uint32_t shapeSize,
    int32_t dim,
    int32_t dtype
) {
    SoftmaxKernel softmaxKernel;
    softmaxKernel.Init(shape, shapeSize, dim, dtype);
    softmaxKernel.Process(reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output));
}
