/**
 * Softmax Kernel Implementation - Simplified Version
 */

#include "softmax_kernel_simple.h"

extern "C" void SoftmaxKernelSimpleFunc(
    void* input,
    void* output,
    uint32_t* shape,
    uint32_t shapeSize,
    int32_t dim,
    int32_t dtype
) {
    SoftmaxKernelSimple softmax;
    softmax.Init(shape, shapeSize, dim, dtype);
    softmax.Process(reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output));
}
