// softmax_host.cpp
// Host-side code for softmax kernel

#include <cstdint>

extern "C" {
void SoftmaxKernel(
    void* input,
    void* output,
    uint32_t* shape,
    uint32_t shapeSize,
    int32_t dim,
    int32_t dtype
);
}

void softmax_launch(
    void* input,
    void* output,
    uint32_t* shape,
    uint32_t shapeSize,
    int32_t dim,
    int32_t dtype
) {
    SoftmaxKernel(input, output, shape, shapeSize, dim, dtype);
}
