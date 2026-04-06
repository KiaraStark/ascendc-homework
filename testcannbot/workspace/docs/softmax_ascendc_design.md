# Softmax 算子 AscendC 实现设计文档

## 1. 概述

Softmax 算子将 K 维向量缩放到 (0,1) 区间，所有元素和为 1，常用于多分类任务输出层。

使用 AscendC 在华为昇腾 AI 处理器上实现。

## 2. 功能规格

| 特性 | 说明 |
|------|------|
| 计算维度 | dim=-1 (沿最后一维) |
| 输入精度 | fp16 (ACL_FLOAT16), fp32 (ACL_FLOAT32) |
| 输出精度 | 与输入相同 |
| 数值稳定 | 减最大值后再指数运算 |
| 输出范围 | (0, 1)，各维度和为 1 |

## 3. 文件结构

```
ops/
├── softmax_op.py          # NumPy 参考实现
├── softmax_kernel.h       # AscendC Kernel 头文件
├── softmax_kernel.cpp     # AscendC Kernel 实现
├── softmax_launcher.py     # Python 启动器
├── softmax_kernel.json    # Kernel 配置文件
└── softmax_test.py        # 测试用例
```

## 4. 核心算法

### 4.1 数值稳定优化

直接计算 `exp(x_i)` 可能导致上溢（尤其是 fp16，最大值约 65504）。

优化方法：
```
softmax(x_i) = exp(x_i - max(x)) / Σexp(x_j - max(x))
```

### 4.2 FP16 vs FP32

| 精度 | 数值范围 | 适用场景 |
|------|----------|----------|
| fp16 | ±65504 | 显存受限，batch size 大 |
| fp32 | ±3.4e38 | 精度要求高 |

## 5. AscendC Kernel 设计

### 5.1 核函数签名

```cpp
extern "C" __global__ __ascend__ void SoftmaxKernel(
    __gm__ void* input,
    __gm__ void* output,
    uint32_t* shape,
    uint32_t shapeSize,
    int32_t dim,
    int32_t dtype
);
```

### 5.2 计算流程

1. **初始化**: 从 shape 获取张量维度信息
2. **分块**: 将最后一维分成 256 元素块
3. **找最大值**: 每块内求 max(x_i - max)
4. **求指数和**: Σexp(x_i - max)
5. **归一化**: exp(x_i - max) / Σexp(x_i - max)

### 5.3 内存布局

- GM (Global Memory): 存储输入输出张量
- LM (Local Memory): 分块加载到本地进行计算
- 块大小: 256 元素
- Tile 大小: 32 元素

## 6. 接口

### 6.1 Python 接口

```python
from softmax_launcher import softmax_ascend, softmax_ascend_fp16, softmax_ascend_fp32, SoftmaxOpAscend

# 函数接口
result = softmax_ascend(x, dim=-1, dtype=np.float32)
result = softmax_ascend_fp16(x, dim=-1)
result = softmax_ascend_fp32(x, dim=-1)

# 类接口
op = SoftmaxOpAscend(dim=-1, dtype=np.float32)
result = op(x)
```

### 6.2 C++ 接口

```cpp
#include "softmax_kernel.h"

SoftmaxKernel softmaxKernel;
softmaxKernel.Init(shape, shapeSize, dim, dtype);
softmaxKernel.Process(input, output);
```

## 7. 测试用例

- [x] 基础 1D 向量
- [x] 2D 张量 dim=-1
- [x] FP16 精度
- [x] FP32 精度
- [x] 大数值稳定性 (数值稳定优化验证)
- [x] 负数输入
- [x] 3D 张量
- [x] 空张量

## 8. 编译和部署

### 8.1 编译命令

```bash
# 使用 AscendC 编译器
ascendc --gen.sh=SoftmaxKernel.gn \
        --kernel.name=SoftmaxKernel \
        --kernel.type=AscendC \
        softmax_kernel.cpp
```

### 8.2 运行

```python
from softmax_launcher import softmax_ascend
import numpy as np

x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
result = softmax_ascend(x, dim=-1, dtype=np.float32)
print(result)  # [[0.09003057, 0.24472847, 0.66524096]]
```

## 9. 性能考量

- 块大小 256 是 AscendC SIMD 最优选择
- 数值稳定优化避免上溢同时保持精度
- 分块计算减少寄存器压力
