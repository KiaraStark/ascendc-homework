# Softmax 算子设计文档

## 1. 概述

Softmax 算子将一个 K 维向量缩放到 (0,1) 区间，且所有元素和为 1，常用于多分类任务的输出层。

公式：
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

## 2. 功能规格

| 特性 | 说明 |
|------|------|
| 计算维度 | dim=-1 (沿最后一维) |
| 输入精度 | fp16, fp32 |
| 数值稳定 | 减最大值后再指数运算 |
| 输出范围 | (0, 1)，各维度和为 1 |

## 3. 实现细节

### 3.1 数值稳定优化

直接计算 `exp(x_i)` 可能导致上溢。优化方法：
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

### 3.2 精度支持

- `np.float16`: 半精度，适合显存受限场景
- `np.float32`: 单精度，默认推荐

## 4. 接口

### 函数接口

```python
softmax(x, dim=-1, dtype=None) -> np.ndarray
softmax_fp16(x, dim=-1) -> np.ndarray
softmax_fp32(x, dim=-1) -> np.ndarray
```

### 类接口

```python
SoftmaxOp(dim=-1, dtype=None)
op(x) -> np.ndarray
```

## 5. 测试用例

- [x] 基础1D向量
- [x] 2D张量 dim=-1
- [x] fp16精度
- [x] fp32精度
- [x] 大数值稳定性
- [x] 负数输入
- [x] 3D张量
- [x] 空张量
