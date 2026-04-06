"""
Softmax Operator Implementation

支持 dim=-1 (沿最后一维计算), 输入精度支持 fp16 和 fp32
"""

import numpy as np
from typing import Optional, Union

DTYPE_FP16 = np.float16
DTYPE_FP32 = np.float32


def softmax(x: np.ndarray, dim: int = -1, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Softmax operator along specified dimension.

    Args:
        x: 输入张量
        dim: 计算维度，默认 -1 (最后一维)
        dtype: 输出数据类型，支持 np.float16, np.float32. 若为 None，则与输入相同

    Returns:
        Softmax 后的张量，shape 与输入相同
    """
    if x.size == 0:
        return x.copy()

    # 确定计算维度索引
    if dim == -1:
        axis = x.ndim - 1
    else:
        axis = dim

    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Invalid dim={dim} for tensor with {x.ndim} dimensions")

    # 转换类型进行计算
    compute_dtype = dtype if dtype is not None else x.dtype
    if compute_dtype not in (DTYPE_FP16, DTYPE_FP32):
        raise ValueError(f"Unsupported dtype: {compute_dtype}")

    x_compute = x.astype(compute_dtype)

    # 计算 softmax: exp(x - max) / sum(exp(x - max))
    # 减去最大值保证数值稳定
    x_max = np.max(x_compute, axis=axis, keepdims=True)
    x_exp = np.exp(x_compute - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    result = x_exp / x_sum

    return result


class SoftmaxOp:
    """
    Softmax 算子类，支持 fp16/fp32 精度
    """

    def __init__(self, dim: int = -1, dtype: Optional[np.dtype] = None):
        """
        Args:
            dim: 沿哪个维度计算 softmax，默认 -1
            dtype: 计算精度，None 表示与输入相同
        """
        self.dim = dim
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return softmax(x, dim=self.dim, dtype=self.dtype)


def softmax_fp16(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Softmax with fp16 precision"""
    return softmax(x, dim=dim, dtype=DTYPE_FP16)


def softmax_fp32(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Softmax with fp32 precision"""
    return softmax(x, dim=dim, dtype=DTYPE_FP32)
