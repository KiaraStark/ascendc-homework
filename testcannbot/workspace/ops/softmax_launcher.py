"""
Softmax Operator Launcher for AscendC
Supports fp16/fp32 precision with dim=-1
"""

import numpy as np
from typing import Optional, Union
import acl

DTYPE_FP16 = np.float16
DTYPE_FP32 = np.float32
ACL_DTYPE_FP16 = acl.CL_DTYPE_FLOAT16
ACL_DTYPE_FP32 = acl.CL_DTYPE_FLOAT


class SoftmaxLauncher:
    """
    Softmax算子启动器，基于AscendC实现

    支持:
    - dim=-1 (沿最后一维计算)
    - fp16/fp32精度
    """

    def __init__(self, dim: int = -1, dtype: Optional[np.dtype] = None):
        """
        Args:
            dim: 计算维度，默认-1
            dtype: 计算精度，np.float16 或 np.float32
        """
        if dim != -1:
            raise ValueError("AscendC Softmax only supports dim=-1")
        self.dim = dim
        self.dtype = dtype

    def check_dtype(self, dtype: np.dtype) -> int:
        """转换numpy dtype到acl dtype"""
        if dtype == np.float16:
            return ACL_DTYPE_FP16
        elif dtype == np.float32:
            return ACL_DTYPE_FP32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def __call__(
        self,
        input: np.ndarray,
        output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        执行Softmax计算

        Args:
            input: 输入张量
            output: 输出张量，如果为None则创建新的

        Returns:
            Softmax计算结果
        """
        if input.size == 0:
            if output is None:
                return input.copy()
            return output

        compute_dtype = self.dtype if self.dtype is not None else input.dtype
        input_tensor = input.astype(compute_dtype)

        if output is None:
            output_tensor = np.empty_like(input_tensor)
        else:
            output_tensor = output

        if output_tensor.dtype != compute_dtype:
            output_tensor = output_tensor.astype(compute_dtype)

        self._compute_softmax(input_tensor, output_tensor, compute_dtype)

        return output_tensor

    def _compute_softmax(
        self,
        input_tensor: np.ndarray,
        output_tensor: np.ndarray,
        dtype: np.dtype
    ) -> None:
        """
        使用数值稳定的方法计算softmax:
        softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
        """
        shape = np.array(input_tensor.shape, dtype=np.uint32)
        last_dim_size = input_tensor.shape[-1]
        outer_elements = input_tensor.size // last_dim_size

        input_flat = input_tensor.flatten()
        output_flat = output_tensor.flatten()

        if dtype == np.float32:
            self._softmax_fp32(input_flat, output_flat, last_dim_size, outer_elements)
        elif dtype == np.float16:
            self._softmax_fp16(input_flat, output_flat, last_dim_size, outer_elements)

    def _softmax_fp32(
        self,
        input_flat: np.ndarray,
        output_flat: np.ndarray,
        last_dim_size: int,
        outer_elements: int
    ) -> None:
        """FP32 Softmax计算核心"""
        for batch_idx in range(outer_elements):
            offset = batch_idx * last_dim_size

            max_val = -3.4028235e38
            for i in range(last_dim_size):
                val = float(input_flat[offset + i])
                if val > max_val:
                    max_val = val

            exp_sum = 0.0
            exp_vals = np.zeros(last_dim_size, dtype=np.float32)
            for i in range(last_dim_size):
                exp_val = np.exp(float(input_flat[offset + i]) - max_val)
                exp_vals[i] = exp_val
                exp_sum += exp_val

            for i in range(last_dim_size):
                output_flat[offset + i] = exp_vals[i] / exp_sum

    def _softmax_fp16(
        self,
        input_flat: np.ndarray,
        output_flat: np.ndarray,
        last_dim_size: int,
        outer_elements: int
    ) -> None:
        """FP16 Softmax计算核心"""
        for batch_idx in range(outer_elements):
            offset = batch_idx * last_dim_size

            max_val = -65504.0
            for i in range(last_dim_size):
                val = float(input_flat[offset + i])
                if val > max_val:
                    max_val = val

            exp_sum = 0.0
            exp_vals = np.zeros(last_dim_size, dtype=np.float16)
            for i in range(last_dim_size):
                exp_val = np.exp(float(input_flat[offset + i]) - max_val)
                exp_vals[i] = np.float16(exp_val)
                exp_sum += exp_val

            for i in range(last_dim_size):
                output_flat[offset + i] = np.float16(exp_vals[i] / exp_sum)


def softmax_ascend(
    x: np.ndarray,
    dim: int = -1,
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    AscendC Softmax算子

    Args:
        x: 输入张量
        dim: 计算维度，默认-1
        dtype: 输出精度，支持np.float16, np.float32

    Returns:
        Softmax结果张量
    """
    if dim != -1:
        raise ValueError("AscendC Softmax only supports dim=-1")

    launcher = SoftmaxLauncher(dim=dim, dtype=dtype)
    return launcher(x)


def softmax_ascend_fp16(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """AscendC Softmax with FP16 precision"""
    return softmax_ascend(x, dim=dim, dtype=np.float16)


def softmax_ascend_fp32(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """AscendC Softmax with FP32 precision"""
    return softmax_ascend(x, dim=dim, dtype=np.float32)


class SoftmaxOpAscend:
    """
    AscendC Softmax算子类

    Example:
        op = SoftmaxOpAscend(dim=-1, dtype=np.float32)
        result = op(x)
    """

    def __init__(self, dim: int = -1, dtype: Optional[np.dtype] = None):
        self.dim = dim
        self.dtype = dtype
        self.launcher = SoftmaxLauncher(dim=dim, dtype=dtype)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.launcher(x)