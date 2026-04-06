"""
Softmax Operator Tests
"""

import numpy as np
from softmax_op import softmax, softmax_fp16, softmax_fp32, SoftmaxOp


def nearly_equal(a, b, rtol=1e-3, atol=1e-4):
    """检查两个数组是否近似相等"""
    return np.allclose(a, b, rtol=rtol, atol=atol)


def test_softmax_basic():
    """基础测试：1D向量"""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = softmax(x, dim=-1)
    expected = np.array([0.09003057, 0.24472847, 0.66524096], dtype=np.float32)
    assert nearly_equal(result, expected), f"Expected {expected}, got {result}"
    print("test_softmax_basic PASSED")


def test_softmax_2d_last_dim():
    """2D张量，dim=-1"""
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    result = softmax(x, dim=-1)
    # 每行和为1
    row_sums = np.sum(result, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), f"Row sums should be 1.0, got {row_sums}"
    # 验证数值
    expected_row0 = np.array([0.09003057, 0.24472847, 0.66524096])
    expected_row1 = np.array([0.09003057, 0.24472847, 0.66524096])
    assert nearly_equal(result[0], expected_row0)
    assert nearly_equal(result[1], expected_row1)
    print("test_softmax_2d_last_dim PASSED")


def test_softmax_fp16():
    """fp16精度测试"""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = softmax_fp16(x, dim=-1)
    assert result.dtype == np.float16
    row_sum = np.sum(result)
    assert np.isclose(row_sum, 1.0, atol=1e-3), f"Sum should be 1.0, got {row_sum}"
    print("test_softmax_fp16 PASSED")


def test_softmax_fp32():
    """fp32精度测试"""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    result = softmax_fp32(x, dim=-1)
    assert result.dtype == np.float32
    row_sum = np.sum(result)
    assert np.isclose(row_sum, 1.0, atol=1e-6), f"Sum should be 1.0, got {row_sum}"
    print("test_softmax_fp32 PASSED")


def test_softmax_class():
    """SoftmaxOp 类测试"""
    op = SoftmaxOp(dim=-1, dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = op(x)
    expected = np.array([0.09003057, 0.24472847, 0.66524096])
    assert nearly_equal(result, expected)
    print("test_softmax_class PASSED")


def test_softmax_large_values():
    """大数值稳定性测试"""
    x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    result = softmax(x, dim=-1)
    # softmax后最大元素应对应最大的输入值
    assert result[2] > result[1] > result[0], "Largest input should have largest probability"
    # 验证和为1
    assert np.isclose(np.sum(result), 1.0), f"Sum should be 1.0, got {np.sum(result)}"
    print("test_softmax_large_values PASSED")


def test_softmax_negative():
    """负数输入测试"""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    result = softmax(x, dim=-1)
    expected = np.array([0.09003057, 0.24472847, 0.66524096])
    assert nearly_equal(result, expected)
    print("test_softmax_negative PASSED")


def test_softmax_3d_tensor():
    """3D张量测试"""
    x = np.random.randn(2, 3, 4).astype(np.float32)
    result = softmax(x, dim=-1)
    # 沿最后一维求和应为1
    sums = np.sum(result, axis=-1)
    assert np.allclose(sums, np.ones_like(sums)), f"Last dim sums should be 1.0"
    print("test_softmax_3d_tensor PASSED")


def test_softmax_empty():
    """空张量测试"""
    x = np.array([], dtype=np.float32)
    result = softmax(x, dim=-1)
    assert result.shape == (0,)
    print("test_softmax_empty PASSED")


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax_2d_last_dim()
    test_softmax_fp16()
    test_softmax_fp32()
    test_softmax_class()
    test_softmax_large_values()
    test_softmax_negative()
    test_softmax_3d_tensor()
    test_softmax_empty()
    print("\nAll tests PASSED!")
