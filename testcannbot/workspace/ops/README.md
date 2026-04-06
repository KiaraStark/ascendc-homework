# Softmax AscendC Operator

## 项目结构

```
ops/
├── softmax_op.py              # 纯numpy实现的softmax算子
├── softmax_launcher.py        # AscendC风格的启动器（numpy模拟）
├── softmax_kernel.cpp/h       # AscendC kernel源码
├── softmax_kernel_simple.cpp/h # 简化的kernel实现（可用于测试）
├── softmax_kernel.json        # 算子定义
├── softmax_test.py            # 原始测试脚本
├── test_softmax_kernel.py      # 扩展测试脚本
├── libsoftmax_kernel.a        # 编译后的静态库
├── CMakeLists.txt             # CMake构建配置
└── run_tests.sh               # 构建和测试脚本
```

## 构建和测试

### 运行所有测试

```bash
cd /home/HwHiAiUser/homework/testcannbot/workspace/ops
./run_tests.sh
```

### 仅运行Python测试

```bash
python3 test_softmax_kernel.py
```

### 在AscendC环境中编译kernel

AscendC kernel需要在aarch64架构的AscendC环境中编译：

```bash
# 在AscendC环境中
cd ops
mkdir build && cd build
cmake ..
make
```

## 重要说明

1. **AscendC kernel编译** (`softmax_kernel.cpp`) 需要：
   - aarch64架构的机器（或交叉编译工具链）
   - 完整的AscendC CANN开发环境
   - 正确的include路径配置

2. **简化版本** (`softmax_kernel_simple.cpp`) 可以在任何x86平台上编译，用于：
   - 验证算法逻辑
   - 快速原型开发
   - 测试框架验证

3. **numpy实现** (`softmax_op.py`) 是纯Python实现，不依赖任何硬件，可以直接运行测试。

## 测试覆盖

- 基础1D/2D/3D张量测试
- FP16/FP32精度测试
- 大数值稳定性测试
- 负数输入测试
- 空张量测试
- Kernel编译测试
