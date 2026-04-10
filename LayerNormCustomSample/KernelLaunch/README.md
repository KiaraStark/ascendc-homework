## 概述
本样例介绍LayerNorm算子的核函数直调方法。
## 目录结构介绍
``` 
├── KernelLaunch                      // 使用核函数直调的方式调用LayerNorm自定义算子
│   └── LayerNormKernelInvocation           // host侧的核函数调用程序
``` 
## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1.&nbsp;获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。
### 2.&nbsp;编译运行样例工程
- [LayerNormKernelInvocation样例运行](./LayerNormKernelInvocation/README.md)
## 更新说明
  | 时间 | 更新事项 | 注意事项 |
|----|------|------|
| 2024/06/26 | 更新readme结构 |需要基于社区CANN包8.0.RC1.alpha003及之后版本运行 |