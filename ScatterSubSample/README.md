## ScatterSub自定义算子样例说明 
本样例通过Ascend C编程语言实现了ScatterSub算子，并按照不同的算子调用方式分别给出了对应的端到端实现。
- [FrameworkLaunch](./FrameworkLaunch)：使用框架调用ScatterSub自定义算子。  
按照工程创建->算子实现->编译部署>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。
- [KernelLaunch](./KernelLaunch)：使用核函数直调ScatterSub自定义算子。  
核函数的基础调用（Kernel Launch）方式，开发者完成算子核函数的开发和Tiling实现后，即可通过AscendCL运行时接口，完成算子的调用。

本样例中包含如下调用方式：
<table>
    <th>调用方式</th><th>目录</th><th>描述</th>
    <tr>
        <!-- 列的方向占据1个cell -->
        <td rowspan='1'><a href="./FrameworkLaunch"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation"> AclNNInvocation</td><td>通过aclnn调用的方式调用ScatterSubCustom算子。</td>
    </tr>
    <tr>
        <!-- 列的方向占据1个cell -->
        <td rowspan='1'><a href="./KernelLaunch"> KernelLaunch</td><td><a href="./KernelLaunch/ScatterSubKernelInvocation"> ScatterSubKernelInvocation</td><td>host侧的核函数调用程序，包含CPU侧和NPU侧两种运行验证方法。</td>
    </tr>
</table>

## 算子描述
将var中的数据用indices进行索引, 索引结果与updates进行减法操作。
```
    # Scalar indices
    ref[indices, ...] -= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
```
## 算子规格描述
<!-- <table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">48 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">48 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">48 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table> -->
<table>  
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">ScatterSub</th></tr>  
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
<tr><td align="center">var</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">indices</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">updates</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td><td align="center">var</td><td align="center">-</td><td align="center">float3,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">attr属性</td><td align="center">use_locking</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">false</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="8" align="center">scattersub_custom</td></tr>  
</table>


## 支持的产品型号
本样例支持如下产品型号：
- Atlas 200/500 A2推理产品 
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 目录结构介绍
```
├── FrameworkLaunch    //使用框架调用的方式调用ScatterSub自定义算子工程。
└── KernelLaunch       //使用核函数直调的方式调用ScatterSub自定义算子。
```
## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 编译运行样例算子

### 1. 准备：获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitee.com/ascend/samples.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```bash
   git checkout v0.5.0
   ```
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。

   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**
   ```bash
   # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
   # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/ascend-samples-master.zip】。
   # 3. 开发环境中，执行以下命令，解压zip包。
   cd ${git_clone_path}
   unzip ascend-samples-master.zip
   ```
### 2. 编译运行样例工程
- 若使用框架调用的方式，编译运行操作请参见[FrameworkLaunch](./FrameworkLaunch)。    
- 若使用核函数直调的方式，编译运行操作请参见[KernelLaunch](./KernelLaunch)。
## 更新说明
  | 时间 | 更新事项 |
|----|------|
| 2024/05/24 | 新版readme更新 |
| 2024/07/22 | 修改clone到任意目录 |