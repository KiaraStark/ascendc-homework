## 目录结构介绍
``` 
├── LayerNormKernelInvocation
│   ├── cmake                   // 编译工程文件
│   ├── input                   // 存放脚本生成的输入数据目录
│   ├── output                  // 存放算子运行输出数据和真实数据的目录
│   ├── scripts
│   │   ├── acl.json            // acl配置文件
│   │   ├── gen_data.py         // 验证输出数据和真值数据是否一致的验证脚本
│   │   └── verify_result.py    // 真值对比文件
│   ├── layer_norm_custom.cpp   // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
``` 
## 代码实现介绍
本样例中实现的是固定shape的LayerNormCustom算子。
- kernel实现   
 LayerNorm算子的功能为：LayerNorm算子是计算归一化的一种算子，该算子有3个输入x, beta, gamma。根据系数beta 和偏置gamma对输入数据x收敛到固定区间。

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口和beta, gamma的修正完成对输入x的计算，得到最终结果，再搬出到外部存储上。   

  LayerNorm算子的实现流程分为3个基本任务：CopyIn, Compute, CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm, betaGm和gammaGm搬运至Local Memory，分别存储在xLocal, betaLocal, gammaLocal, Compute任务负责对xLocal, betaLocal, gammaLocal执行相关操作，计算结果存储在outLocal中，CopyOut任务负责将输出数据从outLocal搬运至Global Memory上的输出Tensor outGm中。具体请参考[layer_norm_custom.cpp](./layer_norm_custom.cpp)。

- 调用实现  
  1.&nbsp;CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；  
  2.&nbsp;NPU侧运行验证主要通过使用<<<>>>内核调用符来完成。    
应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。
## 运行样例算子
- 打开样例目录

  ```bash
  cd ${git_clone_path}/samples/operator_contrib/LayerNormCustomSample/KernelLaunch/LayerNormKernelInvocation/
  ```
- 配置环境变量

    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ````
    
    配置仿真模式日志文件目录，默认为sim_log。
    ```bash
    export CAMODEL_LOG_PATH=./sim_log
    ```

- 样例执行

  ```bash
  bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
  ```
  - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]，默认值为cpu。
  - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
    - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

  示例如下。
  ```bash
  bash run.sh -r cpu -v Ascend910B4
  ```   
## 更新说明
  | 时间 | 更新事项 |
|----|------|
| 2023/6/25 | 新增本readme |