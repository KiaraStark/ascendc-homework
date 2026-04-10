## 目录结构介绍
``` 
├── ScatterSubKernelInvocation
│   ├── cmake                   // 编译工程文件
│   ├── input                   // 存放脚本生成的输入数据目录
│   ├── output                  // 存放算子运行输出数据和真实数据的目录
│   ├── scripts
│   │   ├── acl.json            // acl配置文件
│   │   ├── gen_data.py         // 验证输出数据和真值数据是否一致的验证脚本
│   │   └── verify_result.py    // 真值对比文件
│   ├── scatter_sub_custom.cpp      // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
``` 
## 代码实现介绍
本样例中实现的是固定shape为3 * 4 * 24 * 24的ScatterSub算子。
- kernel实现   
  ScatterSub算子的数学表达式为：  
  ```
    # Scalar indices
    ref[indices, ...] -= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
   ```
  计算逻辑是：首先判断输入Var中需要进行减法操作的维度，是否是32B对齐。如果是32对齐，则对这些维度进行矢量减法操作。如果是32B非对齐，则需要将每一个标量值取出来进行标量减法操作。
  
  ScatterSub中32B对齐的输入的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。对于非对齐的输入，则只需要进行循环遍历，执行标量减法即可。具体请参考[scatter_sub_custom.cpp](./scatter_sub_custom.cpp)。

- 调用实现  
  1.&nbsp;CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；  
  2.&nbsp;NPU侧运行验证主要通过使用<<<>>>内核调用符来完成。    
应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。
## 运行样例算子
  - 打开样例目录

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/ScatterSubCustomSample/KernelLaunch/ScatterSubKernelInvocation
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
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas 训练系列产品
      - Atlas A2训练系列产品

    注：针对Atlas 训练系列产品使用NPU仿真调试，会存在精度问题，可选择其他芯片进行NPU仿真调试。

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```   
## 更新说明
  | 时间 | 更新事项 |
|----|------|
| 2023/5/24 | 新增本readme |
| 2024/07/24 | 修改readme格式 |