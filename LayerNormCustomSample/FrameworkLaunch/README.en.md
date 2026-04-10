## Overview
This example, based on the LayerNormCustom operator project, introduces the invocation method for single operator calls.

## Directory Structure Introduction
``` 
├── FrameworkLaunch           // Invokes the LayerNorm operator using the framework
│   ├── AclNNInvocation       // Invokes the LayerNormCustom operator using the aclnn method
│   ├── LayerNormCustom             // LayerNormCustom operator project
│   └── LayerNorm.json        // Prototype definition json file for the LayerNormCustom operator
``` 

## Operator Project Introduction
The operator project directory LayerNormCustom contains template files for operator implementation, compilation scripts, etc., as shown below:
``` 
├── LayerNormCustom         // LayerNorm custom operator project
│   ├── cmake
│   ├── framework           // Directory for operator plugin implementation files, not dependent on operator adaptation plugins for generating single operator model files, no need to pay attention to
│   ├── op_host             // Host-side implementation files
│   ├── op_kernel           // Kernel-side implementation files
│   ├── scripts             // Directory for custom operator project packaging-related scripts
│   ├── build.sh            // Compilation entry script
│   ├── CMakeLists.txt      // CMakeLists.txt for the operator project
│   └── CMakePresets.json   // Compilation configuration items
``` 
The CANN software package provides the project creation tool msopgen, and the LayerNormCustom operator project can be automatically created through LayerNorm.json. For details, please refer to the [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC) > Operator Development > Operator Development Project > Operator Development Based on Custom Operator Project > Creating Operator Project section.

## Compiling and Running the Example Operator
For custom operator projects, the compilation and running process includes the following steps:
- Compile the custom operator project to generate the operator installation package;
- Install the custom operator into the operator library;
- Invoke and execute the custom operator;

Detailed operations are as follows.

### 1. Obtain the Source Code Package
Before compiling this example, please refer to [Preparation: Obtain the Example Code](../README.en.md#codeready) to obtain the source code package.

### 2. Compile the Operator Project<a name="operatorcompile"></a>
Compile the custom operator project to build and generate the custom operator package.

  - Execute the following command to switch to the LayerNormCustom operator project directory.

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/LayerNormCustomSample/FrameworkLaunch/LayerNormCustom
    ```

  - Modify the ASCEND_CANN_PACKAGE_PATH in CMakePresets.json to the actual path after the CANN software package is installed.

    ```json
    {
        ……
        "configurePresets": [
            {
                    ……
                    "ASCEND_CANN_PACKAGE_PATH": {
                        "type": "PATH",
                        "value": "/usr/local/Ascend/ascend-toolkit/latest"   // Please replace with the actual path after the CANN software package is installed. eg: /home/HwHiAiUser/Ascend/ascend-toolkit/latest
                    },
                    ……
            }
        ]
    }
    ```
  - Execute the following command in the LayerNormCustom operator project directory to compile the operator project.

    ```bash
    ./build.sh
    ```
  After successful compilation, a build_out directory will be created in the current directory, and the custom operator installation package custom_opp_<target os>_<target architecture>.run will be generated in the build_out directory, for example, "custom_opp_ubuntu_x86_64.run".  
  Note: If you want to use the dump debugging function, you need to remove the configurations for Atlas training series products and Atlas 200/500 A2 inference products from op_host and CMakeLists.txt.

### 3. Deploy the Operator Package

Execute the following command to install the custom operator package in the path where the custom operator installation package is located.
  ```bash
  cd build_out
  ./custom_opp_<target os>_<target architecture>.run
  ```

After the command is successfully executed, the relevant files in the custom operator package will be deployed to the vendors/customize directory of the OPP operator library in the current environment.

### 4. Configure Environment Variables

Please select the corresponding command to configure environment variables according to the [installation method](https://hiascend.com/document/redirect/CannCommunityInstSoftware) of the CANN development toolkit package on the current environment.
  - Default path, root user installs CANN software package
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    ```
  - Default path, non-root user installs CANN software package
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    ```
  - Specified path install_path, installs CANN software package
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
    ```

### 5. Invoke and Execute the Operator Project
- [Invoke the LayerNormCustom Operator Project using aclnn](./AclNNInvocation/README.en.md)

## Update Log
  | Date | Update Items |
|----|------|
| 2024/06/25 | New version of readme updated |
| 2024/07/22 | Modified environment configuration for different user environments |