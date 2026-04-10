#!/bin/bash
clear;clear
rm -rf *_cpu *_sim *_npu build cceprint npuchk output/*.bin input/*.bin > /dev/null
export PRINT_TIK_MEM_ACCESS=FALSE

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR

declare -A VersionMap
VersionMap["Ascend910A"]="Ascend910A"
VersionMap["Ascend910B"]="Ascend910A"
VersionMap["Ascend910ProA"]="Ascend910A"
VersionMap["Ascend910ProB"]="Ascend910A"
VersionMap["Ascend910PremiumA"]="Ascend910A"
VersionMap["Ascend310B1"]="Ascend310B1"
VersionMap["Ascend310B2"]="Ascend310B1"
VersionMap["Ascend310B3"]="Ascend310B1"
VersionMap["Ascend310B4"]="Ascend310B1"
VersionMap["Ascend310P1"]="Ascend310P1"
VersionMap["Ascend310P3"]="Ascend310P1"
VersionMap["Ascend910B1"]="Ascend910B1"
VersionMap["Ascend910B2"]="Ascend910B1"
VersionMap["Ascend910B3"]="Ascend910B1"
VersionMap["Ascend910B4"]="Ascend910B1"
# legacy
VersionMap["ascend910"]="Ascend910A"
VersionMap["ascend310p"]="Ascend310P1"
VersionMap["ascend310B1"]="Ascend310B1"
VersionMap["ascend910B1"]="Ascend910B1"

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

FILE_NAME="layer_norm_custom"

function useage() {
    echo "eg.:"
    echo "INFO: bash run.sh RUN_MODE"
    echo "INFO: RUN_MODE    should be in [cpu sim npu]"
}

if [ $# != 1 ]; then
    echo "ERROR: Need two params."
    useage
    exit
fi

SOC_VERSION=Ascend910B1
CORE_TYPE="VectorCore"
RUN_MODE_LIST="cpu sim npu"
RUN_MODE=$1
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

if [ "$RUN_MODE" = "sim" ]; then
    export LD_LIBRARY_PATH=${ASCEND_HOME_DIR}/runtime/lib64/stub:${LD_LIBRARY_PATH}
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=./sim_log
    fi
    rm -rf $CAMODEL_LOG_PATH
    mkdir -p $CAMODEL_LOG_PATH
fi

source $ASCEND_HOME_DIR/bin/setenv.bash

python3 scripts/gen_data.py

function main() {
    cd $CURRENT_DIR; rm -rf build; mkdir -p build; cd build
    cmake ..                               \
        -Dsmoke_testcase=${FILE_NAME}      \
        -DASCEND_PRODUCT_TYPE=$SOC_VERSION \
        -DASCEND_CORE_TYPE=${CORE_TYPE}    \
        -DASCEND_RUN_MODE=${RUN_MODE}      \
        -DASCEND_INSTALL_PATH=${ASCEND_HOME_DIR}
    cmake --build . --target ${FILE_NAME}_${RUN_MODE}
    if [ $? -ne 0 ]; then
        echo "ERROR: compile op on failed!"
        return 1
    fi
    cd -
    echo "INFO: compile op on ${RUN_MODE} succeed!"

    (export LD_LIBRARY_PATH=`pwd`:${ASCEND_HOME_DIR}/tools/simulator/$SOC_VERSION/lib:$LD_LIBRARY_PATH && ./${FILE_NAME}_${RUN_MODE})
    if [ $? -ne 0 ]; then
        echo "ERROR: execute op on ${RUN_MODE} failed!"
        return 1
    fi
    echo "INFO: execute op on ${RUN_MODE} succeed!"
    python3 scripts/verify_result.py output/output_z.bin output/golden.bin 
}
main
