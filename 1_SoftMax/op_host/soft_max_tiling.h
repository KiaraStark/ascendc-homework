#ifndef SOFTMAX_TILLING_H
#define SOFTMAX_TILLING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, columnLength);
  TILING_DATA_FIELD_DEF(uint32_t, rowLength);
  TILING_DATA_FIELD_DEF(uint32_t, sharedTmpBufferSize);
  TILING_DATA_FIELD_DEF(uint32_t, usedBlockDim);
  TILING_DATA_FIELD_DEF(uint32_t, coreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, singleLoopCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreLoopCount);
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreLoopTail);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleLoopCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleCoreLoopCount);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleCoreLoopTail);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);//相当于SoftMaxTiling softmaxTilingData;用来处理核内 Tiling
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)//将算子类的名称与上面的结构体关联
}

namespace SoftmaxCustomTiling {
constexpr uint32_t SHARED_TMP_BUFFER_SIZE = 61440; // reserved tmpbuffer 60K for softmax compute
struct SingleCoreLoopParam { //定义结构体并初始化
    uint32_t singleLoopCoreRowNum{ 0 };  // row num processed in single loop
    uint32_t singleCoreLoopCount{ 0 };   // loop count in single loop
    uint32_t singleCoreLoopTail{ 0 };    // row num of last loop in single core
};

const std::vector<std::pair<uint32_t, uint32_t>> SLICE_TABLE = {
    // {reduce axis length, slice factor}
    //如果一行数据很长（如 8192），单次循环处理 1 行即可占满计算单元进行softmax归一化；如果一行数据很短（如 256），为了充分利用算力，可以在一个循环中打包处理多行（如 32 行）
    { 8192, 1 }, { 4096, 2 }, { 2048, 4 }, { 1024, 8 }, { 512, 16 }, { 256, 32 }, { 0, 64 }
};

SingleCoreLoopParam GetSingleCoreLoopParam(const uint32_t colNum, const uint32_t coreRowNum)
{
    //  Determine the params of single core based on the reduce axis length
    for (auto param : SLICE_TABLE) {
        if (colNum >= param.first) {
            SingleCoreLoopParam singleCoreLoopParam;
            singleCoreLoopParam.singleLoopCoreRowNum = param.second;//一个核单次compute处理的数据行数
            singleCoreLoopParam.singleCoreLoopCount = coreRowNum / param.second;//每次都打满，单个核总共要处理的compute次数
            singleCoreLoopParam.singleCoreLoopTail = coreRowNum % param.second;//尾部数据行数余数
            return singleCoreLoopParam;
        }
    }
    return SingleCoreLoopParam{};
}

void ComputeTiling(const uint32_t rowNum, const uint32_t colNum, const uint32_t coreNum,
                   optiling::SoftmaxCustomTilingData& tiling)
{
    uint32_t localworkspaceSize = SHARED_TMP_BUFFER_SIZE;
    auto alignedRowNum = (rowNum + coreNum - 1) / coreNum * coreNum;//向上对齐 (假设你有 105 行数据,8 个 Core,直接除是 13.125),代码通过将行数向上对齐到 Core 数量的倍数（112）,确保每个 Core 分配到的blockLength是整数(14)
    auto coreRowNum = alignedRowNum / coreNum;  // each core equal distribution，相当于blockLength
    auto tailCoreRowNum = rowNum % coreRowNum;  // last core process the tail rownum,最后要处理的“尾数”行数
    auto usedBlockDim = rowNum / coreRowNum;    // the core num used actually,blockDim

    SingleCoreLoopParam mainCoreLoopParam = GetSingleCoreLoopParam(colNum, coreRowNum);
    SingleCoreLoopParam tailCoreLoopParam;
    if (usedBlockDim == coreNum && tailCoreRowNum == 0) {
        tailCoreLoopParam = GetSingleCoreLoopParam(colNum, coreRowNum);
    } else {
        tailCoreLoopParam = GetSingleCoreLoopParam(colNum, tailCoreRowNum);
    }

    tiling.set_columnLength(colNum);
    tiling.set_rowLength(rowNum);
    tiling.set_sharedTmpBufferSize(localworkspaceSize);
    tiling.set_usedBlockDim(usedBlockDim);
    tiling.set_coreRowNum(coreRowNum);
    tiling.set_tailCoreRowNum(tailCoreRowNum);

    tiling.set_singleLoopCoreRowNum(mainCoreLoopParam.singleLoopCoreRowNum);
    tiling.set_singleCoreLoopCount(mainCoreLoopParam.singleCoreLoopCount);
    tiling.set_singleCoreLoopTail(mainCoreLoopParam.singleCoreLoopTail);
    tiling.set_tailCoreSingleLoopCoreRowNum(tailCoreLoopParam.singleLoopCoreRowNum);
    tiling.set_tailCoreSingleCoreLoopCount(tailCoreLoopParam.singleCoreLoopCount);
    tiling.set_tailCoreSingleCoreLoopTail(tailCoreLoopParam.singleCoreLoopTail);

    // get SoftMax Tiling
    ge::Shape softmaxComputeShape({ mainCoreLoopParam.singleLoopCoreRowNum, colNum });//初始化张量维度，这里的张量维度指的一次打满的compute维度
    AscendC::SoftMaxTilingFunc(softmaxComputeShape, sizeof(float), localworkspaceSize, tiling.softmaxTilingData);//这里是底层自己在处理核内Tiling策略
}
}  // namespace SoftmaxCustomTiling
#endif // EXAMPLES_ACTIVATION_SOFTMAX_CUSTOM_TILING_H