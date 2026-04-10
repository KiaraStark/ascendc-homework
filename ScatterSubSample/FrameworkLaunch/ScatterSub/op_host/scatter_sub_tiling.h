/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterSubTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);
  TILING_DATA_FIELD_DEF(uint32_t, lastDim);
  TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
  TILING_DATA_FIELD_DEF(uint32_t, var1stDim);
  TILING_DATA_FIELD_DEF(uint32_t, firstTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterSub, ScatterSubTilingData)
}
