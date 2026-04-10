
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowNum);
  TILING_DATA_FIELD_DEF(uint32_t, rowNumSp);
  TILING_DATA_FIELD_DEF(uint32_t, rowLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockPivot);
  TILING_DATA_FIELD_DEF(uint32_t, tileLoop);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, loopCount);
  TILING_DATA_FIELD_DEF(float, factor);
  TILING_DATA_FIELD_DEF(float, mfactor);
  TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormCustom, LayerNormCustomTilingData)
}
