
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, var_M);
  TILING_DATA_FIELD_DEF(uint32_t, var_N);
  TILING_DATA_FIELD_DEF(uint32_t, var_K);
  TILING_DATA_FIELD_DEF(uint32_t, index_M);
  TILING_DATA_FIELD_DEF(uint32_t, index_N);
  TILING_DATA_FIELD_DEF(uint32_t, index_K);
  TILING_DATA_FIELD_DEF(uint32_t, updates_M);
  TILING_DATA_FIELD_DEF(uint32_t, updates_N);
  TILING_DATA_FIELD_DEF(uint32_t, updates_K);
  TILING_DATA_FIELD_DEF(uint32_t, axis);
  TILING_DATA_FIELD_DEF(uint32_t, reduce);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}
