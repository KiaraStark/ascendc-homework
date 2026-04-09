
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, TilingData)
}
