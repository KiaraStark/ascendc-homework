
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplicationPad2dTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, loopCount);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReplicationPad2d, ReplicationPad2dTilingData)
}
