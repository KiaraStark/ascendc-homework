
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, varShape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, indicesShape);
    TILING_DATA_FIELD_DEF(int64_t, dimNum);
    TILING_DATA_FIELD_DEF(int64_t, mode);
    TILING_DATA_FIELD_DEF(int64_t, axis);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}
