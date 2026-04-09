
#include "softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include<tiling/tiling_api.h>
#include<vector>
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    TilingData tiling;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum(); 

    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    const uint32_t dimNum = shape_x.GetDimNum();
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT16) {
        context->SetTilingKey(1);
    }
    else{
        context->SetTilingKey(2);
    }

    int32_t dim = *context->GetAttrs()->GetInt(0);
    if(dim < 0){
        dim += dimNum;
    }
    int32_t width = shape_x.GetDim(dim);
    int32_t height = totalLength / width;
    
    tiling.set_height(height);
    tiling.set_width(width);

    context->SetBlockDim(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Softmax : public OpDef {
public:
    explicit Softmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b")
                      .AddConfig("ascend310b");

    }
};

OP_ADD(Softmax);
}
