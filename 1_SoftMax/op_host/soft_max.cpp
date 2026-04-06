#include "soft_max_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t USED_CORE_NUM = 40;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftmaxCustomTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::Shape shape = x1_shape->GetStorageShape();
    auto rowNum = shape.GetDim(0);
    auto colNum = shape.GetDim(1); // colNum should be 32Byte aligned

    SoftmaxCustomTiling::ComputeTiling(rowNum, colNum, USED_CORE_NUM, tiling);

    context->SetBlockDim(USED_CORE_NUM);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
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
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("max")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("sum")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(SoftmaxCustom);
}
