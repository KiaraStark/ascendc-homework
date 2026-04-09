
#include "scatter_elements_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ScatterElementsTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNum(); 

    auto shape_var = context->GetInputTensor(0)->GetOriginShape();
    auto shape_indices = context->GetInputTensor(1)->GetOriginShape();
    auto shape_updates = context->GetInputTensor(2)->GetOriginShape();
    uint32_t var_M, var_N, var_K;
    uint32_t updates_M, updates_N, updates_K;
    uint32_t index_M, index_N, index_K;

    if(shape_var.GetDimNum() == 1){
        var_M = 1;
        var_N = 1;
        var_K = shape_var.GetDim(0);
    }
    else if(shape_var.GetDimNum() == 2){
        var_M = 1;
        var_N = shape_var.GetDim(0);
        var_K = shape_var.GetDim(1);
    }
    else{
        var_M = shape_var.GetDim(0);
        var_N = shape_var.GetDim(1);
        var_K = shape_var.GetDim(2);
    }

    if(shape_updates.GetDimNum() == 1){
        updates_M = 1;
        updates_N = 1;
        updates_K = shape_updates.GetDim(0);
    }
    else if(shape_updates.GetDimNum() == 2){
        updates_M = 1;
        updates_N = shape_updates.GetDim(0);
        updates_K = shape_updates.GetDim(1);
    }
    else{
        updates_M = shape_updates.GetDim(0);
        updates_N = shape_updates.GetDim(1);
        updates_K = shape_updates.GetDim(2);
    }

    int32_t axis = *context->GetAttrs()->GetInt(0);
    if(shape_indices.GetDimNum() == 1){
        index_M = 1;
        index_N = 1;
        index_K = shape_indices.GetDim(0);
        axis += 2;
    }
    else if(shape_indices.GetDimNum() == 2){
        index_M = 1;
        index_N = shape_indices.GetDim(0);
        index_K = shape_indices.GetDim(1);
        axis += 1;
    }
    else{
        index_M = shape_indices.GetDim(0);
        index_N = shape_indices.GetDim(1);
        index_K = shape_indices.GetDim(2);
    }

    tiling.set_var_M(var_M);
    tiling.set_var_N(var_N);
    tiling.set_var_K(var_K);
    tiling.set_index_M(index_M);
    tiling.set_index_N(index_N);
    tiling.set_index_K(index_K);
    tiling.set_updates_M(updates_M);
    tiling.set_updates_N(updates_N);
    tiling.set_updates_K(updates_K);

    
    const char *str = context->GetAttrs()->GetAttrPointer<char>(1);
    int reduce;
    if (strcmp(str, "None") == 0) {
        reduce = 0;
    }
    else if(strcmp(str, "add") == 0) {
        reduce = 1;
    }
    else if(strcmp(str, "multipy") == 0){
        reduce = 2;
    }
    else{
        reduce = 0;
    }

    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_UINT8) {
        context->SetTilingKey(1);
    }
    else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        context->SetTilingKey(2);
    }
    else if (dt == ge::DT_FLOAT){
        context->SetTilingKey(3);
    }
    else{
        context->SetTilingKey(4);
    }


    tiling.set_axis(axis);
    tiling.set_reduce(reduce);
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
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduce").AttrType(OPTIONAL).String("None");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b")
                      .AddConfig("ascend310b");

    }
};

OP_ADD(ScatterElements);
}
