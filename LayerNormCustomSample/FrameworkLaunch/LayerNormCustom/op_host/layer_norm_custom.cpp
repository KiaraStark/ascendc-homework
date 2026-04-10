
#include "layer_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 48;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  LayerNormCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::Shape shape = x1_shape->GetStorageShape();
  auto rowNum = shape.GetDim(0);
  auto colNum = shape.GetDim(1);
  auto coreRowNum = rowNum / BLOCK_DIM;

  const float* epsilonAttr = context->GetAttrs()->GetAttrPointer<float>(0);
  tiling.set_eps(*epsilonAttr);
  tiling.set_rowNum(coreRowNum);
  tiling.set_rowNumSp(coreRowNum + 1);
  tiling.set_rowLength(colNum);
  tiling.set_blockPivot(rowNum - coreRowNum * BLOCK_DIM);
  uint32_t tileLoop = 18;
  tiling.set_tileLoop(tileLoop);
  tiling.set_tileLength(tileLoop * colNum);
  tiling.set_loopCount(coreRowNum / tileLoop);
  tiling.set_factor(1.0f / colNum);
  tiling.set_mfactor(-1.0f / colNum);

  context->SetBlockDim(BLOCK_DIM);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(1);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LayerNormCustom : public OpDef {
 public:
  explicit LayerNormCustom(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("res_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-05);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(LayerNormCustom);
}  // namespace ops
