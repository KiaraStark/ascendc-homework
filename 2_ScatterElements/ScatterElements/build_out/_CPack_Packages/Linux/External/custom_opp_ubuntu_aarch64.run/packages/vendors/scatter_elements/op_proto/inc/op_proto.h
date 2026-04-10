#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ScatterElements)
    .INPUT(var, ge::TensorType::ALL())
    .INPUT(indices, ge::TensorType::ALL())
    .INPUT(updates, ge::TensorType::ALL())
    .OUTPUT(var, ge::TensorType::ALL())
    .ATTR(axis, Int, 0)
    .ATTR(reduce, String, "None")
    .OP_END_FACTORY_REG(ScatterElements);

}

#endif
