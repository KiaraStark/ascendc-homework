#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LogSumExp)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(dim, ListInt, {})
    .ATTR(keep_dim, Bool, false)
    .OP_END_FACTORY_REG(LogSumExp);

}

#endif
