#pragma once
#include "common/op.hpp"

namespace infinicore::op {
// Packed weight is [up; gate], shape [2D,K]; neither bias nor alpha scaling.
// Preserves projection storage-dtype rounding before SwiGLU.
// Unsupported cases execute linear + split + swiglu; no launch-error retry.
Tensor projection_swiglu(const Tensor &input, const Tensor &packed_weight,
                         bool prefer_fusion = true);
}
