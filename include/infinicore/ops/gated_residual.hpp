#pragma once
#include "common/op.hpp"
namespace infinicore::op {
// residual + stored_dtype(gate * branch); retain the original product rounding.
// The optional Hygon path requires --ninetoothed-hygon-gated-residual=y.
Tensor gated_residual(const Tensor &residual, const Tensor &gate,
                      const Tensor &branch, bool prefer_fusion = true);
}
