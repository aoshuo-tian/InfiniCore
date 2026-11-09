#pragma once

#include "common/op.hpp"

namespace infinicore::op {

// Preserve H3's three materialization/rounding steps. The optional Hygon
// implementation is enabled only by --ninetoothed-hygon-adaln=y at build time.
// prefer_fusion=false explicitly retains the original composition.
Tensor adaln(const Tensor &normalized, const Tensor &scale,
             const Tensor &shift, bool prefer_fusion = true);

} // namespace infinicore::op
