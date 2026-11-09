#pragma once
#include "infinicore.h"
#include "../../../../dependencies/abi/ninetoothed.h"
#ifdef __cplusplus
extern "C" {
#endif
NineToothedResult launch_h3_attention_profile(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype);
#ifdef __cplusplus
}
#endif
