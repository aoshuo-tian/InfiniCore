#include "h3_flash_attention_profile.h"
#include "h3_profile_d64_f16_short.h"
#include "h3_profile_d64_f16_long.h"
#include "h3_profile_d64_bf16_short.h"
#include "h3_profile_d64_bf16_long.h"
#include "h3_profile_d72_f16_short.h"
#include "h3_profile_d72_f16_long.h"
#include "h3_profile_d72_bf16_short.h"
#include "h3_profile_d72_bf16_long.h"
#include "h3_profile_d128_f16_short.h"
#include "h3_profile_d128_f16_long.h"
#include "h3_profile_d128_bf16_short.h"
#include "h3_profile_d128_bf16_long.h"
NineToothedResult launch_h3_attention_profile(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype) {
    const int use_long = key.shape[2] > 512;
    if (dim==64 && dtype==INFINI_DTYPE_F16) return use_long ? launch_h3_profile_d64_f16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d64_f16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    if (dim==64 && dtype==INFINI_DTYPE_BF16) return use_long ? launch_h3_profile_d64_bf16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d64_bf16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    if (dim==72 && dtype==INFINI_DTYPE_F16) return use_long ? launch_h3_profile_d72_f16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d72_f16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    if (dim==72 && dtype==INFINI_DTYPE_BF16) return use_long ? launch_h3_profile_d72_bf16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d72_bf16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    if (dim==128 && dtype==INFINI_DTYPE_F16) return use_long ? launch_h3_profile_d128_f16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d128_f16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    if (dim==128 && dtype==INFINI_DTYPE_BF16) return use_long ? launch_h3_profile_d128_bf16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d128_bf16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    return 1;
}
