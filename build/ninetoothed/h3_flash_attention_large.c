#include "h3_flash_attention_large.h"
#include "h3_large_d128_f16_medium.h"
#include "h3_large_d128_f16_large.h"
#include "h3_large_d128_bf16_medium.h"
#include "h3_large_d128_bf16_large.h"
#ifdef ENABLE_NINETOOTHED_HYGON_MHA_INNER32
#include "h3_flash_attention_inner32.h"
#endif
#ifdef H3_LARGE_TEST_COUNTER
static thread_local uint64_t medium_calls=0,wide_calls=0;
extern "C" uint64_t h3_large_calls() { return medium_calls+wide_calls; }
extern "C" uint64_t h3_medium_profile_calls() { return medium_calls; }
extern "C" uint64_t h3_wide_profile_calls() { return wide_calls; }
#endif
NineToothedResult launch_h3_attention_large(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype) {
    int wide = query.shape[2] >= 2048 && key.shape[2] >= 2048;
#ifdef ENABLE_NINETOOTHED_HYGON_MHA_HEAD56
    // H3 B1 self-attention only; preserve all other profile choices.
    wide = wide || (query.shape[0] == 1 && key.shape[0] == 1
        && query.shape[1] == 56 && key.shape[1] == 56
        && query.shape[2] == key.shape[2]
        && query.shape[2] >= 1024 && query.shape[2] < 2048);
#endif
#ifdef H3_LARGE_TEST_COUNTER
    if (wide) ++wide_calls; else ++medium_calls;
#endif
#ifdef ENABLE_NINETOOTHED_HYGON_MHA_INNER32
    if (h3_inner32_eligible(query, key, value, total_kv_length, output, dim, dtype)) {
        /* Masking is disabled in this path; provide valid unused metadata. */
        uint64_t unused_shape[4] = {1, 1, 1, 1};
        int64_t unused_strides[4] = {16, 16, 16, 1};
        NineToothedTensor unused_mask = {0, unused_shape, unused_strides};
        return dtype == INFINI_DTYPE_F16
            ? launch_h3_long_inner32_f16_m128n32(stream, query, key, value, total_kv_length, unused_mask, is_causal, scale, output, with_attn_mask, causal_variant)
            : launch_h3_long_inner32_bf16_m128n32(stream, query, key, value, total_kv_length, unused_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    }
#endif
    if (dim==128 && dtype==INFINI_DTYPE_F16) return wide ? launch_h3_large_d128_f16_large(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_large_d128_f16_medium(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    if (dim==128 && dtype==INFINI_DTYPE_BF16) return wide ? launch_h3_large_d128_bf16_large(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_large_d128_bf16_medium(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);
    return 1;
}
