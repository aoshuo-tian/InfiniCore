import os,json,time,hashlib
from pathlib import Path
import ninetoothed as nt
from triton.runtime import driver
from infiniop.ops.flash_attention.ninetoothed import flash_attention as fa
import argparse
from ninetoothed.aot import _HEADER_PATH
parser=argparse.ArgumentParser(description="Generate guarded D128 long-sequence NineToothed kernels")
parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[5]/"build/ninetoothed")
out=parser.parse_args().output
os.environ.setdefault("NINETOOTHED_AOT_VARIANTS","generic")
out.mkdir(parents=True,exist_ok=True)
profiles=[('medium',64,32,4,1),('large',128,32,4,1)]
assert driver.active.get_current_target().arch=='gfx936'
records=[]
for dtype,label in [(nt.float16,'f16'),(nt.bfloat16,'bf16')]:
    for profile,m,n,warps,stages in profiles:
        name=f'h3_large_d128_{label}_{profile}'
        begin=time.monotonic();print('LARGE_GENERATE',name,profile,flush=True)
        arrange,app,tensors=fa.premake(with_kv_cache=0,emb_dim=128,is_causal=0,with_attn_mask=0,
            causal_variant=fa.CausalVariant.LOWER_RIGHT,dtype=dtype,block_size_m=m,block_size_n=n)
        nt.make(arrange,app,tensors,caller='cuda',kernel_name=name,output_dir=out,num_warps=warps,num_stages=stages)
        records.append({'name':name,'dtype':label,'profile':profile,'m':m,'n':n,'warps':warps,'stages':stages,'seconds':time.monotonic()-begin})
result={'target':str(driver.active.get_current_target()),'profiles':profiles,'records':records,
    'expression_path':str(Path(fa.__file__).resolve()),'expression_sha256':hashlib.sha256(Path(fa.__file__).read_bytes()).hexdigest()}
(out/'h3-large-generation.json').write_text(json.dumps(result,indent=2))
print('LARGE_GENERATION_PASS',len(records),flush=True)

(out/"h3_flash_attention_large.h").write_text('#pragma once\n#include "infinicore.h"\n#include "ninetoothed.h"\n#ifdef __cplusplus\nextern "C" {\n#endif\nNineToothedResult launch_h3_attention_large(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype);\n#ifdef __cplusplus\n}\n#endif\n'.replace('#include "ninetoothed.h"','#include "'+str(_HEADER_PATH)+'"'))
(out/"h3_flash_attention_large.c").write_text('#include "h3_flash_attention_large.h"\n#include "h3_large_d128_f16_medium.h"\n#include "h3_large_d128_f16_large.h"\n#include "h3_large_d128_bf16_medium.h"\n#include "h3_large_d128_bf16_large.h"\n#ifdef ENABLE_NINETOOTHED_HYGON_MHA_INNER32\n#include "h3_flash_attention_inner32.h"\n#endif\n#ifdef H3_LARGE_TEST_COUNTER\nstatic thread_local uint64_t medium_calls=0,wide_calls=0;\nextern "C" uint64_t h3_large_calls() { return medium_calls+wide_calls; }\nextern "C" uint64_t h3_medium_profile_calls() { return medium_calls; }\nextern "C" uint64_t h3_wide_profile_calls() { return wide_calls; }\n#endif\nNineToothedResult launch_h3_attention_large(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype) {\n    int wide = query.shape[2] >= 2048 && key.shape[2] >= 2048;\n#ifdef ENABLE_NINETOOTHED_HYGON_MHA_HEAD56\n    // H3 B1 self-attention only; preserve all other profile choices.\n    wide = wide || (query.shape[0] == 1 && key.shape[0] == 1\n        && query.shape[1] == 56 && key.shape[1] == 56\n        && query.shape[2] == key.shape[2]\n        && query.shape[2] >= 1024 && query.shape[2] < 2048);\n#endif\n#ifdef H3_LARGE_TEST_COUNTER\n    if (wide) ++wide_calls; else ++medium_calls;\n#endif\n#ifdef ENABLE_NINETOOTHED_HYGON_MHA_INNER32\n    if (h3_inner32_eligible(query, key, value, total_kv_length, output, dim, dtype)) {\n        /* Masking is disabled in this path; provide valid unused metadata. */\n        uint64_t unused_shape[4] = {1, 1, 1, 1};\n        int64_t unused_strides[4] = {16, 16, 16, 1};\n        NineToothedTensor unused_mask = {0, unused_shape, unused_strides};\n        return dtype == INFINI_DTYPE_F16\n            ? launch_h3_long_inner32_f16_m128n32(stream, query, key, value, total_kv_length, unused_mask, is_causal, scale, output, with_attn_mask, causal_variant)\n            : launch_h3_long_inner32_bf16_m128n32(stream, query, key, value, total_kv_length, unused_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    }\n#endif\n    if (dim==128 && dtype==INFINI_DTYPE_F16) return wide ? launch_h3_large_d128_f16_large(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_large_d128_f16_medium(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==128 && dtype==INFINI_DTYPE_BF16) return wide ? launch_h3_large_d128_bf16_large(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_large_d128_bf16_medium(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    return 1;\n}\n')

(out/"h3_flash_attention_inner32.h").write_text(Path(__file__).with_name("h3_flash_attention_inner32.h").read_text())
