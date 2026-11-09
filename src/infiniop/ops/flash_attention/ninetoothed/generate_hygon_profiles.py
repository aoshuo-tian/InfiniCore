import os,json,time,hashlib
from pathlib import Path
import ninetoothed as nt
from triton.runtime import driver
from infiniop.ops.flash_attention.ninetoothed import flash_attention as fa

import argparse
from ninetoothed.aot import _HEADER_PATH
parser=argparse.ArgumentParser(description="Generate two NineToothed Hygon noncausal MHA profiles")
parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[5]/"build/ninetoothed")
out=parser.parse_args().output
os.environ.setdefault("NINETOOTHED_AOT_VARIANTS","generic")
out.mkdir(parents=True,exist_ok=True)
assert driver.active.get_current_target().arch=='gfx936'
records=[]
for dim in [64,72,128]:
    for dtype,label in [(nt.float16,'f16'),(nt.bfloat16,'bf16')]:
        for profile,m,n in [('short',32,32),('long',64,64)]:
            name=f'h3_profile_d{dim}_{label}_{profile}'
            begin=time.monotonic();print('PROFILE_GENERATE',name,flush=True)
            arrange,app,tensors=fa.premake(with_kv_cache=0,emb_dim=dim,is_causal=0,with_attn_mask=0,
                causal_variant=fa.CausalVariant.LOWER_RIGHT,dtype=dtype,block_size_m=m,block_size_n=n)
            nt.make(arrange,app,tensors,caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1)
            records.append({'name':name,'dim':dim,'dtype':label,'profile':profile,'m':m,'n':n,'warps':4,'stages':1,'seconds':time.monotonic()-begin})
            (out/'h3-profile-generation.json').write_text(json.dumps({'target':str(driver.active.get_current_target()),'records':records,
                'expression_path':str(Path(fa.__file__).resolve()),'expression_sha256':hashlib.sha256(Path(fa.__file__).read_bytes()).hexdigest()},indent=2))
            print('PROFILE_GENERATED',name,flush=True)
print('PROFILE_GENERATION_PASS',len(records),flush=True)

(out/"h3_flash_attention_profile.h").write_text('#pragma once\n#include "infinicore.h"\n#include "ninetoothed.h"\n#ifdef __cplusplus\nextern "C" {\n#endif\nNineToothedResult launch_h3_attention_profile(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype);\n#ifdef __cplusplus\n}\n#endif\n'.replace('#include "ninetoothed.h"','#include "'+str(_HEADER_PATH)+'"'))
(out/"h3_flash_attention_profile.c").write_text('#include "h3_flash_attention_profile.h"\n#include "h3_profile_d64_f16_short.h"\n#include "h3_profile_d64_f16_long.h"\n#include "h3_profile_d64_bf16_short.h"\n#include "h3_profile_d64_bf16_long.h"\n#include "h3_profile_d72_f16_short.h"\n#include "h3_profile_d72_f16_long.h"\n#include "h3_profile_d72_bf16_short.h"\n#include "h3_profile_d72_bf16_long.h"\n#include "h3_profile_d128_f16_short.h"\n#include "h3_profile_d128_f16_long.h"\n#include "h3_profile_d128_bf16_short.h"\n#include "h3_profile_d128_bf16_long.h"\nNineToothedResult launch_h3_attention_profile(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype) {\n    const int use_long = key.shape[2] > 512;\n    if (dim==64 && dtype==INFINI_DTYPE_F16) return use_long ? launch_h3_profile_d64_f16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d64_f16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==64 && dtype==INFINI_DTYPE_BF16) return use_long ? launch_h3_profile_d64_bf16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d64_bf16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==72 && dtype==INFINI_DTYPE_F16) return use_long ? launch_h3_profile_d72_f16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d72_f16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==72 && dtype==INFINI_DTYPE_BF16) return use_long ? launch_h3_profile_d72_bf16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d72_bf16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==128 && dtype==INFINI_DTYPE_F16) return use_long ? launch_h3_profile_d128_f16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d128_f16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==128 && dtype==INFINI_DTYPE_BF16) return use_long ? launch_h3_profile_d128_bf16_long(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant) : launch_h3_profile_d128_bf16_short(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    return 1;\n}\n')
