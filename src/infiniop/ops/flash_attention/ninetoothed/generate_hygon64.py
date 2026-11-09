import json,os,sys,time,hashlib
from pathlib import Path
import ninetoothed as nt
from triton.runtime import driver
from infiniop.ops.flash_attention.ninetoothed import flash_attention as fa

import argparse
from ninetoothed.aot import _HEADER_PATH
parser=argparse.ArgumentParser(description="Generate Hygon gfx936 noncausal M64N64 NineToothed attention")
parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[5]/"build/ninetoothed")
output=parser.parse_args().output
output.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("NINETOOTHED_AOT_VARIANTS","generic")
assert driver.active.get_current_target().arch=='gfx936'
records=[]
for dim in [128,72,64]:
    for dtype,label in [(nt.bfloat16,'bf16'),(nt.float16,'f16')]:
        name=f'h3_attention_d{dim}_{label}_m64'
        begin=time.monotonic();print('TILE_GENERATE',name,flush=True)
        arrangement,application,tensors=fa.premake(with_kv_cache=0,emb_dim=dim,is_causal=0,with_attn_mask=0,
            causal_variant=fa.CausalVariant.LOWER_RIGHT,dtype=dtype,block_size_m=64,block_size_n=64)
        nt.make(arrangement,application,tensors,caller='cuda',kernel_name=name,output_dir=output)
        records.append({'name':name,'dim':dim,'dtype':label,'seconds':time.monotonic()-begin})
        print('TILE_GENERATED',name,records[-1]['seconds'],flush=True)
result={'target':str(driver.active.get_current_target()),'records':records,
        'source':str(Path(fa.__file__).resolve()),'source_sha256':hashlib.sha256(Path(fa.__file__).read_bytes()).hexdigest(),
        'change':'existing NineToothed expression; tile M=64 N=64, generic AOT','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*') if p.is_file()}}
(output/'h3_attention64-generation.json').write_text(json.dumps(result,indent=2))
print('TILE_GENERATION_PASS',len(records),flush=True)

(output/"h3_flash_attention64.h").write_text('#pragma once\n#include "infinicore.h"\n#include "ninetoothed.h"\n#ifdef __cplusplus\nextern "C" {\n#endif\nNineToothedResult launch_h3_attention_tile64(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype);\n#ifdef __cplusplus\n}\n#endif\n'.replace('#include "ninetoothed.h"','#include "'+str(_HEADER_PATH)+'"'))
(output/"h3_flash_attention64.c").write_text('#include "h3_flash_attention64.h"\n#include "h3_attention_d64_f16_m64.h"\n#include "h3_attention_d64_bf16_m64.h"\n#include "h3_attention_d72_f16_m64.h"\n#include "h3_attention_d72_bf16_m64.h"\n#include "h3_attention_d128_f16_m64.h"\n#include "h3_attention_d128_bf16_m64.h"\nNineToothedResult launch_h3_attention_tile64(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_length, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor scale, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int dim, int dtype) {\n    if (dim==64 && dtype==INFINI_DTYPE_F16) return launch_h3_attention_d64_f16_m64(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==64 && dtype==INFINI_DTYPE_BF16) return launch_h3_attention_d64_bf16_m64(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==72 && dtype==INFINI_DTYPE_F16) return launch_h3_attention_d72_f16_m64(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==72 && dtype==INFINI_DTYPE_BF16) return launch_h3_attention_d72_bf16_m64(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==128 && dtype==INFINI_DTYPE_F16) return launch_h3_attention_d128_f16_m64(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    if (dim==128 && dtype==INFINI_DTYPE_BF16) return launch_h3_attention_d128_bf16_m64(stream, query, key, value, total_kv_length, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant);\n    return 1;\n}\n')
