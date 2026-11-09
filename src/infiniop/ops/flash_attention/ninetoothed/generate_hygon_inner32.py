"""Generate two guarded gfx936 Attention AOT kernels via NineToothed."""
import argparse,os,json,hashlib
from pathlib import Path
import ninetoothed as nt
from triton.runtime import driver
from infiniop.ops.flash_attention.ninetoothed import flash_attention as fa
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[5]/'build/ninetoothed');out=parser.parse_args().output
os.environ['NINETOOTHED_AOT_VARIANTS']='inner32';assert driver.active.get_current_target().arch=='gfx936';out.mkdir(parents=True,exist_ok=True)
records=[]
for dtype,label in [(nt.float16,'f16'),(nt.bfloat16,'bf16')]:
 name=f'h3_long_inner32_{label}_m128n32';arrange,app,tensors=fa.premake(with_kv_cache=0,emb_dim=128,is_causal=0,with_attn_mask=0,causal_variant=fa.CausalVariant.LOWER_RIGHT,dtype=dtype,block_size_m=128,block_size_n=32)
 nt.make(arrange,app,tensors,caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1);records.append(name)
(out/'h3_flash_attention_inner32.h').write_text(Path(__file__).with_name('h3_flash_attention_inner32.h').read_text())
(out/'h3-inner32-generation.json').write_text(json.dumps(dict(records=records,policy='inner32',target='gfx936',expression=str(fa.__file__),expression_sha256=hashlib.sha256(Path(fa.__file__).read_bytes()).hexdigest()),indent=2))
