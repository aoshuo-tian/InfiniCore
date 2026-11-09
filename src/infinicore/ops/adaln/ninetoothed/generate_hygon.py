"""Production entry: two NineToothed kernels, preserving two dtype round trips."""
import json
import os
from pathlib import Path
import time
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from triton.runtime import driver

def arrange(o,x,s,b):return tuple(t.tile((1,1024)) for t in (o,x,s,b))
def application_half(o,x,s,b):
    x32=ntl.cast(x,ntl.float32)
    p=ntl.cast(x32*ntl.cast(s,ntl.float32),ntl.float16)
    a=ntl.cast(x32+ntl.cast(p,ntl.float32),ntl.float16)
    o=ntl.cast(a,ntl.float32)+ntl.cast(b,ntl.float32)
def application_bfloat(o,x,s,b):
    x32=ntl.cast(x,ntl.float32)
    p=ntl.cast(x32*ntl.cast(s,ntl.float32),ntl.bfloat16)
    a=ntl.cast(x32+ntl.cast(p,ntl.float32),ntl.bfloat16)
    o=ntl.cast(a,ntl.float32)+ntl.cast(b,ntl.float32)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('build/ninetoothed-adaln'));args=parser.parse_args()
    target=driver.active.get_current_target()
    if (target.backend,target.arch)!=('hip','gfx936'):raise RuntimeError(f'Unsupported AdaLN target: {target}')
    os.environ['NINETOOTHED_AOT_VARIANTS']='inner16'
    os.environ['TRITON_DEFAULT_FP_FUSION']='0'
    args.output.mkdir(parents=True,exist_ok=True);records=[]
    for label,dt,app in [('f16',nt.float16,application_half),('bf16',nt.bfloat16,application_bfloat)]:
        name=f'h3_adaln_{label}_1024';start=time.monotonic();print('GENERATE',name,flush=True)
        nt.make(arrange,app,tuple(Tensor(2,dtype=dt) for _ in range(4)),caller='cuda',kernel_name=name,
                output_dir=args.output,num_warps=4,num_stages=1)
        records.append(dict(name=name,seconds=time.monotonic()-start))
    (args.output/'generation.json').write_text(json.dumps(dict(target=str(target),fp_fusion=False,records=records),indent=2))
    print('ADALN_PRODUCTION_GENERATION_PASS',flush=True)
if __name__=='__main__':main()
