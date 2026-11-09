"""Production NineToothed entry for two qualified gated-residual kernels."""
import os,json
from pathlib import Path
os.environ['NINETOOTHED_AOT_VARIANTS']='inner16'
os.environ['TRITON_DEFAULT_FP_FUSION']='0'
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from triton.runtime import driver
def arrange(o,x,g,v):return tuple(t.tile((1,1024)) for t in (o,x,g,v))
def application_half(o,x,g,v):
    p=ntl.cast(ntl.cast(g,ntl.float32)*ntl.cast(v,ntl.float32),ntl.float16)
    o=ntl.cast(x,ntl.float32)+ntl.cast(p,ntl.float32)
def application_bfloat(o,x,g,v):
    p=ntl.cast(ntl.cast(g,ntl.float32)*ntl.cast(v,ntl.float32),ntl.bfloat16)
    o=ntl.cast(x,ntl.float32)+ntl.cast(p,ntl.float32)
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('build/ninetoothed-gated-residual'));args=p.parse_args()
    target=driver.active.get_current_target();assert (target.backend,target.arch)==('hip','gfx936'),target
    args.output.mkdir(parents=True,exist_ok=True)
    for label,dt,app in [('f16',nt.float16,application_half),('bf16',nt.bfloat16,application_bfloat)]:
        name=f'h3_gate_{label}_1024';print('GENERATE',name,flush=True)
        nt.make(arrange,app,tuple(Tensor(2,dtype=dt) for _ in range(4)),caller='cuda',kernel_name=name,output_dir=args.output,num_warps=4,num_stages=1)
    (args.output/'generation.json').write_text(json.dumps(dict(target=str(target),fp_fusion=False,kernels=2),indent=2))
    print('PRODUCTION_GATED_GENERATION_PASS',flush=True)
if __name__=='__main__':main()
