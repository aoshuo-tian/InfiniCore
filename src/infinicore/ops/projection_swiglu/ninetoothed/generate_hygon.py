"""Gate/up projection + SwiGLU in one NineToothed program.

Weights are [D,K] views of the existing [up;gate] packed [2D,K].
Both projections round to the original storage dtype before activation.
"""
import functools,json,os,time
from pathlib import Path
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(x,wu,wg,o,bm=16,bn=32,bk=32):
    ot=o.tile((bm,bn))
    xt=x.tile((bm,bk)).tile((1,-1)).expand((-1,ot.shape[1]))
    xt.dtype=xt.dtype.squeeze(0)
    ut=wu.permute((1,0)).tile((bk,bn)).tile((-1,1)).expand((ot.shape[0],-1))
    ut.dtype=ut.dtype.squeeze(1)
    gt=wg.permute((1,0)).tile((bk,bn)).tile((-1,1)).expand((ot.shape[0],-1))
    gt.dtype=gt.dtype.squeeze(1)
    return xt,ut,gt,ot

def application_half(x,wu,wg,o):
    u=ntl.zeros(o.shape,dtype=ntl.float32)
    g=ntl.zeros(o.shape,dtype=ntl.float32)
    for k in range(x.shape[0]):
        u+=ntl.dot(x[k],wu[k])
        g+=ntl.dot(x[k],wg[k])
    ur=ntl.cast(ntl.cast(u,ntl.float16),ntl.float32)
    gr=ntl.cast(ntl.cast(g,ntl.float16),ntl.float32)
    o=ntl.sigmoid(gr)*gr*ur

def application_bfloat(x,wu,wg,o):
    u=ntl.zeros(o.shape,dtype=ntl.float32)
    g=ntl.zeros(o.shape,dtype=ntl.float32)
    for k in range(x.shape[0]):
        u+=ntl.dot(x[k],wu[k])
        g+=ntl.dot(x[k],wg[k])
    ur=ntl.cast(ntl.cast(u,ntl.bfloat16),ntl.float32)
    gr=ntl.cast(ntl.cast(g,ntl.bfloat16),ntl.float32)
    o=ntl.sigmoid(gr)*gr*ur

def main():
    import argparse
    from triton.runtime import driver
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--wide-only',action='store_true');a=p.parse_args()
    target=driver.active.get_current_target();assert (target.backend,target.arch)==('hip','gfx936'),target
    os.environ['TRITON_DEFAULT_FP_FUSION']='0'
    os.environ['NINETOOTHED_AOT_VARIANTS']='inner16'
    a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for label,dt,app in [('f16',nt.float16,application_half),('bf16',nt.bfloat16,application_bfloat)]:
        configs=[(128,64,64,8,2)] if a.wide_only else [(16,32,32,4,1),(32,64,32,4,1),(128,64,64,8,2)]
        for bm,bn,bk,warps,stages in configs:
            name=f'h3_proj_swiglu_{label}_m{bm}n{bn}k{bk}'
            print('GENERATE',name,flush=True);t=time.monotonic()
            nt.make(functools.partial(arrangement,bm=bm,bn=bn,bk=bk),app,tuple(Tensor(2,dtype=dt) for _ in range(4)),caller='cuda',kernel_name=name,output_dir=a.output,num_warps=warps,num_stages=stages)
            rows.append(dict(name=name,seconds=time.monotonic()-t))
    (a.output/('generation-wide.json' if a.wide_only else 'generation.json')).write_text(json.dumps(dict(target=str(target),records=rows,fp_fusion=False),indent=2))

if __name__=='__main__':main()
