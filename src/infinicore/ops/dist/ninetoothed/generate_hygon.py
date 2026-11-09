"""Dist reduction expressed through NineToothed; no handwritten device code."""
from pathlib import Path
import os, json, time, importlib.util
import ninetoothed as nt
from ninetoothed import Tensor
from triton.runtime import driver
w=Path(os.environ['DIST_FIX_WORK'])
out=w/'generated'

def arrangement(y,a,b,p,mode):
    a=a.tile((512,)).tile((-1,))
    b=b.tile((512,)).tile((-1,))
    y=y.tile((1,))
    return y,a,b,p,mode

def main():
    target=driver.active.get_current_target()
    assert (target.backend,target.arch)==('hip','gfx936')
    source='import ninetoothed.language as nl\n'
    for name,dt in [('f16','float16'),('bf16','bfloat16'),('f32','float32')]:
        source+=f'''def apply_{name}(y,a,b,p,mode):
    power=nl.cast(p,nl.float32)
    lo=float("inf")
    hi=0.0
    count=0.0
    nan_seen=0
    for j in range(a.shape[0]):
        valid=j*512+nl.arange(0,512)<a.source.shape[0]
        delta=nl.cast(nl.cast(nl.cast(a[j],nl.float32)-nl.cast(b[j],nl.float32),nl.{dt}),nl.float32)
        d=nl.abs(delta)
        lo=nl.minimum(lo,nl.min(nl.where(valid,d,float("inf")),0))
        hi=nl.maximum(hi,nl.max(nl.where(valid,d,0.0),0))
        count+=nl.sum(nl.cast(valid&(d!=0.0),nl.float32),0)
        nan_seen=nan_seen|nl.max(nl.cast(valid&(d!=d),nl.int32),0)
    if mode==0:
        result=count
    elif mode==1:
        result=hi
    elif mode==2:
        result=lo
    else:
        scale=nl.where(power>0.0,hi,lo)
        safe=nl.where((scale>0.0)&(scale<float("inf")),scale,1.0)
        total=0.0
        for j in range(a.shape[0]):
            valid=j*512+nl.arange(0,512)<a.source.shape[0]
            delta=nl.cast(nl.cast(nl.cast(a[j],nl.float32)-nl.cast(b[j],nl.float32),nl.{dt}),nl.float32)
            d=nl.abs(delta)
            value=nl.exp(nl.log(d/safe)*power)
            total+=nl.sum(nl.where(valid,value,0.0),0)
        result=safe*nl.exp(nl.log(total)/power)
        result=nl.where(scale==0.0,0.0,nl.where(scale==float("inf"),float("inf"),result))
    if mode!=0:
        result=nl.where(nan_seen!=0,float("nan"),result)
    y=result

'''
    p=w/'scripts/expressions.py';p.write_text(source)
    spec=importlib.util.spec_from_file_location('dist_expressions',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for label,dt in [('f16',nt.float16),('bf16',nt.bfloat16),('f32',nt.float32)]:
        name='h3_dist_'+label;start=time.monotonic();print('GENERATE',name,flush=True)
        nt.make(arrangement,getattr(module,'apply_'+label),(Tensor(1,dtype=dt),Tensor(1,dtype=dt,other=0),Tensor(1,dtype=dt,other=0),Tensor(0,dtype=nt.float32),Tensor(0,dtype=nt.int32)),caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1)
        records.append(dict(name=name,seconds=time.monotonic()-start));(w/'generation-progress.json').write_text(json.dumps(records,indent=2))
    (w/'generation-status.json').write_text(json.dumps(dict(passed=True,records=records),indent=2))
    print('DIST_GENERATION_PASS',flush=True)
if __name__=='__main__':main()
