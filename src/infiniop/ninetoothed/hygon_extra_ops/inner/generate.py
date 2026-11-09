"""Generate a strict inner-contiguous int32 AOT candidate through NineToothed."""
import hashlib
import json
import os
from pathlib import Path
import time

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from triton.runtime import driver


def embedding1024(output, indices, weight):
    output=output.tile((1,1024))
    indices=indices.tile((1,)).unsqueeze(1).expand(output.shape)
    weight=weight.flatten().tile((1,)).unsqueeze(1).expand(output.shape)
    return output,indices,weight


def embedding4096(output, indices, weight):
    output=output.tile((1,4096))
    indices=indices.tile((1,)).unsqueeze(1).expand(output.shape)
    weight=weight.flatten().tile((1,)).unsqueeze(1).expand(output.shape)
    return output,indices,weight


def embedding_application(output,indices,weight):
    row=ntl.expand_dims(output.offsets(0),1)
    feature=ntl.expand_dims(output.offsets(1),0)
    valid_row=row<indices.source.shape[0]
    index=ntl.load(indices.source.data_ptr()+row*indices.source.stride(0),mask=valid_row,other=0)
    valid=valid_row & (index>=0) & (index<weight.source.shape[0]) & (feature<weight.source.shape[1])
    index=ntl.where(valid,index,0)
    output=ntl.load(weight.source.data_ptr()+index*weight.source.stride(0)+feature*weight.source.stride(1),
                    mask=valid,other=0)


def swiglu256(output,up,gate):
    return tuple(t.tile((256,)) for t in (output,up,gate))


def swiglu1024(output,up,gate):
    return tuple(t.tile((1024,)) for t in (output,up,gate))


def swiglu4096(output,up,gate):
    return tuple(t.tile((4096,)) for t in (output,up,gate))


def swiglu_row256(output,up,gate):
    return tuple(t.tile((1,256)) for t in (output,up,gate))


def swiglu_application(output,up,gate):
    u=ntl.cast(up,ntl.float32)
    g=ntl.cast(gate,ntl.float32)
    output=u*g*ntl.sigmoid(g)


DTYPES={'f16':nt.float16,'bf16':nt.bfloat16,'f32':nt.float32}
ITYPES={'i32':nt.int32,'i64':nt.int64}


def jobs():
    for dtype in DTYPES:
        yield f'h3_swiglu_inner_256_{dtype}',(swiglu256,swiglu_application,tuple(Tensor(1,dtype=DTYPES[dtype]) for _ in range(3)))
        yield f'h3_swiglu_inner_row_256_{dtype}',(swiglu_row256,swiglu_application,tuple(Tensor(2,dtype=DTYPES[dtype]) for _ in range(3)))
        for it in ITYPES:
            for block,arrange in ((1024,embedding1024),(4096,embedding4096)):
                yield f'h3_embedding_inner_{block}_{dtype}_{it}',(arrange,embedding_application,(
                    Tensor(2,dtype=DTYPES[dtype]),Tensor(1,dtype=ITYPES[it]),Tensor(2,dtype=DTYPES[dtype])))


def main():
    w=Path(os.environ['H3_WORK']);out=Path(os.environ['NINE_INNER_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    target=driver.active.get_current_target();assert (target.backend,target.arch)==('hip','gfx936'),target
    assert Path(nt.__file__).resolve().is_relative_to(w/'sources/ninetoothed')
    requested_policy=os.environ.get('NINE_INNER_POLICY','auto')
    selected=os.environ.get('NINE_INNER_SELECT','').split(',');records=[]
    for name,(arrange,application,tensors) in jobs():
        policy=requested_policy
        if policy=='auto':policy='inner16' if 'embedding' in name or '_row_' in name else 'inner32'
        if policy=='inner16':name=name.replace('_inner_','_inner16_')
        if selected!=[''] and name not in selected:continue
        os.environ['NINETOOTHED_AOT_VARIANTS']=policy
        print('GENERATE',name,flush=True);start=time.monotonic()
        nt.make(arrange,application,tensors,caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1)
        records.append(dict(name=name,policy=policy,seconds=time.monotonic()-start))
        (out/'progress.json').write_text(json.dumps(records,indent=2))
    if selected==['']:
        includes='\n'.join(f'#include "{rec["name"]}.h"' for rec in records)
        header='#pragma once\n'+includes+'\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
        header+='int launch_h3_embedding_inner_all(NineToothedStream,NineToothedTensor,NineToothedTensor,NineToothedTensor,int,int,int);\n'
        header+='int launch_h3_swiglu_inner_all(NineToothedStream,NineToothedTensor,NineToothedTensor,NineToothedTensor,int,int);\n'
        header+='#ifdef __cplusplus\n}\n#endif\n'
        (out/'h3_nine_inner_ops.h').write_text(header)
        source='#include "h3_nine_inner_ops.h"\n'
        source+='int launch_h3_embedding_inner_all(NineToothedStream s,NineToothedTensor o,NineToothedTensor i,NineToothedTensor w,int dtype,int itype,int block){\n'
        for dc,dtype in enumerate(DTYPES):
            for ic,it in enumerate(ITYPES):
                for block in (1024,4096):
                    source+=f' if(dtype=={dc} && itype=={ic} && block=={block})return launch_h3_embedding_inner16_{block}_{dtype}_{it}(s,o,i,w);\n'
        source+=' return 1;\n}\n'
        source+='int launch_h3_swiglu_inner_all(NineToothedStream s,NineToothedTensor o,NineToothedTensor u,NineToothedTensor g,int dtype,int row){\n'
        for dc,dtype in enumerate(DTYPES):
            source+=f' if(dtype=={dc} && row==0)return launch_h3_swiglu_inner_256_{dtype}(s,o,u,g);\n'
            source+=f' if(dtype=={dc} && row==1)return launch_h3_swiglu_inner16_row_256_{dtype}(s,o,u,g);\n'
        source+=' return 1;\n}\n'
        (out/'h3_nine_inner_ops.c').write_text(source)
    (out/'manifest.json').write_text(json.dumps(dict(target=str(target),policy=requested_policy,records=records,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2))
    print('INNER_GENERATION_PASS',len(records),flush=True)


if __name__=='__main__':main()
