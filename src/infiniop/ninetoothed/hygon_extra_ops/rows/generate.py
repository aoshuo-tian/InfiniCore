"""NineToothed row-gather: preserve feature alignment, permit arbitrary row count."""
from pathlib import Path
import os,json,time,hashlib
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from triton.runtime import driver
def rows1024(output,indices,weight):
 output=output.tile((1,1024));indices=indices.tile((1,)).unsqueeze(1).expand(output.shape);weight=weight.flatten().tile((1,)).unsqueeze(1).expand(output.shape)
 return output,indices,weight
def rows4096(output,indices,weight):
 output=output.tile((1,4096));indices=indices.tile((1,)).unsqueeze(1).expand(output.shape);weight=weight.flatten().tile((1,)).unsqueeze(1).expand(output.shape)
 return output,indices,weight
def application(output,indices,weight):
 row=ntl.expand_dims(output.offsets(0),1);feature=ntl.expand_dims(output.offsets(1),0)
 valid_row=row<indices.source.shape[0]
 index=ntl.load(indices.source.data_ptr()+row*indices.source.stride(0),mask=valid_row,other=0)
 valid=valid_row&(index>=0)&(index<weight.source.shape[0])&(feature<weight.source.shape[1])
 index=ntl.where(valid,index,0)
 output=ntl.load(weight.source.data_ptr()+index*weight.source.stride(0)+feature*weight.source.stride(1),mask=valid,other=0)
def main():
 w=Path(os.environ['EMBEDDING_ROWS_WORK']);out=w/'generated';target=driver.active.get_current_target();assert (target.backend,target.arch)==('hip','gfx936'),target
 assert Path(nt.__file__).resolve().is_relative_to(w/'compiler/ninetoothed');assert os.environ['NINETOOTHED_AOT_VARIANTS']=='embedding_rows_i32'
 records=[]
 for dc,dt in [('f16',nt.float16),('bf16',nt.bfloat16),('f32',nt.float32)]:
  for ic,it in [('i32',nt.int32),('i64',nt.int64)]:
   for block,arrange in [(1024,rows1024),(4096,rows4096)]:
    name=f'h3_embedding_rows_{block}_{dc}_{ic}';start=time.monotonic();print('GENERATE',name,flush=True)
    nt.make(arrange,application,(Tensor(2,dtype=dt),Tensor(1,dtype=it),Tensor(2,dtype=dt)),caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1)
    records.append({'name':name,'seconds':time.monotonic()-start});(w/'generation-progress.json').write_text(json.dumps(records,indent=2))
 header='#pragma once\n'+''.join('#include "'+r['name']+'.h"\n' for r in records)+'\n#ifdef __cplusplus\nextern "C" {\n#endif\nint launch_h3_embedding_rows_all(NineToothedStream,NineToothedTensor,NineToothedTensor,NineToothedTensor,int,int,int);\n#ifdef __cplusplus\n}\n#endif\n'
 (out/'h3_embedding_rows_ops.h').write_text(header);source='#include "h3_embedding_rows_ops.h"\nint launch_h3_embedding_rows_all(NineToothedStream s,NineToothedTensor o,NineToothedTensor i,NineToothedTensor w,int dtype,int itype,int block){\n'
 for dc,dt in enumerate(['f16','bf16','f32']):
  for ic,it in enumerate(['i32','i64']):
   for block in [1024,4096]:source+=f' if(dtype=={dc} && itype=={ic} && block=={block})return launch_h3_embedding_rows_{block}_{dt}_{it}(s,o,i,w);\n'
 (out/'h3_embedding_rows_ops.c').write_text(source+' return 1;\n}\n')
 (w/'generation-status.json').write_text(json.dumps({'complete':True,'passed':True,'target':str(target),'records':records,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}},indent=2));print('EMBEDDING_ARBITRARY_ROWS_GENERATION_PASS',flush=True)
if __name__=='__main__':main()
