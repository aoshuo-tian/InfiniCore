#include "h3_nine_inner_ops.h"
int launch_h3_embedding_inner_all(NineToothedStream s,NineToothedTensor o,NineToothedTensor i,NineToothedTensor w,int dtype,int itype,int block){
 if(dtype==0 && itype==0 && block==1024)return launch_h3_embedding_inner16_1024_f16_i32(s,o,i,w);
 if(dtype==0 && itype==0 && block==4096)return launch_h3_embedding_inner16_4096_f16_i32(s,o,i,w);
 if(dtype==0 && itype==1 && block==1024)return launch_h3_embedding_inner16_1024_f16_i64(s,o,i,w);
 if(dtype==0 && itype==1 && block==4096)return launch_h3_embedding_inner16_4096_f16_i64(s,o,i,w);
 if(dtype==1 && itype==0 && block==1024)return launch_h3_embedding_inner16_1024_bf16_i32(s,o,i,w);
 if(dtype==1 && itype==0 && block==4096)return launch_h3_embedding_inner16_4096_bf16_i32(s,o,i,w);
 if(dtype==1 && itype==1 && block==1024)return launch_h3_embedding_inner16_1024_bf16_i64(s,o,i,w);
 if(dtype==1 && itype==1 && block==4096)return launch_h3_embedding_inner16_4096_bf16_i64(s,o,i,w);
 if(dtype==2 && itype==0 && block==1024)return launch_h3_embedding_inner16_1024_f32_i32(s,o,i,w);
 if(dtype==2 && itype==0 && block==4096)return launch_h3_embedding_inner16_4096_f32_i32(s,o,i,w);
 if(dtype==2 && itype==1 && block==1024)return launch_h3_embedding_inner16_1024_f32_i64(s,o,i,w);
 if(dtype==2 && itype==1 && block==4096)return launch_h3_embedding_inner16_4096_f32_i64(s,o,i,w);
 return 1;
}
int launch_h3_swiglu_inner_all(NineToothedStream s,NineToothedTensor o,NineToothedTensor u,NineToothedTensor g,int dtype,int row){
 if(dtype==0 && row==0)return launch_h3_swiglu_inner_256_f16(s,o,u,g);
 if(dtype==0 && row==1)return launch_h3_swiglu_inner16_row_256_f16(s,o,u,g);
 if(dtype==1 && row==0)return launch_h3_swiglu_inner_256_bf16(s,o,u,g);
 if(dtype==1 && row==1)return launch_h3_swiglu_inner16_row_256_bf16(s,o,u,g);
 if(dtype==2 && row==0)return launch_h3_swiglu_inner_256_f32(s,o,u,g);
 if(dtype==2 && row==1)return launch_h3_swiglu_inner16_row_256_f32(s,o,u,g);
 return 1;
}
