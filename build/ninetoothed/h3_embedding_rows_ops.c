#include "h3_embedding_rows_ops.h"
int launch_h3_embedding_rows_all(NineToothedStream s,NineToothedTensor o,NineToothedTensor i,NineToothedTensor w,int dtype,int itype,int block){
 if(dtype==0 && itype==0 && block==1024)return launch_h3_embedding_rows_1024_f16_i32(s,o,i,w);
 if(dtype==0 && itype==0 && block==4096)return launch_h3_embedding_rows_4096_f16_i32(s,o,i,w);
 if(dtype==0 && itype==1 && block==1024)return launch_h3_embedding_rows_1024_f16_i64(s,o,i,w);
 if(dtype==0 && itype==1 && block==4096)return launch_h3_embedding_rows_4096_f16_i64(s,o,i,w);
 if(dtype==1 && itype==0 && block==1024)return launch_h3_embedding_rows_1024_bf16_i32(s,o,i,w);
 if(dtype==1 && itype==0 && block==4096)return launch_h3_embedding_rows_4096_bf16_i32(s,o,i,w);
 if(dtype==1 && itype==1 && block==1024)return launch_h3_embedding_rows_1024_bf16_i64(s,o,i,w);
 if(dtype==1 && itype==1 && block==4096)return launch_h3_embedding_rows_4096_bf16_i64(s,o,i,w);
 if(dtype==2 && itype==0 && block==1024)return launch_h3_embedding_rows_1024_f32_i32(s,o,i,w);
 if(dtype==2 && itype==0 && block==4096)return launch_h3_embedding_rows_4096_f32_i32(s,o,i,w);
 if(dtype==2 && itype==1 && block==1024)return launch_h3_embedding_rows_1024_f32_i64(s,o,i,w);
 if(dtype==2 && itype==1 && block==4096)return launch_h3_embedding_rows_4096_f32_i64(s,o,i,w);
 return 1;
}
