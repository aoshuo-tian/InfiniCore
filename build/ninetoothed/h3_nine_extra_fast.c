#include "h3_nine_extra_fast.h"
int launch_h3_embedding_fast_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor i, NineToothedTensor w, int dtype, int itype, int block) {
 if(dtype==0 && itype==0 && block==1024)return launch_h3_embedding_fast_1024_f16_i32(s,o,i,w);
 if(dtype==0 && itype==0 && block==4096)return launch_h3_embedding_fast_4096_f16_i32(s,o,i,w);
 if(dtype==0 && itype==1 && block==1024)return launch_h3_embedding_fast_1024_f16_i64(s,o,i,w);
 if(dtype==0 && itype==1 && block==4096)return launch_h3_embedding_fast_4096_f16_i64(s,o,i,w);
 if(dtype==1 && itype==0 && block==1024)return launch_h3_embedding_fast_1024_bf16_i32(s,o,i,w);
 if(dtype==1 && itype==0 && block==4096)return launch_h3_embedding_fast_4096_bf16_i32(s,o,i,w);
 if(dtype==1 && itype==1 && block==1024)return launch_h3_embedding_fast_1024_bf16_i64(s,o,i,w);
 if(dtype==1 && itype==1 && block==4096)return launch_h3_embedding_fast_4096_bf16_i64(s,o,i,w);
 if(dtype==2 && itype==0 && block==1024)return launch_h3_embedding_fast_1024_f32_i32(s,o,i,w);
 if(dtype==2 && itype==0 && block==4096)return launch_h3_embedding_fast_4096_f32_i32(s,o,i,w);
 if(dtype==2 && itype==1 && block==1024)return launch_h3_embedding_fast_1024_f32_i64(s,o,i,w);
 if(dtype==2 && itype==1 && block==4096)return launch_h3_embedding_fast_4096_f32_i64(s,o,i,w);
 return -1;
}
int launch_h3_swiglu_small_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor a, NineToothedTensor b, int dtype, int row) {
 if(dtype==0 && row==0)return launch_h3_swiglu_small_dense_f16(s,o,a,b);
 if(dtype==0 && row==1)return launch_h3_swiglu_small_row_f16(s,o,a,b);
 if(dtype==1 && row==0)return launch_h3_swiglu_small_dense_bf16(s,o,a,b);
 if(dtype==1 && row==1)return launch_h3_swiglu_small_row_bf16(s,o,a,b);
 if(dtype==2 && row==0)return launch_h3_swiglu_small_dense_f32(s,o,a,b);
 if(dtype==2 && row==1)return launch_h3_swiglu_small_row_f32(s,o,a,b);
 return -1;
}
int launch_h3_swiglu_fast_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor a, NineToothedTensor b, int dtype, int row) {
 if(dtype==0 && row==0)return launch_h3_swiglu_fast_dense_f16(s,o,a,b);
 if(dtype==0 && row==1)return launch_h3_swiglu_fast_row_f16(s,o,a,b);
 if(dtype==1 && row==0)return launch_h3_swiglu_fast_dense_bf16(s,o,a,b);
 if(dtype==1 && row==1)return launch_h3_swiglu_fast_row_bf16(s,o,a,b);
 if(dtype==2 && row==0)return launch_h3_swiglu_fast_dense_f32(s,o,a,b);
 if(dtype==2 && row==1)return launch_h3_swiglu_fast_row_f32(s,o,a,b);
 return -1;
}
