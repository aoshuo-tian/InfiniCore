#include "h3_nine_extra_ops.h"
int launch_h3_swiglu_fp32_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor a, NineToothedTensor b, int dtype) {
 if (dtype == 0) return launch_h3_swiglu_fp32_f16(s,o,a,b);
 if (dtype == 1) return launch_h3_swiglu_fp32_bf16(s,o,a,b);
 if (dtype == 2) return launch_h3_swiglu_fp32_f32(s,o,a,b);
 return -1;
}
int launch_h3_embedding_strided_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor i, NineToothedTensor w, int dtype, int itype) {
 if (dtype == 0 && itype == 0) return launch_h3_embedding_strided_f16_i32(s,o,i,w);
 if (dtype == 0 && itype == 1) return launch_h3_embedding_strided_f16_i64(s,o,i,w);
 if (dtype == 1 && itype == 0) return launch_h3_embedding_strided_bf16_i32(s,o,i,w);
 if (dtype == 1 && itype == 1) return launch_h3_embedding_strided_bf16_i64(s,o,i,w);
 if (dtype == 2 && itype == 0) return launch_h3_embedding_strided_f32_i32(s,o,i,w);
 if (dtype == 2 && itype == 1) return launch_h3_embedding_strided_f32_i64(s,o,i,w);
 return -1;
}
