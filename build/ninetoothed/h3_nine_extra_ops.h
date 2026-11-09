#pragma once
#include "h3_swiglu_fp32_f16.h"
#include "h3_embedding_strided_f16_i32.h"
#include "h3_embedding_strided_f16_i64.h"
#include "h3_swiglu_fp32_bf16.h"
#include "h3_embedding_strided_bf16_i32.h"
#include "h3_embedding_strided_bf16_i64.h"
#include "h3_swiglu_fp32_f32.h"
#include "h3_embedding_strided_f32_i32.h"
#include "h3_embedding_strided_f32_i64.h"
#ifdef __cplusplus
extern "C" {
#endif
int launch_h3_swiglu_fp32_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int);
int launch_h3_embedding_strided_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int);
#ifdef __cplusplus
}
#endif
