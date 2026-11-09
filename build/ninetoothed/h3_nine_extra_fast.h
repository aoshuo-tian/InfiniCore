#pragma once
#include "h3_swiglu_fast_dense_f16.h"
#include "h3_swiglu_fast_row_f16.h"
#include "h3_embedding_fast_1024_f16_i32.h"
#include "h3_embedding_fast_4096_f16_i32.h"
#include "h3_embedding_fast_1024_f16_i64.h"
#include "h3_embedding_fast_4096_f16_i64.h"
#include "h3_swiglu_fast_dense_bf16.h"
#include "h3_swiglu_fast_row_bf16.h"
#include "h3_embedding_fast_1024_bf16_i32.h"
#include "h3_embedding_fast_4096_bf16_i32.h"
#include "h3_embedding_fast_1024_bf16_i64.h"
#include "h3_embedding_fast_4096_bf16_i64.h"
#include "h3_swiglu_fast_dense_f32.h"
#include "h3_swiglu_fast_row_f32.h"
#include "h3_embedding_fast_1024_f32_i32.h"
#include "h3_embedding_fast_4096_f32_i32.h"
#include "h3_embedding_fast_1024_f32_i64.h"
#include "h3_embedding_fast_4096_f32_i64.h"
#include "h3_swiglu_small_dense_f16.h"
#include "h3_swiglu_small_row_f16.h"
#include "h3_swiglu_small_dense_bf16.h"
#include "h3_swiglu_small_row_bf16.h"
#include "h3_swiglu_small_dense_f32.h"
#include "h3_swiglu_small_row_f32.h"
#ifdef __cplusplus
extern "C" {
#endif
int launch_h3_embedding_fast_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int, int);
int launch_h3_swiglu_fast_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int);
int launch_h3_swiglu_small_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int);
#ifdef __cplusplus
}
#endif
