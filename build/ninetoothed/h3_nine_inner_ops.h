#pragma once
#include "h3_swiglu_inner_256_f16.h"
#include "h3_swiglu_inner16_row_256_f16.h"
#include "h3_embedding_inner16_1024_f16_i32.h"
#include "h3_embedding_inner16_4096_f16_i32.h"
#include "h3_embedding_inner16_1024_f16_i64.h"
#include "h3_embedding_inner16_4096_f16_i64.h"
#include "h3_swiglu_inner_256_bf16.h"
#include "h3_swiglu_inner16_row_256_bf16.h"
#include "h3_embedding_inner16_1024_bf16_i32.h"
#include "h3_embedding_inner16_4096_bf16_i32.h"
#include "h3_embedding_inner16_1024_bf16_i64.h"
#include "h3_embedding_inner16_4096_bf16_i64.h"
#include "h3_swiglu_inner_256_f32.h"
#include "h3_swiglu_inner16_row_256_f32.h"
#include "h3_embedding_inner16_1024_f32_i32.h"
#include "h3_embedding_inner16_4096_f32_i32.h"
#include "h3_embedding_inner16_1024_f32_i64.h"
#include "h3_embedding_inner16_4096_f32_i64.h"
#ifdef __cplusplus
extern "C" {
#endif
int launch_h3_embedding_inner_all(NineToothedStream,NineToothedTensor,NineToothedTensor,NineToothedTensor,int,int,int);
int launch_h3_swiglu_inner_all(NineToothedStream,NineToothedTensor,NineToothedTensor,NineToothedTensor,int,int);
#ifdef __cplusplus
}
#endif
