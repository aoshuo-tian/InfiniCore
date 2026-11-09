#pragma once
#include "h3_embedding_rows_1024_f16_i32.h"
#include "h3_embedding_rows_4096_f16_i32.h"
#include "h3_embedding_rows_1024_f16_i64.h"
#include "h3_embedding_rows_4096_f16_i64.h"
#include "h3_embedding_rows_1024_bf16_i32.h"
#include "h3_embedding_rows_4096_bf16_i32.h"
#include "h3_embedding_rows_1024_bf16_i64.h"
#include "h3_embedding_rows_4096_bf16_i64.h"
#include "h3_embedding_rows_1024_f32_i32.h"
#include "h3_embedding_rows_4096_f32_i32.h"
#include "h3_embedding_rows_1024_f32_i64.h"
#include "h3_embedding_rows_4096_f32_i64.h"

#ifdef __cplusplus
extern "C" {
#endif
int launch_h3_embedding_rows_all(NineToothedStream,NineToothedTensor,NineToothedTensor,NineToothedTensor,int,int,int);
#ifdef __cplusplus
}
#endif
