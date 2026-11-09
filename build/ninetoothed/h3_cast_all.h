#include "h3_cast_f16_f16.h"
#include "h3_cast_f16_bf16.h"
#include "h3_cast_f16_f32.h"
#include "h3_cast_bf16_f16.h"
#include "h3_cast_bf16_bf16.h"
#include "h3_cast_bf16_f32.h"
#include "h3_cast_f32_f16.h"
#include "h3_cast_f32_bf16.h"
#include "h3_cast_f32_f32.h"
#ifdef __cplusplus
extern "C" {
#endif
int launch_h3_cast_all(NineToothedStream, NineToothedTensor, NineToothedTensor, int, int);
#ifdef __cplusplus
}
#endif
