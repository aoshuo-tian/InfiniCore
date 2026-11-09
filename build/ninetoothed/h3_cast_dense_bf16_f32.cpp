#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_cast_dense_bf16_f32_divisibility_1_1_contiguity_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_78, NineToothedTensor ninetoothed_tensor_79);

extern "C" NineToothedResult launch_h3_cast_dense_bf16_f32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_78, NineToothedTensor ninetoothed_tensor_79) {
    if (ninetoothed_tensor_78.shape[0] > 2147483647ULL || ninetoothed_tensor_78.strides[0] > 2147483647LL || ninetoothed_tensor_78.strides[0] < -2147483648LL || ninetoothed_tensor_79.shape[0] > 2147483647ULL || ninetoothed_tensor_79.strides[0] > 2147483647LL || ninetoothed_tensor_79.strides[0] < -2147483648LL) return launch_h3_cast_dense_bf16_f32_divisibility_1_1_contiguity_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_78, ninetoothed_tensor_79);
    return launch_h3_cast_dense_bf16_f32_divisibility_1_1_contiguity_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_78, ninetoothed_tensor_79);
}
