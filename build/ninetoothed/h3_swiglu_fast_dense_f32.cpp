#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_swiglu_fast_dense_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_148, NineToothedTensor ninetoothed_tensor_149, NineToothedTensor ninetoothed_tensor_150);

extern "C" NineToothedResult launch_h3_swiglu_fast_dense_f32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_148, NineToothedTensor ninetoothed_tensor_149, NineToothedTensor ninetoothed_tensor_150) {
    if (ninetoothed_tensor_148.shape[0] > 2147483647ULL || ninetoothed_tensor_148.strides[0] > 2147483647LL || ninetoothed_tensor_148.strides[0] < -2147483648LL || ninetoothed_tensor_149.shape[0] > 2147483647ULL || ninetoothed_tensor_149.strides[0] > 2147483647LL || ninetoothed_tensor_149.strides[0] < -2147483648LL || ninetoothed_tensor_150.shape[0] > 2147483647ULL || ninetoothed_tensor_150.strides[0] > 2147483647LL || ninetoothed_tensor_150.strides[0] < -2147483648LL) return launch_h3_swiglu_fast_dense_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_148, ninetoothed_tensor_149, ninetoothed_tensor_150);
    return launch_h3_swiglu_fast_dense_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_148, ninetoothed_tensor_149, ninetoothed_tensor_150);
}
