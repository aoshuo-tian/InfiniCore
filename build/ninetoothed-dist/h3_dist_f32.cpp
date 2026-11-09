#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_dist_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_30, NineToothedTensor ninetoothed_tensor_31, NineToothedTensor ninetoothed_tensor_32, NineToothedTensor ninetoothed_tensor_33, NineToothedTensor ninetoothed_tensor_34);

extern "C" NineToothedResult launch_h3_dist_f32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_30, NineToothedTensor ninetoothed_tensor_31, NineToothedTensor ninetoothed_tensor_32, NineToothedTensor ninetoothed_tensor_33, NineToothedTensor ninetoothed_tensor_34) {
    if (ninetoothed_tensor_30.shape[0] > 2147483647ULL || ninetoothed_tensor_30.strides[0] > 2147483647LL || ninetoothed_tensor_30.strides[0] < -2147483648LL || ninetoothed_tensor_31.shape[0] > 2147483647ULL || ninetoothed_tensor_31.strides[0] > 2147483647LL || ninetoothed_tensor_31.strides[0] < -2147483648LL || ninetoothed_tensor_32.shape[0] > 2147483647ULL || ninetoothed_tensor_32.strides[0] > 2147483647LL || ninetoothed_tensor_32.strides[0] < -2147483648LL) return launch_h3_dist_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_30, ninetoothed_tensor_31, ninetoothed_tensor_32, ninetoothed_tensor_33, ninetoothed_tensor_34);
    return launch_h3_dist_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_30, ninetoothed_tensor_31, ninetoothed_tensor_32, ninetoothed_tensor_33, ninetoothed_tensor_34);
}
