#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_dist_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_15, NineToothedTensor ninetoothed_tensor_16, NineToothedTensor ninetoothed_tensor_17, NineToothedTensor ninetoothed_tensor_18, NineToothedTensor ninetoothed_tensor_19);

extern "C" NineToothedResult launch_h3_dist_bf16(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_15, NineToothedTensor ninetoothed_tensor_16, NineToothedTensor ninetoothed_tensor_17, NineToothedTensor ninetoothed_tensor_18, NineToothedTensor ninetoothed_tensor_19) {
    if (ninetoothed_tensor_15.shape[0] > 2147483647ULL || ninetoothed_tensor_15.strides[0] > 2147483647LL || ninetoothed_tensor_15.strides[0] < -2147483648LL || ninetoothed_tensor_16.shape[0] > 2147483647ULL || ninetoothed_tensor_16.strides[0] > 2147483647LL || ninetoothed_tensor_16.strides[0] < -2147483648LL || ninetoothed_tensor_17.shape[0] > 2147483647ULL || ninetoothed_tensor_17.strides[0] > 2147483647LL || ninetoothed_tensor_17.strides[0] < -2147483648LL) return launch_h3_dist_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_15, ninetoothed_tensor_16, ninetoothed_tensor_17, ninetoothed_tensor_18, ninetoothed_tensor_19);
    return launch_h3_dist_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_15, ninetoothed_tensor_16, ninetoothed_tensor_17, ninetoothed_tensor_18, ninetoothed_tensor_19);
}
