#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_swiglu_inner_256_f16_divisibility_1_1_1_contiguity_1_1_1_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_0, NineToothedTensor ninetoothed_tensor_1, NineToothedTensor ninetoothed_tensor_2);

extern "C" NineToothedResult launch_h3_swiglu_inner_256_f16(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_0, NineToothedTensor ninetoothed_tensor_1, NineToothedTensor ninetoothed_tensor_2) {
    if (ninetoothed_tensor_0.shape[0] > 2147483647ULL || ninetoothed_tensor_0.strides[0] > 2147483647LL || ninetoothed_tensor_0.strides[0] < -2147483648LL || ninetoothed_tensor_1.shape[0] > 2147483647ULL || ninetoothed_tensor_1.strides[0] > 2147483647LL || ninetoothed_tensor_1.strides[0] < -2147483648LL || ninetoothed_tensor_2.shape[0] > 2147483647ULL || ninetoothed_tensor_2.strides[0] > 2147483647LL || ninetoothed_tensor_2.strides[0] < -2147483648LL) return (NineToothedResult)1;
    if (ninetoothed_tensor_0.strides[0] == 1 && ninetoothed_tensor_1.strides[0] == 1 && ninetoothed_tensor_2.strides[0] == 1) return launch_h3_swiglu_inner_256_f16_divisibility_1_1_1_contiguity_1_1_1_size_i32_stride_i32(stream, ninetoothed_tensor_0, ninetoothed_tensor_1, ninetoothed_tensor_2);
    return (NineToothedResult)1;
}
