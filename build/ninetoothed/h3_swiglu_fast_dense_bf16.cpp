#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_swiglu_fast_dense_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_74, NineToothedTensor ninetoothed_tensor_75, NineToothedTensor ninetoothed_tensor_76);

extern "C" NineToothedResult launch_h3_swiglu_fast_dense_bf16(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_74, NineToothedTensor ninetoothed_tensor_75, NineToothedTensor ninetoothed_tensor_76) {
    if (ninetoothed_tensor_74.shape[0] > 2147483647ULL || ninetoothed_tensor_74.strides[0] > 2147483647LL || ninetoothed_tensor_74.strides[0] < -2147483648LL || ninetoothed_tensor_75.shape[0] > 2147483647ULL || ninetoothed_tensor_75.strides[0] > 2147483647LL || ninetoothed_tensor_75.strides[0] < -2147483648LL || ninetoothed_tensor_76.shape[0] > 2147483647ULL || ninetoothed_tensor_76.strides[0] > 2147483647LL || ninetoothed_tensor_76.strides[0] < -2147483648LL) return launch_h3_swiglu_fast_dense_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_74, ninetoothed_tensor_75, ninetoothed_tensor_76);
    return launch_h3_swiglu_fast_dense_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_74, ninetoothed_tensor_75, ninetoothed_tensor_76);
}
