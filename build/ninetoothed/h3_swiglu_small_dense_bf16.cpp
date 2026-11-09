#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_swiglu_small_dense_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_60, NineToothedTensor ninetoothed_tensor_61, NineToothedTensor ninetoothed_tensor_62);

extern "C" NineToothedResult launch_h3_swiglu_small_dense_bf16(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_60, NineToothedTensor ninetoothed_tensor_61, NineToothedTensor ninetoothed_tensor_62) {
    if (ninetoothed_tensor_60.shape[0] > 2147483647ULL || ninetoothed_tensor_60.strides[0] > 2147483647LL || ninetoothed_tensor_60.strides[0] < -2147483648LL || ninetoothed_tensor_61.shape[0] > 2147483647ULL || ninetoothed_tensor_61.strides[0] > 2147483647LL || ninetoothed_tensor_61.strides[0] < -2147483648LL || ninetoothed_tensor_62.shape[0] > 2147483647ULL || ninetoothed_tensor_62.strides[0] > 2147483647LL || ninetoothed_tensor_62.strides[0] < -2147483648LL) return launch_h3_swiglu_small_dense_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_60, ninetoothed_tensor_61, ninetoothed_tensor_62);
    return launch_h3_swiglu_small_dense_bf16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_60, ninetoothed_tensor_61, ninetoothed_tensor_62);
}
