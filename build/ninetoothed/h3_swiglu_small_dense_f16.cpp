#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_swiglu_small_dense_f16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_54, NineToothedTensor ninetoothed_tensor_55, NineToothedTensor ninetoothed_tensor_56);

extern "C" NineToothedResult launch_h3_swiglu_small_dense_f16(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_54, NineToothedTensor ninetoothed_tensor_55, NineToothedTensor ninetoothed_tensor_56) {
    if (ninetoothed_tensor_54.shape[0] > 2147483647ULL || ninetoothed_tensor_54.strides[0] > 2147483647LL || ninetoothed_tensor_54.strides[0] < -2147483648LL || ninetoothed_tensor_55.shape[0] > 2147483647ULL || ninetoothed_tensor_55.strides[0] > 2147483647LL || ninetoothed_tensor_55.strides[0] < -2147483648LL || ninetoothed_tensor_56.shape[0] > 2147483647ULL || ninetoothed_tensor_56.strides[0] > 2147483647LL || ninetoothed_tensor_56.strides[0] < -2147483648LL) return launch_h3_swiglu_small_dense_f16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_54, ninetoothed_tensor_55, ninetoothed_tensor_56);
    return launch_h3_swiglu_small_dense_f16_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_54, ninetoothed_tensor_55, ninetoothed_tensor_56);
}
