#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_swiglu_small_dense_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_66, NineToothedTensor ninetoothed_tensor_67, NineToothedTensor ninetoothed_tensor_68);

extern "C" NineToothedResult launch_h3_swiglu_small_dense_f32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_66, NineToothedTensor ninetoothed_tensor_67, NineToothedTensor ninetoothed_tensor_68) {
    if (ninetoothed_tensor_66.shape[0] > 2147483647ULL || ninetoothed_tensor_66.strides[0] > 2147483647LL || ninetoothed_tensor_66.strides[0] < -2147483648LL || ninetoothed_tensor_67.shape[0] > 2147483647ULL || ninetoothed_tensor_67.strides[0] > 2147483647LL || ninetoothed_tensor_67.strides[0] < -2147483648LL || ninetoothed_tensor_68.shape[0] > 2147483647ULL || ninetoothed_tensor_68.strides[0] > 2147483647LL || ninetoothed_tensor_68.strides[0] < -2147483648LL) return launch_h3_swiglu_small_dense_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_66, ninetoothed_tensor_67, ninetoothed_tensor_68);
    return launch_h3_swiglu_small_dense_f32_divisibility_1_1_1_contiguity_0_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_66, ninetoothed_tensor_67, ninetoothed_tensor_68);
}
