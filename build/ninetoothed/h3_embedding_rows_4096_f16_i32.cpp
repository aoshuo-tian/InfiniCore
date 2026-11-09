#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_embedding_rows_4096_f16_i32_divisibility_1_16_1_1_16_contiguity_0_1_1_0_1_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_14, NineToothedTensor ninetoothed_tensor_15, NineToothedTensor ninetoothed_tensor_16);

extern "C" NineToothedResult launch_h3_embedding_rows_4096_f16_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_14, NineToothedTensor ninetoothed_tensor_15, NineToothedTensor ninetoothed_tensor_16) {
    if (ninetoothed_tensor_14.shape[0] > 2147483647ULL || ninetoothed_tensor_14.strides[0] > 2147483647LL || ninetoothed_tensor_14.strides[0] < -2147483648LL || ninetoothed_tensor_14.shape[1] > 2147483647ULL || ninetoothed_tensor_14.strides[1] > 2147483647LL || ninetoothed_tensor_14.strides[1] < -2147483648LL || ninetoothed_tensor_15.shape[0] > 2147483647ULL || ninetoothed_tensor_15.strides[0] > 2147483647LL || ninetoothed_tensor_15.strides[0] < -2147483648LL || ninetoothed_tensor_16.shape[0] > 2147483647ULL || ninetoothed_tensor_16.strides[0] > 2147483647LL || ninetoothed_tensor_16.strides[0] < -2147483648LL || ninetoothed_tensor_16.shape[1] > 2147483647ULL || ninetoothed_tensor_16.strides[1] > 2147483647LL || ninetoothed_tensor_16.strides[1] < -2147483648LL) return (NineToothedResult)1;
    if (ninetoothed_tensor_14.shape[1] % 16 == 0 && ninetoothed_tensor_16.shape[1] % 16 == 0 && ninetoothed_tensor_14.strides[1] == 1 && ninetoothed_tensor_15.strides[0] == 1 && ninetoothed_tensor_16.strides[1] == 1) return launch_h3_embedding_rows_4096_f16_i32_divisibility_1_16_1_1_16_contiguity_0_1_1_0_1_size_i32_stride_i32(stream, ninetoothed_tensor_14, ninetoothed_tensor_15, ninetoothed_tensor_16);
    return (NineToothedResult)1;
}
