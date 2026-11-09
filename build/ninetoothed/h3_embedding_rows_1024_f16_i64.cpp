#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_embedding_rows_1024_f16_i64_divisibility_1_16_1_1_16_contiguity_0_1_1_0_1_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_28, NineToothedTensor ninetoothed_tensor_29, NineToothedTensor ninetoothed_tensor_30);

extern "C" NineToothedResult launch_h3_embedding_rows_1024_f16_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_28, NineToothedTensor ninetoothed_tensor_29, NineToothedTensor ninetoothed_tensor_30) {
    if (ninetoothed_tensor_28.shape[0] > 2147483647ULL || ninetoothed_tensor_28.strides[0] > 2147483647LL || ninetoothed_tensor_28.strides[0] < -2147483648LL || ninetoothed_tensor_28.shape[1] > 2147483647ULL || ninetoothed_tensor_28.strides[1] > 2147483647LL || ninetoothed_tensor_28.strides[1] < -2147483648LL || ninetoothed_tensor_29.shape[0] > 2147483647ULL || ninetoothed_tensor_29.strides[0] > 2147483647LL || ninetoothed_tensor_29.strides[0] < -2147483648LL || ninetoothed_tensor_30.shape[0] > 2147483647ULL || ninetoothed_tensor_30.strides[0] > 2147483647LL || ninetoothed_tensor_30.strides[0] < -2147483648LL || ninetoothed_tensor_30.shape[1] > 2147483647ULL || ninetoothed_tensor_30.strides[1] > 2147483647LL || ninetoothed_tensor_30.strides[1] < -2147483648LL) return (NineToothedResult)1;
    if (ninetoothed_tensor_28.shape[1] % 16 == 0 && ninetoothed_tensor_30.shape[1] % 16 == 0 && ninetoothed_tensor_28.strides[1] == 1 && ninetoothed_tensor_29.strides[0] == 1 && ninetoothed_tensor_30.strides[1] == 1) return launch_h3_embedding_rows_1024_f16_i64_divisibility_1_16_1_1_16_contiguity_0_1_1_0_1_size_i32_stride_i32(stream, ninetoothed_tensor_28, ninetoothed_tensor_29, ninetoothed_tensor_30);
    return (NineToothedResult)1;
}
