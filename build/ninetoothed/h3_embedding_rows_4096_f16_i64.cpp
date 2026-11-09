#include "../../../../dependencies/abi/ninetoothed.h"

extern "C" NineToothedResult launch_h3_embedding_rows_4096_f16_i64_divisibility_1_16_1_1_16_contiguity_0_1_1_0_1_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_42, NineToothedTensor ninetoothed_tensor_43, NineToothedTensor ninetoothed_tensor_44);

extern "C" NineToothedResult launch_h3_embedding_rows_4096_f16_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_42, NineToothedTensor ninetoothed_tensor_43, NineToothedTensor ninetoothed_tensor_44) {
    if (ninetoothed_tensor_42.shape[0] > 2147483647ULL || ninetoothed_tensor_42.strides[0] > 2147483647LL || ninetoothed_tensor_42.strides[0] < -2147483648LL || ninetoothed_tensor_42.shape[1] > 2147483647ULL || ninetoothed_tensor_42.strides[1] > 2147483647LL || ninetoothed_tensor_42.strides[1] < -2147483648LL || ninetoothed_tensor_43.shape[0] > 2147483647ULL || ninetoothed_tensor_43.strides[0] > 2147483647LL || ninetoothed_tensor_43.strides[0] < -2147483648LL || ninetoothed_tensor_44.shape[0] > 2147483647ULL || ninetoothed_tensor_44.strides[0] > 2147483647LL || ninetoothed_tensor_44.strides[0] < -2147483648LL || ninetoothed_tensor_44.shape[1] > 2147483647ULL || ninetoothed_tensor_44.strides[1] > 2147483647LL || ninetoothed_tensor_44.strides[1] < -2147483648LL) return (NineToothedResult)1;
    if (ninetoothed_tensor_42.shape[1] % 16 == 0 && ninetoothed_tensor_44.shape[1] % 16 == 0 && ninetoothed_tensor_42.strides[1] == 1 && ninetoothed_tensor_43.strides[0] == 1 && ninetoothed_tensor_44.strides[1] == 1) return launch_h3_embedding_rows_4096_f16_i64_divisibility_1_16_1_1_16_contiguity_0_1_1_0_1_size_i32_stride_i32(stream, ninetoothed_tensor_42, ninetoothed_tensor_43, ninetoothed_tensor_44);
    return (NineToothedResult)1;
}
