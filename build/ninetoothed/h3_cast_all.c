#include "h3_cast_all.h"
int launch_h3_cast_all(NineToothedStream stream, NineToothedTensor input, NineToothedTensor output, int src, int dst) {
    if (src == 1 && dst == 1) return launch_h3_cast_f16_f16(stream, input, output);
    if (src == 1 && dst == 2) return launch_h3_cast_f16_bf16(stream, input, output);
    if (src == 1 && dst == 3) return launch_h3_cast_f16_f32(stream, input, output);
    if (src == 2 && dst == 1) return launch_h3_cast_bf16_f16(stream, input, output);
    if (src == 2 && dst == 2) return launch_h3_cast_bf16_bf16(stream, input, output);
    if (src == 2 && dst == 3) return launch_h3_cast_bf16_f32(stream, input, output);
    if (src == 3 && dst == 1) return launch_h3_cast_f32_f16(stream, input, output);
    if (src == 3 && dst == 2) return launch_h3_cast_f32_bf16(stream, input, output);
    if (src == 3 && dst == 3) return launch_h3_cast_f32_f32(stream, input, output);
    return -1;
}
