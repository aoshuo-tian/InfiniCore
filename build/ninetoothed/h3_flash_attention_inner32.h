#pragma once
#include <stdint.h>
#include <limits.h>
#include "h3_long_inner32_f16_m128n32.h"
#include "h3_long_inner32_bf16_m128n32.h"

/* The strict AOT entry points assume aligned pointers, unit feature strides,
 * and outer strides divisible by 16. Keep every other request on the old path. */
static int h3_inner32_span(NineToothedTensor t, uint64_t sequence, uint64_t *bytes) {
    const uint64_t expected[4] = {1, 56, sequence, 128};
    uint64_t span = 0;
    if (!t.data || !t.shape || !t.strides || ((uintptr_t)t.data % 16)) return 0;
    for (int i = 0; i < 4; ++i) {
        if (t.shape[i] != expected[i] || t.shape[i] > INT32_MAX
            || t.strides[i] < 0 || t.strides[i] > INT32_MAX) return 0;
        if (i == 3 ? t.strides[i] != 1 : t.strides[i] % 16 != 0) return 0;
        const uint64_t stride = (uint64_t)t.strides[i];
        if (stride && t.shape[i] - 1 > ((uint64_t)INT32_MAX / 2 - span) / stride) return 0;
        span += (t.shape[i] - 1) * stride;
    }
    *bytes = (span + 1) * 2;
    return *bytes <= INT32_MAX && (uintptr_t)t.data <= UINTPTR_MAX - *bytes;
}

static int h3_inner32_eligible(NineToothedTensor q, NineToothedTensor k,
                             NineToothedTensor v, NineToothedTensor lengths,
                             NineToothedTensor out, int dim, int dtype) {
    if (dim != 128 || (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16)
        || !q.shape || q.shape[2] < 2048 || q.shape[2] > INT32_MAX) return 0;
    if (!lengths.data || !lengths.shape || !lengths.strides
        || (uintptr_t)lengths.data % 16 || lengths.shape[0] != 1 || lengths.strides[0] != 1) return 0;
    NineToothedTensor tensors[4] = {q, k, v, out};
    uint64_t bytes[4];
    for (int i = 0; i < 4; ++i)
        if (!h3_inner32_span(tensors[i], q.shape[2], bytes + i)) return 0;
    const uintptr_t ob = (uintptr_t)out.data, oe = ob + bytes[3];
    for (int i = 0; i < 3; ++i) {
        const uintptr_t ib = (uintptr_t)tensors[i].data;
        if (ob < ib + bytes[i] && ib < oe) return 0;
    }
    const uintptr_t lb = (uintptr_t)lengths.data;
    if (lb > UINTPTR_MAX - 4 || (ob < lb + 4 && lb < oe)) return 0;
    return 1;
}
