#include "infinicore/ops/adaln.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/mul.hpp"

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(ENABLE_NINETOOTHED_HYGON_ADALN) && defined(ENABLE_HYGON_API)
#include "h3_adaln_f16_1024.h"
#include "h3_adaln_bf16_1024.h"
#endif

namespace infinicore::op {
namespace {

enum Path { Fused = 0, Disabled, DeviceFallback, TypeFallback,
            LayoutFallback, GraphFallback, PathCount };
#ifdef INFINICORE_ADALN_TESTING
thread_local uint64_t counts[PathCount] = {};
void count(Path p) { ++counts[p]; }
#else
void count(Path) {}
#endif

Tensor original(const Tensor &x, const Tensor &scale, const Tensor &shift, Path path) {
    count(path);
    auto product = mul(x, scale);
    auto sum = add(x, product);
    return add(sum, shift);
}

#if defined(ENABLE_NINETOOTHED_HYGON_ADALN) && defined(ENABLE_HYGON_API)
struct RowMeta {
    uint64_t shape[2];
    int64_t strides[2];
};

bool rows(const Tensor &t, RowMeta &meta) {
    const auto n = t->ndim();
    if (n != 2 && n != 3) { return false; }
    const auto h = t->size(n - 1), s = t->size(n - 2);
    const auto b = n == 3 ? t->size(0) : 1;
    if (h == 0 || s == 0 || b == 0 || h % 16 != 0
        || h > INT_MAX || s > INT_MAX || b > INT_MAX / s) { return false; }
    const auto &st = t->strides();
    if (st[n - 1] != 1) { return false; }
    int64_t row = st[n - 2];
    if (n == 3 && b > 1) {
        if (s == 1) { row = st[0]; }
        else if (st[1] < 0 || static_cast<uint64_t>(st[1]) > INT_MAX / s
                 || st[0] != static_cast<int64_t>(s) * st[1]) { return false; }
    }
    const uint64_t r = b * s;
    if (row < static_cast<int64_t>(h) || row > INT_MAX || row % 16 != 0
        || r - 1 > (INT_MAX - h) / static_cast<uint64_t>(row)
        || reinterpret_cast<uintptr_t>(t->data()) % 16 != 0) { return false; }
    meta = {{r, h}, {row, 1}};
    return true;
}
#endif
} // namespace

Tensor adaln(const Tensor &x, const Tensor &scale, const Tensor &shift, bool prefer_fusion) {
    if (!prefer_fusion) { return original(x, scale, shift, Disabled); }
#if defined(ENABLE_NINETOOTHED_HYGON_ADALN) && defined(ENABLE_HYGON_API)
    if (context::isGraphRecording()) { return original(x, scale, shift, GraphFallback); }
    if (x->device().getType() != Device::Type::HYGON
        || x->device() != scale->device() || x->device() != shift->device()) {
        return original(x, scale, shift, DeviceFallback);
    }
    const auto dtype = x->dtype();
    // Original F32 AddOp uses __fadd_rd. Do not replace it with RNE arithmetic.
    if ((dtype != DataType::F16 && dtype != DataType::BF16)
        || scale->dtype() != dtype || shift->dtype() != dtype) {
        return original(x, scale, shift, TypeFallback);
    }
    std::array<RowMeta, 4> m;
    if (x->shape() != scale->shape() || x->shape() != shift->shape()
        || !rows(x, m[1]) || !rows(scale, m[2]) || !rows(shift, m[3])) {
        return original(x, scale, shift, LayoutFallback);
    }
    context::setDevice(x->device());
    // A fresh, contiguous output cannot alias a live input. No output values or
    // intermediate activations are cached between calls.
    auto out = Tensor::empty(x->shape(), dtype, x->device());
    if (!rows(out, m[0])) { return original(x, scale, shift, LayoutFallback); }
    NineToothedTensor o{out->data(), m[0].shape, m[0].strides};
    // NineToothed's ABI uses void* for both load-only and store tensors.
    NineToothedTensor a{const_cast<std::byte *>(x->data()), m[1].shape, m[1].strides};
    NineToothedTensor s{const_cast<std::byte *>(scale->data()), m[2].shape, m[2].strides};
    NineToothedTensor b{const_cast<std::byte *>(shift->data()), m[3].shape, m[3].strides};
    const auto rc = dtype == DataType::F16
        ? launch_h3_adaln_f16_1024(context::getStream(), o, a, s, b)
        : launch_h3_adaln_bf16_1024(context::getStream(), o, a, s, b);
    if (rc != 0) {
        // A failed device launch must propagate; retrying could hide an async
        // fault or duplicate partially submitted work.
        throw std::runtime_error("[adaln/ninetoothed-hygon] launch failed: " + std::to_string(rc));
    }
    count(Fused);
    return out;
#else
    return original(x, scale, shift, Disabled);
#endif
}

#ifdef INFINICORE_ADALN_TESTING
extern "C" uint64_t h3_adaln_dispatch_count(unsigned path, int reset) {
    const auto value = path < PathCount ? counts[path] : 0;
    if (reset) { for (auto &v : counts) { v = 0; } }
    return value;
}
#endif

} // namespace infinicore::op
