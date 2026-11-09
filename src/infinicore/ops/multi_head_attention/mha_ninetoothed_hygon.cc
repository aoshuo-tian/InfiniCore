#if defined(ENABLE_HYGON_API) && defined(ENABLE_NINETOOTHED_FLASH_ATTENTION)

#include "../infiniop_impl.hpp"
#include "infinicore/ops/mha.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <deque>

namespace infinicore::op::mha_impl::ninetoothed_hygon {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FlashAttention, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, q, k, v;
    // Own the lengths allocation for the lifetime of this execution plan.
    Tensor lengths;
};

namespace {
struct LengthEntry {
    Device device;
    size_t batch, kv_length;
    Tensor tensor;
};
struct LengthCache {
    std::deque<LengthEntry> entries;
    size_t bytes = 0, hits = 0, allocations = 0;
};
thread_local LengthCache length_cache;
constexpr size_t kLengthCacheEntries = 32;
constexpr size_t kLengthCacheBytes = 256 * 1024;
constexpr size_t kLengthEntryBytes = 64 * 1024;

Tensor get_lengths(const Device &device, size_t batch, size_t kv_length) {
    for (const auto &entry : length_cache.entries) {
        if (entry.device == device && entry.batch == batch && entry.kv_length == kv_length) {
            ++length_cache.hits;
            return entry.tensor;
        }
    }
    auto lengths = Tensor::empty({batch}, DataType::I32, device);
    std::vector<int32_t> host_lengths(batch, static_cast<int32_t>(kv_length));
    // Synchronous initialization also makes immutable reuse safe on another stream.
    context::memcpyH2D(lengths->data(), host_lengths.data(), batch * sizeof(int32_t), false);
    ++length_cache.allocations;
    const size_t bytes = batch * sizeof(int32_t);
    if (bytes <= kLengthEntryBytes) {
        while (!length_cache.entries.empty()
               && (length_cache.entries.size() >= kLengthCacheEntries
                   || length_cache.bytes + bytes > kLengthCacheBytes)) {
            length_cache.bytes -= length_cache.entries.front().batch * sizeof(int32_t);
            length_cache.entries.pop_front();
        }
        length_cache.entries.push_back({device, batch, kv_length, lengths});
        length_cache.bytes += bytes;
    }
    return lengths;
}
} // namespace

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi_slopes, float scale, bool is_causal) {
    if (context::isGraphRecording()) {
        throw std::runtime_error(
            "[mha/ninetoothed-hygon] graph recording is not supported yet");
    }
    if (alibi_slopes.has_value() || is_causal) {
        throw std::runtime_error(
            "[mha/ninetoothed-hygon] only non-causal attention without ALiBi is supported");
    }
    if (!std::isfinite(scale)) {
        throw std::runtime_error("[mha/ninetoothed-hygon] scale must be finite");
    }

    const auto &qs = q->shape();
    const auto &ks = k->shape();
    const auto &vs = v->shape();
    if (qs.size() != 4 || ks.size() != 4 || vs.size() != 4) {
        throw std::runtime_error("[mha/ninetoothed-hygon] expected [B,S,H,D]");
    }
    if (q->device().getType() != Device::Type::HYGON
        || q->device() != k->device() || q->device() != v->device()
        || q->device() != out->device()) {
        throw std::runtime_error("[mha/ninetoothed-hygon] device mismatch");
    }
    const auto dtype = q->dtype();
    if ((dtype != DataType::F16 && dtype != DataType::BF16)
        || k->dtype() != dtype || v->dtype() != dtype || out->dtype() != dtype
        || (qs[3] != 64 && qs[3] != 72 && qs[3] != 128)
        || ks[3] != qs[3] || vs[3] != qs[3]) {
        throw std::runtime_error(
            "[mha/ninetoothed-hygon] this build supports matching FP16/BF16 tensors with D=64, D=72 or D=128");
    }
    if (qs[0] == 0 || qs[1] == 0 || qs[2] == 0 || ks[1] == 0
        || qs[0] != ks[0] || ks != vs || qs[2] != ks[2]
        || out->shape() != qs) {
        throw std::runtime_error(
            "[mha/ninetoothed-hygon] incompatible shapes or head counts");
    }
    if (ks[1] > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("[mha/ninetoothed-hygon] KV length exceeds int32");
    }

    context::setDevice(q->device());

    // Metadata-only views: BSHD -> BHSD, including the output.
    auto q_view = q->permute({0, 2, 1, 3});
    auto k_view = k->permute({0, 2, 1, 3});
    auto v_view = v->permute({0, 2, 1, 3});
    auto out_view = out->permute({0, 2, 1, 3});

    // Reuse immutable metadata only. Model inputs and generated values are never cached.
    auto lengths = get_lengths(q->device(), qs[0], ks[1]);

    size_t seed = hash_combine(
        out_view, q_view, k_view, v_view, lengths, scale, is_causal);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, FlashAttention, seed,
        out_view->desc(), q_view->desc(), k_view->desc(), v_view->desc(),
        lengths->desc(), scale, is_causal);
    INFINIOP_WORKSPACE_TENSOR(workspace, FlashAttention, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out_view),
        graph::GraphTensor(q_view),
        graph::GraphTensor(k_view),
        graph::GraphTensor(v_view),
        lengths};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    context::setDevice(p->q->device());
    INFINICORE_CHECK_ERROR(infiniopFlashAttention(
        p->descriptor->desc,
        p->workspace->data(), p->workspace->numel(),
        p->out->data(), p->q->data(), p->k->data(), p->v->data(),
        p->lengths->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::ninetoothed_hygon

#endif
