#pragma once
#include "../tensor.h"
#include "h3_nine_extra_ops.h"
#include "h3_nine_extra_fast.h"
#include "h3_nine_inner_ops.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace op::hygon_nine_layout {
inline int dtype_code(infiniDtype_t dtype) {
    if (dtype == INFINI_DTYPE_F16) return 0;
    if (dtype == INFINI_DTYPE_BF16) return 1;
    if (dtype == INFINI_DTYPE_F32) return 2;
    return -1;
}
inline uint64_t itemsize(infiniDtype_t dtype) {
    if (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16) return 2;
    if (dtype == INFINI_DTYPE_F32 || dtype == INFINI_DTYPE_I32) return 4;
    return dtype == INFINI_DTYPE_I64 ? 8 : 0;
}
struct Range { uintptr_t first, last; };
template <size_t N> struct Layout {
    std::array<uint64_t,N> shape{};
    std::array<int64_t,N> strides{};
    uint64_t count=1, span=0, bytes=0;
    infiniDtype_t dtype{};
    NineToothedTensor tensor(const void *ptr) const {
        return {const_cast<void *>(ptr), const_cast<uint64_t *>(shape.data()), const_cast<int64_t *>(strides.data())};
    }
    bool contiguous() const {
        uint64_t expected=1;
        for(size_t i=N;i-->0;) {
            if(shape[i]>1 && uint64_t(strides[i])!=expected)return false;
            expected*=shape[i];
        }
        return true;
    }
    bool range(const void *ptr, Range &r) const {
        const uintptr_t start=reinterpret_cast<uintptr_t>(ptr);
        const auto available=std::numeric_limits<uintptr_t>::max()-start;
        if (!ptr || !bytes || available<bytes-1 || span>(available-(bytes-1))/bytes) return false;
        r={start,start+span*bytes+bytes-1};
        return true;
    }
    bool nonoverlapping() const {
        if (!count) return true;
        std::vector<std::pair<uint64_t,uint64_t>> dims;
        for(size_t i=0;i<N;++i) if(shape[i]>1) dims.emplace_back(strides[i],shape[i]);
        std::sort(dims.begin(),dims.end());
        uint64_t covered=1;
        for(const auto &d:dims) {
            if(d.first<covered) return false;
            covered+=(d.second-1)*d.first;
        }
        return true;
    }
};
template <size_t N> infiniStatus_t make(infiniopTensorDescriptor_t desc, Layout<N> &out) {
    if(!desc) return INFINI_STATUS_NULL_POINTER;
    if(desc->ndim()>N) return INFINI_STATUS_BAD_TENSOR_SHAPE;
    out.shape.fill(1); out.strides.fill(0); out.dtype=desc->dtype(); out.bytes=itemsize(out.dtype);
    if(!out.bytes) return INFINI_STATUS_BAD_TENSOR_DTYPE;
    for(size_t i=0;i<desc->ndim();++i) {
        auto n=desc->dim(i); auto s=desc->stride(i);
        if(s<0 || s>std::numeric_limits<int32_t>::max()) return INFINI_STATUS_BAD_TENSOR_STRIDES;
        if(n>uint64_t(std::numeric_limits<int32_t>::max())) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        if(n && out.count>uint64_t(std::numeric_limits<int32_t>::max())/n) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        out.count*=n; out.span+=(n ? n-1 : 0)*uint64_t(s);
        out.shape[N-desc->ndim()+i]=n; out.strides[N-desc->ndim()+i]=s;
    }
    // Bound metadata to the validated int32 range shared by the AOT variants.
    if(out.span>uint64_t(std::numeric_limits<int32_t>::max())) return INFINI_STATUS_BAD_TENSOR_STRIDES;
    return INFINI_STATUS_SUCCESS;
}
inline bool separate(Range a,Range b) { return a.last<b.first || b.last<a.first; }
template <size_t N> bool exact_alias(const Layout<N> &a,const void *ap,const Layout<N> &b,const void *bp) {
    return ap==bp && a.dtype==b.dtype && a.shape==b.shape && a.strides==b.strides;
}
template <size_t N> bool disjoint_or_packed(const Layout<N> &a,const void *ap,const Layout<N> &b,const void *bp) {
    Range ar{},br{};
    if(!a.range(ap,ar) || !b.range(bp,br)) return false;
    if(separate(ar,br)) return true;
    if(a.dtype!=b.dtype || a.shape!=b.shape || a.strides!=b.strides) return false;
    // Prove disjoint packed row slices even when their bounding ranges overlap.
    uint64_t width=1,step=0;
    for(size_t i=N;i-->0;) {
        if(a.shape[i]<=1) continue;
        if(!step && uint64_t(a.strides[i])==width) width*=a.shape[i];
        else {
            if(!step) step=a.strides[i];
            if(!step || uint64_t(a.strides[i])%step) return false;
        }
    }
    if(!step || width>step) return false;
    const auto distance=ar.first>br.first ? ar.first-br.first : br.first-ar.first;
    const auto delta=distance%(step*a.bytes);
    return delta>=width*a.bytes && delta+width*a.bytes<=step*a.bytes;
}
} // namespace op::hygon_nine_layout
