#include "infinicore/ops/gated_residual.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/mul.hpp"
#include <array>
#include <climits>
#include <cstdint>
#include <stdexcept>
#if defined(ENABLE_NINETOOTHED_HYGON_GATED_RESIDUAL) && defined(ENABLE_HYGON_API)
#include "h3_gate_f16_1024.h"
#include "h3_gate_bf16_1024.h"
#endif
namespace infinicore::op {
namespace {
enum Path {Fused, Disabled, DeviceFallback, TypeFallback, LayoutFallback, GraphFallback, PathCount};
#ifdef INFINICORE_GATED_RESIDUAL_TESTING
thread_local uint64_t counts[PathCount] = {};
void count(Path p) {++counts[p];}
#else
void count(Path) {}
#endif
Tensor original(const Tensor &x,const Tensor &g,const Tensor &v,Path path) {
    count(path);return add(x,mul(g,v));
}
#if defined(ENABLE_NINETOOTHED_HYGON_GATED_RESIDUAL) && defined(ENABLE_HYGON_API)
struct Meta {uint64_t shape[2];int64_t strides[2];};
bool rows(const Tensor &t,Meta &m) {
    const auto n=t->ndim();if(n!=2 && n!=3)return false;
    const auto h=t->size(n-1),s=t->size(n-2),b=n==3?t->size(0):1;
    if(!h||!s||!b||h%16||h>INT_MAX||s>INT_MAX||b>INT_MAX/s)return false;
    const auto &st=t->strides();if(st[n-1]!=1)return false;
    int64_t row=st[n-2];
    if(n==3 && b>1) {
        if(s==1)row=st[0];
        else if(st[1]<0||static_cast<uint64_t>(st[1])>INT_MAX/s||st[0]!=static_cast<int64_t>(s)*st[1])return false;
    }
    const uint64_t r=b*s;
    if(row<static_cast<int64_t>(h)||row>INT_MAX||row%16||r-1>(INT_MAX-h)/static_cast<uint64_t>(row)
       ||reinterpret_cast<uintptr_t>(t->data())%16)return false;
    m={{r,h},{row,1}};return true;
}
#endif
}
Tensor gated_residual(const Tensor &x,const Tensor &g,const Tensor &v,bool prefer_fusion) {
    if(!prefer_fusion)return original(x,g,v,Disabled);
#if defined(ENABLE_NINETOOTHED_HYGON_GATED_RESIDUAL) && defined(ENABLE_HYGON_API)
    if(context::isGraphRecording())return original(x,g,v,GraphFallback);
    if(x->device().getType()!=Device::Type::HYGON||g->device()!=x->device()||v->device()!=x->device())return original(x,g,v,DeviceFallback);
    const auto dt=x->dtype();
    // The old F32 Add uses downward rounding. Its original path is retained.
    if((dt!=DataType::F16&&dt!=DataType::BF16)||g->dtype()!=dt||v->dtype()!=dt)return original(x,g,v,TypeFallback);
    std::array<Meta,4> m;
    if(x->shape()!=g->shape()||x->shape()!=v->shape()||!rows(x,m[1])||!rows(g,m[2])||!rows(v,m[3]))return original(x,g,v,LayoutFallback);
    context::setDevice(x->device());
    auto out=Tensor::empty(x->shape(),dt,x->device());
    if(!rows(out,m[0]))return original(x,g,v,LayoutFallback);
    NineToothedTensor o{out->data(),m[0].shape,m[0].strides};
    NineToothedTensor a{const_cast<std::byte*>(x->data()),m[1].shape,m[1].strides};
    NineToothedTensor gate{const_cast<std::byte*>(g->data()),m[2].shape,m[2].strides};
    NineToothedTensor branch{const_cast<std::byte*>(v->data()),m[3].shape,m[3].strides};
    const auto rc=dt==DataType::F16?launch_h3_gate_f16_1024(context::getStream(),o,a,gate,branch)
                                  :launch_h3_gate_bf16_1024(context::getStream(),o,a,gate,branch);
    if(rc)throw std::runtime_error("[gated-residual/ninetoothed-hygon] launch failed: "+std::to_string(rc));
    count(Fused);return out;
#else
    return original(x,g,v,Disabled);
#endif
}
#ifdef INFINICORE_GATED_RESIDUAL_TESTING
extern "C" uint64_t h3_gated_residual_dispatch_count(unsigned path,int reset) {
    const auto value=path<PathCount?counts[path]:0;
    if(reset)for(auto &v:counts)v=0;
    return value;
}
#endif
}
