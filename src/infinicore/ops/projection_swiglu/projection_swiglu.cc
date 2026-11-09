#include "infinicore/ops/projection_swiglu.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/swiglu.hpp"
#include <climits>
#include <cstdint>
#include <stdexcept>
#if defined(ENABLE_NINETOOTHED_HYGON_PROJECTION_SWIGLU) && defined(ENABLE_HYGON_API)
#include "h3_proj_swiglu_f16_m16n32k32.h"
#include "h3_proj_swiglu_f16_m32n64k32.h"
#include "h3_proj_swiglu_bf16_m16n32k32.h"
#include "h3_proj_swiglu_bf16_m32n64k32.h"
#include "h3_proj_swiglu_f16_m128n64k64.h"
#include "h3_proj_swiglu_bf16_m128n64k64.h"
#endif

namespace infinicore::op {
namespace {
enum Path { Fused16=0, Fused32, Disabled, DeviceFallback, TypeFallback,
            LayoutFallback, SizeFallback, GraphFallback, PerformanceFallback, PathCount };
#ifdef INFINICORE_PROJECTION_SWIGLU_TESTING
thread_local uint64_t counts[PathCount]={};
void count(Path p){++counts[p];}
#else
void count(Path){}
#endif
Tensor original(const Tensor &x,const Tensor &w,Path path){
    count(path);
    auto packed=linear(x->contiguous(),w->contiguous(),std::nullopt);
    const auto axis=packed->ndim()-1;
    const auto d=w->size(0)/2;
    auto up=packed->narrow({{axis,0,d}});
    auto gate=packed->narrow({{axis,d,d}});
    return swiglu(up,gate);
}
}

Tensor projection_swiglu(const Tensor &x,const Tensor &w,bool prefer_fusion){
    if(x->ndim()<2 || w->ndim()!=2 || !w->size(0) || w->size(0)%2
       || x->size(x->ndim()-1)!=w->size(1)
       || x->dtype()!=w->dtype() || x->device()!=w->device()){
        throw std::invalid_argument("projection_swiglu requires matching input and [up;gate] [2D,K] weight");
    }
    if(!prefer_fusion)return original(x,w,Disabled);
#if defined(ENABLE_NINETOOTHED_HYGON_PROJECTION_SWIGLU) && defined(ENABLE_HYGON_API)
    if(context::isGraphRecording())return original(x,w,GraphFallback);
    if(x->device().getType()!=Device::Type::HYGON)return original(x,w,DeviceFallback);
    const auto dtype=x->dtype();
    if(dtype!=DataType::F16 && dtype!=DataType::BF16)return original(x,w,TypeFallback);
    const uint64_t k=w->size(1),d=w->size(0)/2;
    uint64_t m=x->size(0);
    if(x->ndim()==3){
        if(x->size(1) && m>uint64_t(INT_MAX)/x->size(1))return original(x,w,LayoutFallback);
        m*=x->size(1);
    }
    // Larger F16 row counts showed a strict-reference mismatch in validation.
    // Keep their exact original path until wider numerical qualification.
    if(dtype==DataType::F16 && m>256)return original(x,w,SizeFallback);
    if((x->ndim()!=2 && x->ndim()!=3) || !m || !k || !d
       || k%16 || d%16 || k>INT_MAX || d>uint64_t(INT_MAX)/2 || m>INT_MAX
       || m>uint64_t(INT_MAX)/k || m>uint64_t(INT_MAX)/d
       || d>uint64_t(INT_MAX)/k/2 || !x->is_contiguous() || !w->is_contiguous()
       || (reinterpret_cast<uintptr_t>(x->data())|reinterpret_cast<uintptr_t>(w->data()))%16){
        return original(x,w,LayoutFallback);
    }
    // At the qualified real H3 width, retain fusion through 128 rows and
    // the tested FP16 255/256-row cases; other larger requests use GEMM.
    // The earlier FP16 numerical guard and all other shape guards remain.
    if((dtype==DataType::BF16 || (dtype==DataType::F16 && m!=255 && m!=256)) && k==5376 && d==14336 && m>128)
        return original(x,w,PerformanceFallback);
    // TP2/4/8 shards of configured H3 FFN: all qualified fused cases were
    // slower than the existing GEMM + FP32-intermediate SwiGLU path.
    // Do not extend the TP1 performance qualification to these widths.
    if (k==5376 && (d==7168 || d==3584 || d==1792))
        return original(x,w,PerformanceFallback);
    context::setDevice(x->device());
    auto shape=x->shape();shape.back()=d;
    auto out=Tensor::empty(shape,dtype,x->device());
    uint64_t xs[2]={m,k},ws[2]={d,k},ys[2]={m,d};
    int64_t xt[2]={int64_t(k),1},wt[2]={int64_t(k),1},yt[2]={int64_t(d),1};
    auto weight=const_cast<std::byte*>(w->data());
    NineToothedTensor a{const_cast<std::byte*>(x->data()),xs,xt};
    NineToothedTensor up{weight,ws,wt};
    NineToothedTensor gate{weight+d*k*2,ws,wt}; // F16/BF16: two bytes per element.
    NineToothedTensor y{out->data(),ys,yt};
    const bool small=m<=16;
    const bool wide=k>=1024 && m>=64;
    int rc;
    if(dtype==DataType::F16){
        rc=wide ? launch_h3_proj_swiglu_f16_m128n64k64(context::getStream(),a,up,gate,y)
                : small ? launch_h3_proj_swiglu_f16_m16n32k32(context::getStream(),a,up,gate,y)
                 : launch_h3_proj_swiglu_f16_m32n64k32(context::getStream(),a,up,gate,y);
    }else{
        rc=wide ? launch_h3_proj_swiglu_bf16_m128n64k64(context::getStream(),a,up,gate,y)
                : small ? launch_h3_proj_swiglu_bf16_m16n32k32(context::getStream(),a,up,gate,y)
                 : launch_h3_proj_swiglu_bf16_m32n64k32(context::getStream(),a,up,gate,y);
    }
    if(rc!=0)throw std::runtime_error("[projection_swiglu/ninetoothed-hygon] launch failed: "+std::to_string(rc));
    count(small?Fused16:Fused32);
    return out;
#else
    return original(x,w,Disabled);
#endif
}
#ifdef INFINICORE_PROJECTION_SWIGLU_TESTING
extern "C" uint64_t h3_projection_swiglu_count(unsigned path,int reset){
    auto value=path<PathCount?counts[path]:0;
    if(reset)for(auto &v:counts)v=0;
    return value;
}
#endif
}
