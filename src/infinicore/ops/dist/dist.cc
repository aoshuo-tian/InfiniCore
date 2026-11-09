#include "infinicore/ops/dist.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "../../utils.hpp"
#include "../../../utils/custom_types.h"
#include <cmath>
#include <limits>
#include <type_traits>
#ifdef ENABLE_NINETOOTHED_HYGON_DIST
#include "h3_dist_f16.h"
#include "h3_dist_bf16.h"
#include "h3_dist_f32.h"
#endif

namespace infinicore::op {
namespace {
Shape validate(Tensor y, const Tensor &a, const Tensor &b, double p) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y,a,b);
    if(y->shape()!=Shape{} || y->dtype()!=a->dtype() || b->dtype()!=a->dtype())
        throw std::runtime_error("dist: output must be a scalar of the matching input dtype");
    auto dt=a->dtype();
    if(dt!=DataType::F16 && dt!=DataType::BF16 && dt!=DataType::F32 && dt!=DataType::F64)
        throw std::runtime_error("dist: only real floating tensors are supported");
    if(std::isnan(p))throw std::runtime_error("dist: p must not be NaN");
    if(context::isGraphRecording())throw std::runtime_error("dist: graph recording is not supported");
    auto dev=a->device().getType();
    if(dev!=Device::Type::CPU) {
#ifdef ENABLE_NINETOOTHED_HYGON_DIST
        if(dev!=Device::Type::HYGON || dt==DataType::F64)
            throw std::runtime_error("dist: this device/dtype is not supported by this build");
        if(std::isfinite(p) && (!std::isfinite(static_cast<float>(p)) || (p!=0.0 && static_cast<float>(p)==0.0f)))
            throw std::runtime_error("dist: p is outside the Hygon FP32 scalar range");
#else
        throw std::runtime_error("dist: enable ninetoothed-hygon-dist for the Hygon backend");
#endif
    }
    Shape shape(std::max(a->ndim(),b->ndim()),1);
    for(Size i=0;i<shape.size();++i) {
        auto da=i<a->ndim()?a->shape()[a->ndim()-1-i]:1;
        auto db=i<b->ndim()?b->shape()[b->ndim()-1-i]:1;
        if(da!=db && da!=1 && db!=1)throw std::runtime_error("dist: inputs are not broadcastable");
        shape[shape.size()-1-i]=da==1?db:da;
    }
    Size n=1;
    for(auto d:shape) {
        if(d>static_cast<Size>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::runtime_error("dist: broadcast dimension overflow");
        if(d && n>static_cast<Size>(std::numeric_limits<ptrdiff_t>::max())/d)
            throw std::runtime_error("dist: broadcast size overflow");
        n*=d;
    }
    if(n>static_cast<Size>(std::numeric_limits<ptrdiff_t>::max())/a->element_size())
        throw std::runtime_error("dist: broadcast byte size overflow");
    if(!n && (p<0 || std::isinf(p)))throw std::runtime_error("dist: empty reduction has no identity for this p");
    for(auto t:{a,b})for(auto st:t->strides())if(st<0)throw std::runtime_error("dist: negative strides are not supported");
    return shape;
}

template<class T> void cpu_dist(Tensor y,const Tensor &a,const Tensor &b,double p) {
    const auto *ap=reinterpret_cast<const T*>(a->data());
    const auto *bp=reinterpret_cast<const T*>(b->data());
    auto *yp=reinterpret_cast<T*>(y->data());
    using C=std::conditional_t<std::is_same_v<T,double>,double,float>;
    auto diff=[&](Size i) {
        // torch.dist subtracts in the input dtype before norm accumulation.
        return std::abs(utils::cast<double>(utils::cast<T>(utils::cast<C>(ap[i])-utils::cast<C>(bp[i]))));
    };
    double hi=0,lo=std::numeric_limits<double>::infinity(),count=0;
    bool nan=false;
    for(Size i=0;i<a->numel();++i){auto d=diff(i);nan|=std::isnan(d);count+=(d!=0);hi=std::max(hi,d);lo=std::min(lo,d);}
    double result;
    if(p==0)result=count;
    else if(nan)result=std::numeric_limits<double>::quiet_NaN();
    else if(p==std::numeric_limits<double>::infinity())result=hi;
    else if(p==-std::numeric_limits<double>::infinity())result=lo;
    else {
        auto scale=p>0?hi:lo;
        if(scale==0 || std::isinf(scale))result=scale;
        else {double total=0;for(Size i=0;i<a->numel();++i)total+=std::pow(diff(i)/scale,p);result=scale*std::pow(total,1.0/p);}
    }
    *yp=utils::cast<T>(result);
}

struct Plan {Tensor y,a,b,result;double p;};
void *plan(Tensor y,const Tensor &a,const Tensor &b,double p) {
    auto shape=validate(y,a,b,p);
    // Materialize broadcasts and slices through the existing Rearrange path.
    // Copies also protect an out tensor that aliases either input.
    std::vector<int64_t> dims(shape.begin(),shape.end());
    Size count=1;for(auto d:shape)count*=d;
    auto ac=count?broadcast_to(a,dims):Tensor::empty(shape,a->dtype(),a->device());
    auto bc=count?broadcast_to(b,dims):Tensor::empty(shape,b->dtype(),b->device());
    // Generated launchers assume aligned pointers. A public scalar out may be
    // an interior slice; keep it outside the AOT ABI and copy back afterwards.
    auto result=y->device().getType()==Device::Type::CPU?Tensor{}:Tensor::empty({},y->dtype(),y->device());
    return new Plan{y,ac,bc,result,p};
}
void run(void *ptr) {
    auto &m=*static_cast<Plan*>(ptr);
    if(m.a->numel()==0){auto zero=Tensor::zeros({},m.y->dtype(),m.y->device());m.y->copy_from(zero);return;}
    if(m.y->device().getType()==Device::Type::CPU){
        switch(m.y->dtype()) {
            case DataType::F16:return cpu_dist<fp16_t>(m.y,m.a,m.b,m.p);
            case DataType::BF16:return cpu_dist<bf16_t>(m.y,m.a,m.b,m.p);
            case DataType::F32:return cpu_dist<float>(m.y,m.a,m.b,m.p);
            case DataType::F64:return cpu_dist<double>(m.y,m.a,m.b,m.p);
            default:throw std::runtime_error("dist: unsupported dtype");
        }
    }
#ifdef ENABLE_NINETOOTHED_HYGON_DIST
    uint64_t n=m.a->numel(),one=1;int64_t stride=1;
    if((reinterpret_cast<uintptr_t>(m.a->data())|reinterpret_cast<uintptr_t>(m.b->data())|reinterpret_cast<uintptr_t>(m.result->data()))&15)
        throw std::runtime_error("dist: internal AOT buffers must be 16-byte aligned");
    NineToothedTensor a{m.a->data(),&n,&stride},b{m.b->data(),&n,&stride},y{m.result->data(),&one,&stride};
    double p=m.p;int32_t mode=m.p==0?0:(std::isinf(m.p)?(m.p>0?1:2):3);
    NineToothedTensor pt{&p,nullptr,nullptr},mt{&mode,nullptr,nullptr};
    auto launch=m.y->dtype()==DataType::F16?launch_h3_dist_f16:(m.y->dtype()==DataType::BF16?launch_h3_dist_bf16:launch_h3_dist_f32);
    if(launch(context::getStream(),y,a,b,pt,mt)!=0)throw std::runtime_error("dist: NineToothed launch failed");
    m.y->copy_from(m.result);
#endif
}
void cleanup(void **ptr) {delete static_cast<Plan*>(*ptr);*ptr=nullptr;}
}
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dist);
Dist::Dist(Tensor y,const Tensor &a,const Tensor &b,double p) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y,a,b);
    context::setDevice(y->device());
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(),y,a,b,p);
}
void Dist::execute(Tensor y,const Tensor &a,const Tensor &b,double p) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dist,y,a,b,p);
}
Tensor dist(const Tensor &a,const Tensor &b,double p) {
    auto y=Tensor::empty({},a->dtype(),a->device());dist_(y,a,b,p);return y;
}
void dist_(Tensor y,const Tensor &a,const Tensor &b,double p) {Dist::execute(y,a,b,p);}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dist,&plan,&run,&cleanup);
}
