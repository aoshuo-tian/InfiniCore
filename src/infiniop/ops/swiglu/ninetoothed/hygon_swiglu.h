#pragma once
#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../ninetoothed/hygon_layout.h"

namespace op::swiglu::hygon_ninetoothed {
#ifdef H3_NINE_INNER_TEST
// Test builds only; production has neither counters nor exported test accessors.
static thread_local uint64_t inner_test_counts[2]={0,0};
extern "C" __attribute__((used,visibility("default"))) uint64_t h3_swiglu_inner_test_count(unsigned path,int reset) {
    if(reset){inner_test_counts[0]=0;inner_test_counts[1]=0;return 0;}
    return path<2 ? inner_test_counts[path] : 0;
}
#endif

class Descriptor final : public InfiniopDescriptor {
    hygon_nine_layout::Layout<5> out_,up_,gate_;
    int dtype_;
    int fast_=-1;
    bool inner_=false;
    Descriptor(infiniopHandle_t h):InfiniopDescriptor{h->device,h->device_id},dtype_(-1){}
public:
    size_t workspaceSize()const {return 0;}
    static infiniStatus_t create(infiniopHandle_t h,Descriptor **dst,
        infiniopTensorDescriptor_t out,std::vector<infiniopTensorDescriptor_t> inputs) {
        if(!h || !dst || !out || inputs.size()!=2 || !inputs[0] || !inputs[1])return INFINI_STATUS_NULL_POINTER;
        auto up=inputs[0],gate=inputs[1];
        if(out->shape()!=up->shape() || out->shape()!=gate->shape())return INFINI_STATUS_BAD_TENSOR_SHAPE;
        int dtype=hygon_nine_layout::dtype_code(out->dtype());
        if(dtype<0 || up->dtype()!=out->dtype() || gate->dtype()!=out->dtype())return INFINI_STATUS_BAD_TENSOR_DTYPE;
        auto desc=new Descriptor(h);
        auto status=hygon_nine_layout::make(out,desc->out_);
        if(status==INFINI_STATUS_SUCCESS)status=hygon_nine_layout::make(up,desc->up_);
        if(status==INFINI_STATUS_SUCCESS)status=hygon_nine_layout::make(gate,desc->gate_);
        if(status==INFINI_STATUS_SUCCESS && !desc->out_.nonoverlapping())status=INFINI_STATUS_BAD_TENSOR_STRIDES;
        if(status!=INFINI_STATUS_SUCCESS){delete desc;return status;}
        if(desc->out_.contiguous() && desc->up_.contiguous() && desc->gate_.contiguous())desc->fast_=0;
        else if(out->ndim()==2)desc->fast_=1;
        // A singleton batch can be removed without changing any addresses.
        // Keep the original two row/feature strides and all range/alias guards.
        // Limit this new route to the qualified low-precision feature-contiguous
        // case; multi-batch, F32 and other layouts retain their old selection.
        else if(out->ndim()==3 && out->dim(0)==1 && dtype<2
            && desc->out_.strides[4]==1 && desc->up_.strides[4]==1
            && desc->gate_.strides[4]==1)desc->fast_=1;
        desc->inner_=(desc->fast_==1 && desc->out_.shape[4]%16==0
            && desc->out_.strides[4]==1 && desc->up_.strides[4]==1 && desc->gate_.strides[4]==1
            && desc->out_.strides[3]%16==0 && desc->up_.strides[3]%16==0 && desc->gate_.strides[3]%16==0);
        desc->dtype_=dtype;*dst=desc;return INFINI_STATUS_SUCCESS;
    }
    infiniStatus_t calculate(void *workspace,size_t workspace_size,void *out,
        std::vector<const void*> inputs,void *stream)const {
        (void)workspace;(void)workspace_size;
        if(!out_.count)return INFINI_STATUS_SUCCESS;
        if(inputs.size()!=2)return INFINI_STATUS_BAD_PARAM;
        using namespace hygon_nine_layout;
        for(size_t j=0;j<2;++j){
            const auto &source=j ? gate_ : up_;
            if(!exact_alias(out_,out,source,inputs[j]) && !disjoint_or_packed(out_,out,source,inputs[j]))return INFINI_STATUS_BAD_PARAM;
        }
        const bool use_inner=inner_ && out_.count<=262144 && ((reinterpret_cast<uintptr_t>(out)
            |reinterpret_cast<uintptr_t>(inputs[0])|reinterpret_cast<uintptr_t>(inputs[1]))&15)==0;
#ifdef H3_NINE_INNER_TEST
        ++inner_test_counts[use_inner ? 0 : 1];
#endif
        int rc;
        if(fast_==0) {
            uint64_t shape[1]={out_.count};int64_t strides[1]={1};
            NineToothedTensor ot{out,shape,strides},ut{const_cast<void*>(inputs[0]),shape,strides},gt{const_cast<void*>(inputs[1]),shape,strides};
            auto launch=use_inner ? launch_h3_swiglu_inner_all
                : (out_.count<=262144 ? launch_h3_swiglu_small_all : launch_h3_swiglu_fast_all);
            rc=launch(stream,ot,ut,gt,dtype_,0);
        } else if(fast_==1) {
            uint64_t shape[2]={out_.shape[3],out_.shape[4]};
            int64_t ost[2]={out_.strides[3],out_.strides[4]},ust[2]={up_.strides[3],up_.strides[4]},gst[2]={gate_.strides[3],gate_.strides[4]};
            NineToothedTensor ot{out,shape,ost},ut{const_cast<void*>(inputs[0]),shape,ust},gt{const_cast<void*>(inputs[1]),shape,gst};
            auto launch=use_inner ? launch_h3_swiglu_inner_all
                : (out_.count<=262144 ? launch_h3_swiglu_small_all : launch_h3_swiglu_fast_all);
            rc=launch(stream,ot,ut,gt,dtype_,1);
        } else {
            rc=launch_h3_swiglu_fp32_all(stream,out_.tensor(out),up_.tensor(inputs[0]),gate_.tensor(inputs[1]),dtype_);
        }
        return rc==0 ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
    }
};
} // namespace op::swiglu::hygon_ninetoothed
