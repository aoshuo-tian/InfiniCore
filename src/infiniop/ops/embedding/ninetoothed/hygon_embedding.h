#pragma once
#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../ninetoothed/hygon_layout.h"
#include "h3_embedding_rows_ops.h"

namespace op::embedding::hygon_ninetoothed {
#ifdef H3_NINE_INNER_TEST
// Test builds only; production has neither counters nor exported test accessors.
static thread_local uint64_t inner_test_counts[2]={0,0};
extern "C" __attribute__((used,visibility("default"))) uint64_t h3_embedding_inner_test_count(unsigned path,int reset) {
    if(reset){inner_test_counts[0]=0;inner_test_counts[1]=0;return 0;}
    return path<2 ? inner_test_counts[path] : 0;
}
#endif

class Descriptor final : public InfiniopDescriptor {
    hygon_nine_layout::Layout<6> out_;
    hygon_nine_layout::Layout<5> indices_;
    hygon_nine_layout::Layout<2> weight_;
    int dtype_,itype_;
    bool row_=false;
    bool inner_=false;
    bool rows_=false;
    int64_t index_step_=1,out_row_step_=0;
    Descriptor(infiniopHandle_t h):InfiniopDescriptor{h->device,h->device_id},dtype_(-1),itype_(-1){}
public:
    static infiniStatus_t create(infiniopHandle_t h,Descriptor **dst,
        infiniopTensorDescriptor_t out,infiniopTensorDescriptor_t indices,infiniopTensorDescriptor_t weight) {
        if(!h || !dst || !out || !indices || !weight) return INFINI_STATUS_NULL_POINTER;
        if(weight->ndim()!=2 || out->ndim()!=indices->ndim()+1) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        auto expected=indices->shape();expected.push_back(weight->dim(1));
        if(out->shape()!=expected) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        int dtype=hygon_nine_layout::dtype_code(weight->dtype());
        int itype=indices->dtype()==INFINI_DTYPE_I32 ? 0 : indices->dtype()==INFINI_DTYPE_I64 ? 1 : -1;
        if(dtype<0 || itype<0 || out->dtype()!=weight->dtype()) return INFINI_STATUS_BAD_TENSOR_DTYPE;
        auto desc=new Descriptor(h);
        auto status=hygon_nine_layout::make(out,desc->out_);
        if(status==INFINI_STATUS_SUCCESS)status=hygon_nine_layout::make(indices,desc->indices_);
        if(status==INFINI_STATUS_SUCCESS)status=hygon_nine_layout::make(weight,desc->weight_);
        if(status==INFINI_STATUS_SUCCESS && !desc->out_.nonoverlapping())status=INFINI_STATUS_BAD_TENSOR_STRIDES;
        if(status==INFINI_STATUS_SUCCESS && desc->out_.count && weight->dim(0)==0)status=INFINI_STATUS_BAD_TENSOR_SHAPE;
        if(status!=INFINI_STATUS_SUCCESS){delete desc;return status;}
        // Collapse only contiguous index/output pairs, or an already two-dimensional output.
        desc->row_=indices->ndim()==1 || (desc->indices_.contiguous() && desc->out_.contiguous());
        if(desc->row_) {
            desc->index_step_=indices->ndim()==1 ? desc->indices_.strides[4] : 1;
            desc->out_row_step_=indices->ndim()==1 ? desc->out_.strides[4] : weight->dim(1);
        }
        desc->inner_=desc->row_ && desc->out_.strides[5]==1 && desc->weight_.strides[1]==1
            && desc->index_step_==1 && desc->indices_.count%16==0 && weight->dim(1)%16==0
            && desc->out_row_step_%16==0 && desc->weight_.strides[0]%16==0;
        // New separately generated strict variant: only feature sizes, not
        // index count, carry the :16 contract. Keep bounds/alias guards below.
        desc->rows_=desc->row_ && desc->out_.strides[5]==1 && desc->weight_.strides[1]==1
            && desc->index_step_==1 && desc->indices_.count%16!=0 && weight->dim(1)%16==0
            && desc->out_row_step_%16==0 && desc->weight_.strides[0]%16==0;
        desc->dtype_=dtype;desc->itype_=itype;*dst=desc;return INFINI_STATUS_SUCCESS;
    }
    infiniStatus_t calculate(void *out,const void *indices,const void *weight,void *stream) const {
        if(!out_.count) return INFINI_STATUS_SUCCESS;
        hygon_nine_layout::Range o{},i{},w{};
        if(!out_.range(out,o) || !indices_.range(indices,i) || !weight_.range(weight,w))return INFINI_STATUS_BAD_PARAM;
        if(!hygon_nine_layout::separate(o,i) || !hygon_nine_layout::separate(o,w))return INFINI_STATUS_BAD_PARAM;
        const bool use_inner=inner_ && ((reinterpret_cast<uintptr_t>(out)
            |reinterpret_cast<uintptr_t>(indices)|reinterpret_cast<uintptr_t>(weight))&15)==0;
#ifdef H3_NINE_INNER_TEST
        ++inner_test_counts[use_inner ? 0 : 1];
#endif
        const bool use_rows=rows_ && ((reinterpret_cast<uintptr_t>(out)
            |reinterpret_cast<uintptr_t>(indices)|reinterpret_cast<uintptr_t>(weight))&15)==0;
        int rc;
        if(row_) {
            uint64_t os[2]={indices_.count,weight_.shape[1]},is[1]={indices_.count};
            int64_t ost[2]={out_row_step_,out_.strides[5]},ist[1]={index_step_};
            NineToothedTensor ot{out,os,ost},it{const_cast<void*>(indices),is,ist};
            auto launch=use_inner ? launch_h3_embedding_inner_all
                : (use_rows ? launch_h3_embedding_rows_all : launch_h3_embedding_fast_all);
            rc=launch(stream,ot,it,weight_.tensor(weight),dtype_,itype_,os[1]<=1024 ? 1024 : 4096);
        } else {
            rc=launch_h3_embedding_strided_all(stream,out_.tensor(out),indices_.tensor(indices),weight_.tensor(weight),dtype_,itype_);
        }
        return rc==0 ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
    }
};
} // namespace op::embedding::hygon_ninetoothed
