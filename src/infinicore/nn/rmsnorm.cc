#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::nn {

RMSNorm::RMSNorm(size_t normalized_shape, double eps, const DataType &dtype, const Device &device)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      dtype_(dtype) {

    device_ = device;

    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape}, dtype_, device));
}

Tensor RMSNorm::forward(const Tensor &x) const {
    // Delegate to InfiniCore op (backed by InfiniRT/InfiniOP)
    // Validation is handled by the op layer
    // Hygon's descriptor accepts 2D/3D tensors; H3 Q/K have [B,S,H,D].
    if (device_.getType() == Device::Type::HYGON && x->ndim() > 3) {
        if (normalized_shape_ == 0 || x->size(x->ndim() - 1) != normalized_shape_) {
            throw std::invalid_argument("RMSNorm: last dimension must match normalized_shape");
        }
        auto flattened = x->contiguous()->view(
            {x->numel() / normalized_shape_, normalized_shape_});
        return op::rms_norm(flattened, weight_, static_cast<float>(eps_))->view(x->shape());
    }
    return op::rms_norm(x, weight_, static_cast<float>(eps_));
}

void RMSNorm::forward_inplace(Tensor &x, Tensor &residual) const {
    if (!residual) {
        residual = x;
        x = op::rms_norm(x, weight_, static_cast<float>(eps_));
    } else {
        if (device_.getType() == Device::Type::CPU
            || device_.getType() == Device::Type::NVIDIA
            || device_.getType() == Device::Type::ILUVATAR
            || device_.getType() == Device::Type::METAX
            || device_.getType() == Device::Type::MOORE
            || device_.getType() == Device::Type::ALI
            || device_.getType() == Device::Type::CAMBRICON
            || device_.getType() == Device::Type::HYGON) {
            op::add_rms_norm_inplace(x, residual, weight_, static_cast<float>(eps_));
        } else {
            op::add_(residual, x, residual);
            op::rms_norm_(x, residual, weight_, static_cast<float>(eps_));
        }
    }
}

std::string RMSNorm::extra_repr() const {
    return "RMSNorm(normalized_shape=" + std::to_string(normalized_shape_) + ", eps=" + std::to_string(eps_) + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
