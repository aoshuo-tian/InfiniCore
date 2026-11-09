#if defined(ENABLE_HYGON_API) && defined(ENABLE_NINETOOTHED_CAST)

#include "cast_ninetoothed_hygon.hpp"
#include "h3_cast_all.h"
#include "h3_cast_dense_f32_bf16.h"
#include "h3_cast_dense_bf16_f32.h"
#include "infinicore/context/context.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinicore::op::cast_impl {
namespace {

int dtype_code(DataType dtype) {
    switch (dtype) {
    case DataType::F16: return 1;
    case DataType::BF16: return 2;
    case DataType::F32: return 3;
    default:
        throw std::runtime_error("[cast/ninetoothed-hygon] expected F16/BF16/F32");
    }
}

void check_output_layout(const Tensor &output) {
    std::vector<std::pair<uint64_t, uint64_t>> dimensions;
    for (size_t i = 0; i < output->ndim(); ++i) {
        if (output->size(i) > 1) {
            if (output->stride(i) <= 0) {
                throw std::runtime_error("[cast/ninetoothed-hygon] overlapping output strides");
            }
            dimensions.emplace_back(output->stride(i), output->size(i));
        }
    }
    std::sort(dimensions.begin(), dimensions.end());
    uint64_t span = 1;
    for (const auto &dimension : dimensions) {
        if (dimension.first < span) {
            throw std::runtime_error("[cast/ninetoothed-hygon] output layout not provably non-overlapping");
        }
        if ((dimension.second - 1) > (std::numeric_limits<uint64_t>::max() - span) / dimension.first) {
            throw std::runtime_error("[cast/ninetoothed-hygon] output layout overflow");
        }
        span += (dimension.second - 1) * dimension.first;
    }
}

std::pair<uintptr_t, uintptr_t> address_range(const Tensor &tensor) {
    uint64_t last = 0;
    for (size_t i = 0; i < tensor->ndim(); ++i) {
        if (tensor->stride(i) < 0) {
            throw std::runtime_error("[cast/ninetoothed-hygon] negative stride unsupported");
        }
        const uint64_t extent = tensor->size(i) - 1;
        const uint64_t stride = tensor->stride(i);
        if (extent && stride > (std::numeric_limits<uint64_t>::max() - last) / extent) {
            throw std::runtime_error("[cast/ninetoothed-hygon] tensor layout overflow");
        }
        last += extent * stride;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data());
    const uint64_t bytes = dsize(tensor->dtype());
    const uint64_t available = std::numeric_limits<uintptr_t>::max() - begin;
    if (available < bytes - 1 || last > (available - (bytes - 1)) / bytes) {
        throw std::runtime_error("[cast/ninetoothed-hygon] tensor address overflow");
    }
    return {begin, begin + last * bytes + bytes - 1};
}

} // namespace

void ninetoothed_hygon(Tensor output, const Tensor &input) {
    const int input_dtype = dtype_code(input->dtype());
    const int output_dtype = dtype_code(output->dtype());
    if (input->ndim() > 5 || output->ndim() > 5) {
        throw std::runtime_error("[cast/ninetoothed-hygon] supports rank 0 through 5");
    }
    if (input->numel() == 0) {
        return;
    }
    check_output_layout(output);
    if (input->data() == output->data()) {
        if (input_dtype == output_dtype && input->strides() == output->strides()) {
            return;
        }
        throw std::runtime_error("[cast/ninetoothed-hygon] aliased input/output unsupported");
    }
    std::array<uint64_t, 5> shape{1, 1, 1, 1, 1};
    std::array<int64_t, 5> input_strides{0, 0, 0, 0, 0};
    std::array<int64_t, 5> output_strides{0, 0, 0, 0, 0};
    const size_t offset = 5 - input->ndim();
    for (size_t i = 0; i < input->ndim(); ++i) {
        if (input->stride(i) < 0) {
            throw std::runtime_error("[cast/ninetoothed-hygon] negative input stride unsupported");
        }
        shape[offset + i] = input->size(i);
        input_strides[offset + i] = input->stride(i);
        output_strides[offset + i] = output->stride(i);
    }
    const auto source_range = address_range(input);
    const auto output_range = address_range(output);
    if (source_range.first <= output_range.second && output_range.first <= source_range.second) {
        throw std::runtime_error("[cast/ninetoothed-hygon] overlapping source/output address ranges unsupported");
    }
    // H3's hot conversions are dense FP32 <-> BF16. Avoid the generic
    // five-dimensional coordinate-unflattening work in those calls.
    if (input->is_contiguous() && output->is_contiguous()
        && ((input_dtype == 3 && output_dtype == 2)
            || (input_dtype == 2 && output_dtype == 3))) {
        uint64_t count[1]{input->numel()};
        int64_t stride[1]{1};
        NineToothedTensor source{const_cast<std::byte *>(input->data()), count, stride};
        NineToothedTensor destination{output->data(), count, stride};
        context::setDevice(output->device());
        const int status = input_dtype == 3
            ? launch_h3_cast_dense_f32_bf16(context::getStream(), source, destination)
            : launch_h3_cast_dense_bf16_f32(context::getStream(), source, destination);
        if (status != 0) {
            throw std::runtime_error("[cast/ninetoothed-hygon] dense AOT launch failed: " + std::to_string(status));
        }
        return;
    }
    // The NineToothed host ABI uses void* for both read-only and output tensors.
    NineToothedTensor source{const_cast<std::byte *>(input->data()), shape.data(), input_strides.data()};
    NineToothedTensor destination{output->data(), shape.data(), output_strides.data()};
    context::setDevice(output->device());
    const int status = launch_h3_cast_all(
        context::getStream(), source, destination, input_dtype, output_dtype);
    if (status != 0) {
        throw std::runtime_error("[cast/ninetoothed-hygon] AOT launch failed: " + std::to_string(status));
    }
}

} // namespace infinicore::op::cast_impl
#endif
