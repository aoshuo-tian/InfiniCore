"""Device computation expressed through NineToothed arrangement/application."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def swiglu_arrangement(output, up, gate):
    return tuple(t.flatten().tile((256,)) for t in (output, up, gate))


def swiglu_application(output, up, gate):
    up32 = ntl.cast(up, ntl.float32)
    gate32 = ntl.cast(gate, ntl.float32)
    output = up32 * gate32 * ntl.sigmoid(gate32)  # noqa: F841


def embedding_arrangement(output, indices, weight):
    output = output.flatten().tile((256,))
    # Only source-indexed loads are used for these tensors. Broadcast their
    # singleton tile grids to the output grid without loading whole tables.
    indices = indices.flatten().tile((1,)).expand(output.shape)
    weight = weight.flatten().tile((1,)).expand(output.shape)
    return output, indices, weight


def embedding_application(output, indices, weight):
    i0, i1, i2 = output.offsets(0), output.offsets(1), output.offsets(2)
    i3, i4, feature = output.offsets(3), output.offsets(4), output.offsets(5)
    in_bounds = (i0 < indices.source.shape[0]) & (i1 < indices.source.shape[1]) & (i2 < indices.source.shape[2]) & (i3 < indices.source.shape[3]) & (i4 < indices.source.shape[4])
    offset = i0 * indices.source.stride(0) + i1 * indices.source.stride(1) + i2 * indices.source.stride(2) + i3 * indices.source.stride(3) + i4 * indices.source.stride(4)
    index = ntl.load(indices.source.data_ptr() + offset, mask=in_bounds, other=0)
    valid = in_bounds & (index >= 0) & (index < weight.source.shape[0])
    safe_index = ntl.where(valid, index, 0)
    weight_offset = safe_index * weight.source.stride(0) + feature * weight.source.stride(1)
    output = ntl.load(weight.source.data_ptr() + weight_offset, mask=valid, other=0)  # noqa: F841


DTYPES = {"f16": nt.float16, "bf16": nt.bfloat16, "f32": nt.float32}
INDEX_TYPES = {"i32": nt.int32, "i64": nt.int64}


def swiglu_premake(dtype):
    return swiglu_arrangement, swiglu_application, tuple(Tensor(5, dtype=DTYPES[dtype]) for _ in range(3))


def embedding_premake(dtype, index_dtype):
    return embedding_arrangement, embedding_application, (
        Tensor(6, dtype=DTYPES[dtype]),
        Tensor(5, dtype=INDEX_TYPES[index_dtype]),
        Tensor(2, dtype=DTYPES[dtype]),
    )
