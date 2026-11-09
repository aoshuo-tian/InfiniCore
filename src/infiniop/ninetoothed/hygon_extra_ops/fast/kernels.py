"""Layout-specialized device expressions, generated exclusively by NineToothed."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor

DTYPES = {"f16": nt.float16, "bf16": nt.bfloat16, "f32": nt.float32}
INDEX_TYPES = {"i32": nt.int32, "i64": nt.int64}


def embedding_row1024(output, indices, weight):
    output = output.tile((1, 1024))
    indices = indices.tile((1,)).unsqueeze(1).expand(output.shape)
    weight = weight.flatten().tile((1,)).unsqueeze(1).expand(output.shape)
    return output, indices, weight


def embedding_row4096(output, indices, weight):
    output = output.tile((1, 4096))
    indices = indices.tile((1,)).unsqueeze(1).expand(output.shape)
    weight = weight.flatten().tile((1,)).unsqueeze(1).expand(output.shape)
    return output, indices, weight


def embedding_application(output, indices, weight):
    row = ntl.expand_dims(output.offsets(0), 1)
    feature = ntl.expand_dims(output.offsets(1), 0)
    in_bounds = row < indices.source.shape[0]
    index = ntl.load(indices.source.data_ptr() + row * indices.source.stride(0), mask=in_bounds, other=0)
    valid = in_bounds & (index >= 0) & (index < weight.source.shape[0]) & (feature < weight.source.shape[1])
    safe_index = ntl.where(valid, index, 0)
    output = ntl.load(weight.source.data_ptr() + safe_index * weight.source.stride(0) + feature * weight.source.stride(1), mask=valid, other=0)


def swiglu_dense4096(output, up, gate):
    return tuple(t.tile((4096,)) for t in (output, up, gate))


def swiglu_row4096(output, up, gate):
    return tuple(t.tile((1, 4096)) for t in (output, up, gate))


def swiglu_dense256(output, up, gate):
    return tuple(t.tile((256,)) for t in (output, up, gate))


def swiglu_row256(output, up, gate):
    return tuple(t.tile((1, 256)) for t in (output, up, gate))


def swiglu_application(output, up, gate):
    up32 = ntl.cast(up, ntl.float32)
    gate32 = ntl.cast(gate, ntl.float32)
    output = up32 * gate32 * ntl.sigmoid(gate32)


def jobs():
    for dtype in DTYPES:
        for kind, arrangement, rank in (("dense", swiglu_dense4096, 1), ("row", swiglu_row4096, 2)):
            yield f"h3_swiglu_fast_{kind}_{dtype}", (arrangement, swiglu_application, tuple(Tensor(rank, dtype=DTYPES[dtype]) for _ in range(3)))
        for it in INDEX_TYPES:
            for block, arrangement in ((1024, embedding_row1024), (4096, embedding_row4096)):
                yield f"h3_embedding_fast_{block}_{dtype}_{it}", (arrangement, embedding_application, (
                    Tensor(2, dtype=DTYPES[dtype]), Tensor(1, dtype=INDEX_TYPES[it]), Tensor(2, dtype=DTYPES[dtype])))


def extra_jobs():
    for dtype in DTYPES:
        for kind, arrangement, rank in (("dense", swiglu_dense256, 1), ("row", swiglu_row256, 2)):
            yield f"h3_swiglu_small_{kind}_{dtype}", (arrangement, swiglu_application, tuple(Tensor(rank, dtype=DTYPES[dtype]) for _ in range(3)))
