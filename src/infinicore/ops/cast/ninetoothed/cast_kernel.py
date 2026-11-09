"""H3 cast expressed exclusively in NineToothed; no device-kernel fallback."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output):
    # Flatten logical coordinates, retaining the original strided address map.
    return input.flatten().tile((1024,)), output.flatten().tile((1024,))


def arrangement_dense(input, output):
    return input.tile((1024,)), output.tile((1024,))


def to_f16(input, output):
    output = ntl.cast(input, ntl.float16)  # noqa: F841


def to_bf16(input, output):
    output = ntl.cast(input, ntl.bfloat16)  # noqa: F841


def to_f32(input, output):
    output = ntl.cast(input, ntl.float32)  # noqa: F841


DTYPES = {"f16": nt.float16, "bf16": nt.bfloat16, "f32": nt.float32}
APPLICATIONS = {"f16": to_f16, "bf16": to_bf16, "f32": to_f32}


def premake(input_dtype, output_dtype):
    return arrangement, APPLICATIONS[output_dtype], (
        Tensor(5, dtype=DTYPES[input_dtype]),
        Tensor(5, dtype=DTYPES[output_dtype]),
    )


def premake_dense(input_dtype, output_dtype):
    return arrangement_dense, APPLICATIONS[output_dtype], (
        Tensor(1, dtype=DTYPES[input_dtype]),
        Tensor(1, dtype=DTYPES[output_dtype]),
    )
