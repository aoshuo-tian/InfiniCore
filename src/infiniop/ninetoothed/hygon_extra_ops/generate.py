import hashlib
import json
import os
from pathlib import Path
import time

import ninetoothed as nt
from triton.runtime import driver

from kernels import DTYPES, INDEX_TYPES, embedding_premake, swiglu_premake


def main():
    work = Path(os.environ["H3_WORK"])
    output = Path(os.environ.get("NINE_FIX_OUTPUT", work / "sources/InfiniCore/build/ninetoothed"))
    output.mkdir(parents=True, exist_ok=True)
    target = driver.active.get_current_target()
    assert target.backend == "hip" and target.arch == "gfx936", target
    assert Path(nt.__file__).resolve().is_relative_to(work / "sources/ninetoothed")
    records = []
    for dtype in DTYPES:
        jobs = [(f"h3_swiglu_fp32_{dtype}", swiglu_premake(dtype))]
        jobs += [(f"h3_embedding_strided_{dtype}_{it}", embedding_premake(dtype, it)) for it in INDEX_TYPES]
        for name, (arrangement, application, tensors) in jobs:
            print("GENERATE", name, flush=True)
            start = time.monotonic()
            nt.make(arrangement, application, tensors, caller="cuda", kernel_name=name,
                    output_dir=output, num_warps=4, num_stages=1)
            records.append(dict(name=name, seconds=time.monotonic()-start))
    includes = [f'#include "{r["name"]}.h"' for r in records]
    header = "#pragma once\n" + "\n".join(includes) + "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
    header += "int launch_h3_swiglu_fp32_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int);\n"
    header += "int launch_h3_embedding_strided_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int);\n"
    header += "#ifdef __cplusplus\n}\n#endif\n"
    (output / "h3_nine_extra_ops.h").write_text(header)
    source = '#include "h3_nine_extra_ops.h"\n'
    source += "int launch_h3_swiglu_fp32_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor a, NineToothedTensor b, int dtype) {\n"
    for i, dtype in enumerate(DTYPES):
        source += f" if (dtype == {i}) return launch_h3_swiglu_fp32_{dtype}(s,o,a,b);\n"
    source += " return -1;\n}\n"
    source += "int launch_h3_embedding_strided_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor i, NineToothedTensor w, int dtype, int itype) {\n"
    for dc, dtype in enumerate(DTYPES):
        for ic, it in enumerate(INDEX_TYPES):
            source += f" if (dtype == {dc} && itype == {ic}) return launch_h3_embedding_strided_{dtype}_{it}(s,o,i,w);\n"
    source += " return -1;\n}\n"
    (output / "h3_nine_extra_ops.c").write_text(source)
    manifest = dict(target=str(target), records=records,
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                           if p.is_file() and p.name.startswith(("h3_swiglu_fp32_", "h3_embedding_strided_", "h3_nine_extra_ops"))})
    Path(os.environ.get("NINE_FIX_RECORDS", output / "h3_nine_extra_ops.generation.json")).write_text(json.dumps(manifest,indent=2))
    import subprocess
    import sys
    fast_env=os.environ.copy()
    fast_env['NINE_FAST_OUTPUT']=str(output)
    subprocess.run([sys.executable,str(Path(__file__).parent/'fast/generate_fast.py')],env=fast_env,check=True)
    inner_env=os.environ.copy()
    inner_env['NINE_INNER_OUTPUT']=str(output)
    inner_env.pop('NINE_INNER_SELECT',None)
    inner_env.pop('NINE_INNER_POLICY',None)
    subprocess.run([sys.executable,str(Path(__file__).parent/'inner/generate.py')],env=inner_env,check=True)
    print("GENERATION_PASS", len(records), flush=True)


if __name__ == "__main__":
    main()
