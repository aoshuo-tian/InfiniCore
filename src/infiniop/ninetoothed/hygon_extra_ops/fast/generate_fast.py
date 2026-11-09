import hashlib
import json
import os
from pathlib import Path
import time

import ninetoothed as nt
from triton.runtime import driver
from kernels import jobs, extra_jobs, DTYPES, INDEX_TYPES


def main():
    work = Path(os.environ['H3_WORK'])
    output = Path(os.environ['NINE_FAST_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    target = driver.active.get_current_target()
    assert (target.backend, target.arch) == ('hip', 'gfx936'), target
    assert Path(nt.__file__).resolve().is_relative_to(work/'sources/ninetoothed')
    records = []
    selected = os.environ.get('NINE_FAST_SELECT', '')
    all_jobs=list(jobs())+list(extra_jobs())
    for name, (arrangement, application, tensors) in all_jobs:
        if selected and name not in selected.split(','):
            continue
        if os.environ.get('NINE_FAST_EXTRAS_ONLY')=='1' and '_fast_' in name:
            continue
        print('GENERATE', name, flush=True)
        start = time.monotonic()
        nt.make(arrangement, application, tensors, caller='cuda', kernel_name=name,
                output_dir=output, num_warps=4, num_stages=1)
        records.append(dict(name=name, seconds=time.monotonic()-start))
        (output/'generation-progress.json').write_text(json.dumps(records,indent=2))
    if selected:
        print('SELECTED_GENERATION_PASS', len(records), flush=True)
        return
    names = [name for name, _ in all_jobs]
    header = '#pragma once\n'+'\n'.join(f'#include "{n}.h"' for n in names)+'\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
    header += 'int launch_h3_embedding_fast_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int, int);\n'
    header += 'int launch_h3_swiglu_fast_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int);\n'
    header += 'int launch_h3_swiglu_small_all(NineToothedStream, NineToothedTensor, NineToothedTensor, NineToothedTensor, int, int);\n'
    header += '#ifdef __cplusplus\n}\n#endif\n'
    (output/'h3_nine_extra_fast.h').write_text(header)
    source = '#include "h3_nine_extra_fast.h"\n'
    source += 'int launch_h3_embedding_fast_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor i, NineToothedTensor w, int dtype, int itype, int block) {\n'
    for dc, dtype in enumerate(DTYPES):
        for ic, it in enumerate(INDEX_TYPES):
            for block in (1024,4096):
                source += f' if(dtype=={dc} && itype=={ic} && block=={block})return launch_h3_embedding_fast_{block}_{dtype}_{it}(s,o,i,w);\n'
    source += ' return -1;\n}\n'
    source += 'int launch_h3_swiglu_small_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor a, NineToothedTensor b, int dtype, int row) {\n'
    for dc, dtype in enumerate(DTYPES):
        for kind, row in (('dense',0),('row',1)):
            source += f' if(dtype=={dc} && row=={row})return launch_h3_swiglu_small_{kind}_{dtype}(s,o,a,b);\n'
    source += ' return -1;\n}\n'
    source += 'int launch_h3_swiglu_fast_all(NineToothedStream s, NineToothedTensor o, NineToothedTensor a, NineToothedTensor b, int dtype, int row) {\n'
    for dc, dtype in enumerate(DTYPES):
        for kind, row in (('dense',0),('row',1)):
            source += f' if(dtype=={dc} && row=={row})return launch_h3_swiglu_fast_{kind}_{dtype}(s,o,a,b);\n'
    source += ' return -1;\n}\n'
    (output/'h3_nine_extra_fast.c').write_text(source)
    manifest = dict(target=str(target), records=records, files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'generation-manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FAST_GENERATION_PASS',len(records),flush=True)


if __name__=='__main__':
    main()
