import ctypes
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import median

from public_check import WORKLOAD, make_case, mode, torch, triton

artifact = Path(__file__).parent
data, chain = make_case(WORKLOAD)
outputs = [torch.empty_like(data.q) for _ in range(3)]
os.environ["FLAG_ATTN_MSA_SCORE_PAGE_MAJOR"] = "1"
os.environ["FLAG_ATTN_MSA_PRELOAD_PAGE_META"] = "1"
flag = "FLAG_ATTN_MSA_MAX_STATE_COLUMN"
original = triton.runtime.jit.JITFunction.run
from triton.backends.enflame import compiler as enflame_compiler

make_gcuir = enflame_compiler.make_gcuir
active = 0

def replace_once(ir, old, new):
    assert ir.count(old) == 1, (old, ir.count(old))
    return ir.replace(old, new, 1)

def rewrite_state(mod, metadata, options):
    generated = make_gcuir(mod, metadata, options)
    if metadata["name"] != "_gqa_sparse_fwd_direct" or active == 0:
        return generated
    ir = (artifact / "base.gcuir").read_text()
    dma = next(line for line in ir.splitlines(keepends=True)
               if line.startswith('          "memref.dma_start"(%266, %22, %269, %22, %28, %41, %22)'))
    if active == 1:
        ir = replace_once(ir, dma, "")
        pivot = '          "memref.dma_wait"(%41, %14, %3)'
        ir = replace_once(ir, pivot, dma + pivot)
    else:
        assert active == 2
        previous = next(line for line in ir.splitlines(keepends=True)
                        if line.startswith('          %634 = "vector.load"(%arg34,'))
        assert ir.count("%634") == 2 and ir.count("%arg34") == 3
        assert '          %592 = "vector.load"(%arg34,' in ir
        ir = replace_once(ir, previous, "")
        ir = replace_once(ir, '%634, %635', '%592, %635')
        ir = replace_once(ir, dma, "")
        wait_and_yield = ('          "memref.dma_wait"(%41, %22, %28) : (memref<5xi32>, index, index) -> ()\n'
                          '          "scf.yield"(%269, %267, %268)')
        ir = replace_once(ir, wait_and_yield, '          "scf.yield"(%arg34, %267, %268)')
        for buffer in ("%266", "%269"):
            allocation = next(line for line in ir.splitlines(keepends=True)
                              if line.startswith(f'        {buffer} = "memref.alloc"'))
            ir = replace_once(ir, allocation, "")
        assert not re.search(r"%269\b", ir)
        assert len(re.findall(r"%266\b", ir)) == 3
        ir = re.sub(r"%266\b", "%arg34", ir)
    print("PROBE", active, ("baseline", "hoist", "inplace")[active], flush=True)
    return ir

enflame_compiler.make_gcuir = rewrite_state

def run_with_arm(kernel, *params, **meta):
    if getattr(kernel, "__name__", "") == "_gqa_sparse_fwd_direct":
        meta["MAX_STATE_COLUMN"] = active
    return original(kernel, *params, **meta)


def capture(kernel, *params, **meta):
    result = run_with_arm(kernel, *params, **meta)
    if getattr(kernel, "__name__", "") == "_gqa_sparse_fwd_direct":
        index = active
        print("COMPILED", index, meta["grid"], result.metadata, flush=True)
        for key in ("gcuir", "llir", "fatbin"):
            value = result.asm[key]
            path = artifact / f"direct-{index}.{key}"
            if isinstance(value, bytes):
                path.write_bytes(value)
            else:
                path.write_text(value)
            if key == "llir":
                ops = Counter(re.findall(r"callee = @(tops_\w+)", value))
                print("LOWERING", index, json.dumps(ops, sort_keys=True), flush=True)
    return result

def call(arm):
    global active
    active = arm
    chain(outputs[arm])

def time_launch(kernel, *params, **meta):
    if getattr(kernel, "__name__", "") != "_gqa_sparse_fwd_direct":
        return original(kernel, *params, **meta)
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    result = run_with_arm(kernel, *params, **meta)
    b.record()
    direct_events.append((a,b))
    return result

with mode("packed"):
    triton.runtime.jit.JITFunction.run = capture
    try:
        for arm in (0,1,2):
            call(arm)
            torch.gcu.synchronize()
    finally:
        triton.runtime.jit.JITFunction.run = original
    ref = outputs[0].cpu()
    for arm in (1,2):
        got = outputs[arm].cpu()
        error = (got.float() - ref.float()).abs()
        print("VERIFY", arm, torch.equal(got, ref), error.max().item(), flush=True)
        assert torch.isfinite(got).all() and torch.equal(got, ref)
    for arm in (0,1):
        for _ in range(4):
            call(arm)
    torch.gcu.synchronize()
    timings = {arm: {"chain": [], "direct": []} for arm in (0,1,2)}
    direct_events = []
    triton.runtime.jit.JITFunction.run = time_launch
    try:
        for repeat in range(25):
            for arm in ((repeat + offset) % 3 for offset in range(3)):
                a = torch.gcu.Event(enable_timing=True)
                b = torch.gcu.Event(enable_timing=True)
                a.record()
                call(arm)
                b.record()
                b.synchronize()
                start,end = direct_events.pop()
                timings[arm]["chain"].append(a.elapsed_time(b))
                timings[arm]["direct"].append(start.elapsed_time(end))
    finally:
        triton.runtime.jit.JITFunction.run = original
    print("SAMPLES", json.dumps(timings), flush=True)
    print("MEDIANS", json.dumps({a:{k:median(v) for k,v in spans.items()} for a,spans in timings.items()}), flush=True)

    for ref, arm, field in ((0,1,"direct"),(0,2,"direct"),(1,2,"direct"),(1,2,"chain")):
        delta = [b - a for a, b in zip(timings[ref][field], timings[arm][field])]
        print("PAIRED", ref, arm, field, json.dumps({"median_ms": median(delta), "faster": sum(d < 0 for d in delta), "pairs": len(delta)}), flush=True)

    from public_check import b
    standard_args = b.MSABenchmarkArgs

    def seeded_args():
        args = standard_args()
        args.seed = extra_seed
        return args

    triton.runtime.jit.JITFunction.run = run_with_arm
    try:
        for extra_seed in (17, 73):
            b.MSABenchmarkArgs = seeded_args
            data, chain = make_case(WORKLOAD)
            outputs = [torch.empty_like(data.q) for _ in range(3)]
            for arm in (0,1,2):
                call(arm)
            torch.gcu.synchronize()
            ref = outputs[0].cpu()
            for arm in (1,2):
                got = outputs[arm].cpu()
                equal = torch.equal(got, ref)
                print("VERIFY", "seed_base", extra_seed, arm, equal, flush=True)
                assert torch.isfinite(got).all() and equal
    finally:
        b.MSABenchmarkArgs = standard_args
        triton.runtime.jit.JITFunction.run = original
