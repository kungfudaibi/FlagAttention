import importlib.util
import inspect
import json
import os
import re
import statistics
import sys
from pathlib import Path

from public_check import WORKLOAD, b, make_case, mode, torch, triton
from flag_attn.runtime.backend._enflame.msa.sparse_attn import _gqa_sparse_fwd_direct

HERE = Path(__file__).parent
BODY = """
    # Specialization for one request, 13 residual pages, D128 and GQA8.
    pair = 2 * tl.program_id(0)
    head = tl.program_id(1)
    req = tl.program_id(2)
    begin = tl.load(cu_seqlens_q + req)
    prefix = tl.load(prefix_lens + req)
    seq = tl.load(seq_lens + req)
    scale = sm_scale * 1.4426950409
    off_h = tl.arange(0, 8)
    off_d = tl.arange(0, 128)
    off_page = tl.arange(0, 16)
    ix0 = pair
    ix1 = pair + 1
    q0p = tl.make_block_ptr(q_ptr + (begin + ix0)*stride_qn + head*8*stride_qh,
                            (8,128), (stride_qh,stride_qd), (0,0), (8,128), (1,0))
    q1p = tl.make_block_ptr(q_ptr + (begin + ix1)*stride_qn + head*8*stride_qh,
                            (8,128), (stride_qh,stride_qd), (0,0), (8,128), (1,0))
    q0 = tl.load(q0p, boundary_check=(0,1), padding_option="zero")
    q1 = tl.load(q1p, boundary_check=(0,1), padding_option="zero")
    state0 = (begin+ix0)*STRIDE_SN + head*8 + off_h
    state1 = (begin+ix1)*STRIDE_SN + head*8 + off_h
    m0 = tl.load(maximum_ptr + state0)
    m1 = tl.load(maximum_ptr + state1)
    l0 = tl.where(off_d[None,:] == 0, tl.load(denom_ptr + state0)[:,None], 0.0)
    l1 = tl.where(off_d[None,:] == 0, tl.load(denom_ptr + state1)[:,None], 0.0)
    p0 = tl.make_block_ptr(partial_ptr + (begin+ix0)*STRIDE_PN + head*8*STRIDE_PH,
                           (8,128), (STRIDE_PH,STRIDE_PD), (0,0), (8,128), (1,0))
    p1 = tl.make_block_ptr(partial_ptr + (begin+ix1)*STRIDE_PN + head*8*STRIDE_PH,
                           (8,128), (STRIDE_PH,STRIDE_PD), (0,0), (8,128), (1,0))
    a0 = tl.load(p0, boundary_check=(0,1), padding_option="zero")
    a1 = tl.load(p1, boundary_check=(0,1), padding_option="zero")
    tab = block_table_ptr + req*stride_bt_b
    ids0 = t_ptr + head*stride_th + (begin+ix0)*stride_tn
    ids1 = t_ptr + head*stride_th + (begin+ix1)*stride_tn
    block0 = tl.load(ids0 + (3+off_page)*stride_tk, mask=off_page < 13, other=0)
    block1 = tl.load(ids1 + (3+off_page)*stride_tk, mask=off_page < 13, other=0)
    page0 = tl.load(tab + block0, mask=off_page < 13, other=0)
    page1 = tl.load(tab + block1, mask=off_page < 13, other=0)

    for j in range(13):
        select = off_page == j
        b0 = tl.sum(tl.where(select,block0,0),0)
        b1 = tl.sum(tl.where(select,block1,0),0)
        g0 = tl.sum(tl.where(select,page0,0),0)
        g1 = tl.sum(tl.where(select,page1,0),0)
        k0,v0 = _pair_load(packed_k_ptr,kv_cache_ptr,g0,head,num_kv_heads,
                           stride_kv_blk,stride_kv_h,stride_kv_pos,stride_kv_d)
        if MAX_STATE_COLUMN == 2:
            k1 = _pair_load_k(packed_k_ptr,g1,head,num_kv_heads)
        if MAX_STATE_COLUMN == 1:
            k1,v1 = _pair_load(packed_k_ptr,kv_cache_ptr,g1,head,num_kv_heads,
                               stride_kv_blk,stride_kv_h,stride_kv_pos,stride_kv_d)
        m0,l0,a0 = _pair_step(q0,k0,v0,b0,prefix+ix0,seq,scale,m0,l0,a0)
        if MAX_STATE_COLUMN == 2:
            v1 = _pair_load_v(kv_cache_ptr,g1,head,stride_kv_blk,stride_kv_h,stride_kv_pos,stride_kv_d)
        if MAX_STATE_COLUMN == 0:
            k1,v1 = _pair_load(packed_k_ptr,kv_cache_ptr,g1,head,num_kv_heads,
                               stride_kv_blk,stride_kv_h,stride_kv_pos,stride_kv_d)
        m1,l1,a1 = _pair_step(q1,k1,v1,b1,prefix+ix1,seq,scale,m1,l1,a1)
    _pair_store(o_ptr + (begin+ix0)*stride_on + head*8*stride_oh,l0,a0,stride_oh,stride_od)
    _pair_store(o_ptr + (begin+ix1)*stride_on + head*8*stride_oh,l1,a1,stride_oh,stride_od)
"""
HELPERS = """
import triton
import triton.language as tl

@triton.jit
def _pair_load(packed, cache, page, head, nh, KB:tl.constexpr, KH:tl.constexpr, KN:tl.constexpr, KD:tl.constexpr):
    kp = tl.make_block_ptr(packed + (page*nh+head)*16384,
                           (128,128),(128,1),(0,0),(128,128),(1,0))
    vp = tl.make_block_ptr(cache + page*KB + head*KH + 128*KD,
                           (128,128),(KN,KD),(0,0),(128,128),(1,0))
    return (tl.load(kp,boundary_check=(0,1),padding_option="zero"),
            tl.load(vp,boundary_check=(0,1),padding_option="zero"))

@triton.jit
def _pair_load_k(packed, page, head, nh):
    kp = tl.make_block_ptr(packed + (page*nh+head)*16384,
                           (128,128),(128,1),(0,0),(128,128),(1,0))
    return tl.load(kp,boundary_check=(0,1),padding_option="zero")

@triton.jit
def _pair_load_v(cache, page, head, KB:tl.constexpr, KH:tl.constexpr, KN:tl.constexpr, KD:tl.constexpr):
    vp = tl.make_block_ptr(cache + page*KB + head*KH + 128*KD,
                           (128,128),(KN,KD),(0,0),(128,128),(1,0))
    return tl.load(vp,boundary_check=(0,1),padding_option="zero")

@triton.jit
def _pair_step(q,k,v,blk,pos,seq,scale:tl.constexpr,m,l,a):
    dots = tl.dot(q,k)
    start = blk*128
    offs = start + tl.arange(0,128)
    qk = tl.zeros((8,128),tl.float32)
    if start + 128 > pos:
        qk += tl.where(pos >= offs[None,:],0.0,float("-inf"))
    qk += dots * scale
    if start + 128 > seq:
        qk += tl.where(offs[None,:] < seq,0.0,float("-inf"))
    maximum = tl.maximum(m,tl.max(qk,axis=1))
    p = tl.exp2(qk - maximum[:,None])
    alpha = tl.exp2(m-maximum)
    a = a * alpha[:,None]
    l = l * alpha[:,None] + p
    a += tl.dot(p.to(v.dtype),v)
    return maximum,l,a

@triton.jit
def _pair_store(out,l,a,SH:tl.constexpr,SD:tl.constexpr):
    denom = tl.sum(l,axis=1)
    acc = a * tl.where(denom > 0,1.0/denom,0.0)[:,None]
    ptr = tl.make_block_ptr(out,(8,128),(SH,SD),(0,0),(8,128),(1,0))
    tl.store(ptr,acc.to(out.dtype.element_ty),boundary_check=(0,1))

"""
original_src = inspect.getsource(_gqa_sparse_fwd_direct.fn)
header = original_src[:original_src.index("    sm_scale_log2e")].replace("_gqa_sparse_fwd_direct", "_paired_direct")
source_path = HERE / "paired_kernel.py"

# Matched single-query control uses the same 13-page specialization and helpers.
single = BODY.replace("pair = 2 * tl.program_id(0)", "pair = tl.program_id(0)")
single = re.sub(r"^    q1p =.*\n[ ]+\(8,128\).*\n", "", single, flags=re.M)
single = re.sub(r"^    (?:ix1|q1|state1|m1|l1|a1|block1|page1|ids1) = .*\n", "", single, flags=re.M)
single = re.sub(r"^    p1 =.*\n[ ]+\(8,128\).*\n", "", single, flags=re.M)
single = re.sub(r"^        (?:b1|g1) = .*\n", "", single, flags=re.M)
single = re.sub(r"^        if MAX_STATE_COLUMN.*\n(?:[ ]{12,}.*\n)+", "", single, flags=re.M)
single = re.sub(r"^        m1,l1,a1 = .*\n|^    _pair_store.*ix1.*\n", "", single, flags=re.M)
assert not any(s in single for s in ("ix1","g1,","block1","q1p","MAX_STATE_COLUMN"))
source_path.write_text(HELPERS+header+BODY+header.replace("_paired_direct","_single_direct")+single)

spec = importlib.util.spec_from_file_location("msa_paired",source_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

ORIG = triton.runtime.jit.JITFunction.run
NAMES = ("baseline", "single_fixed", "paired_serial", "paired_k", "paired_interleaved")
variant = "baseline"
capture = True
measure = False
events = []
counts = None
first_shape = True

def dispatch(jit, *args, **kwargs):
    global counts
    name = jit.__name__
    if capture and name == "_gqa_sparse_common_prefix":
        counts = args[10]
    if name != "_gqa_sparse_fwd_direct":
        return ORIG(jit,*args,**kwargs)
    if variant != "baseline":
        assert kwargs["grid"] == (1024,1,1)
        assert kwargs["USE_PACKED_K"] and kwargs["PRELOAD_PAGE_META"] and kwargs["USE_SHARED_PREFIX"]
        if capture:
            assert tuple(counts.shape) == (1,1,64)
            assert bool((counts.cpu() == 3).all())
        if variant == "single_fixed":
            jit = mod._single_direct
        else:
            kwargs["grid"] = (512,1,1)
            kwargs["MAX_STATE_COLUMN"] = {"paired_serial":0,"paired_interleaved":1,"paired_k":2}[variant]
            jit = mod._paired_direct
    if measure:
        start, end = (torch.gcu.Event(enable_timing=True) for _ in range(2))
        start.record()
    result = ORIG(jit,*args,**kwargs)
    if measure:
        end.record()
        events.append((start,end))
    if capture:
        print("COMPILED", variant, kwargs["grid"], result.metadata, flush=True)
        for key in ("gcuir","llir","fatbin"):
            value = result.asm[key]
            p = HERE/f"{variant}.{key}"
            if isinstance(value,bytes): p.write_bytes(value)
            else: p.write_text(value)
    return result

os.environ["FLAG_ATTN_MSA_SCORE_PAGE_MAJOR"] = "1"
os.environ["FLAG_ATTN_MSA_PRELOAD_PAGE_META"] = "1"
data, chain = make_case(WORKLOAD)
outputs = {name:torch.empty_like(data.q) for name in NAMES}
samples = {name:{"direct":[],"chain":[]} for name in NAMES}
triton.runtime.jit.JITFunction.run = dispatch
try:
    with mode("packed"):
        for variant in NAMES:
            chain(outputs[variant])
            torch.gcu.synchronize()
        ref = outputs["baseline"].cpu()
        for variant in NAMES[1:]:
            got = outputs[variant].cpu()
            err=(ref.float()-got.float()).abs()
            equal=torch.equal(ref,got)
            print("VERIFY",variant,equal,err.max().item(),int(torch.count_nonzero(err)),flush=True)
            assert torch.isfinite(got).all() and equal
        capture = False
        for variant in NAMES:
            for i in range(4):
                chain(outputs[variant])
        torch.gcu.synchronize()
        measure = True
        for i in range(20):
            for variant in [NAMES[(i+j)%len(NAMES)] for j in range(len(NAMES))]:
                a,z=(torch.gcu.Event(enable_timing=True) for _ in range(2))
                a.record()
                chain(outputs[variant])
                z.record()
                z.synchronize()
                x,y=events.pop()
                samples[variant]["direct"].append(x.elapsed_time(y))
                samples[variant]["chain"].append(a.elapsed_time(z))
finally:
    triton.runtime.jit.JITFunction.run = ORIG
print("SAMPLES",json.dumps(samples),flush=True)
print("MEDIANS",json.dumps({n:{k:statistics.median(v) for k,v in s.items()} for n,s in samples.items()}),flush=True)
for stage in ("direct","chain"):
    for before,after in (("baseline","single_fixed"),("single_fixed","paired_serial"),("paired_serial","paired_k"),("paired_serial","paired_interleaved")):
        diffs=[b-a for a,b in zip(samples[before][stage],samples[after][stage])]
        print("PAIRED",stage,before,after,statistics.median(diffs),sum(x<0 for x in diffs),flush=True)
