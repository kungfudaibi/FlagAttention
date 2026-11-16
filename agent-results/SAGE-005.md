# SAGE-005: why the device-native fused SDPA outruns SageAttention on GCU S60

**Scope.** Investigation only. One clone, one branch, one device. No routing
change, no push, no merge.

> **Revised by [SAGE-006](SAGE-006.md).** The review of this report found that
> several numbers were quoted without committed raw data, that two performance
> claims had the wrong sign or the wrong sample set, and that §11 reported a
> local pre-check as if it were the orchestrator's formal gate. SAGE-006
> recovered the `/tmp` scripts and transcripts, re-ran the short probes into
> `raw/sage006_recheck_*.jsonl`, and corrected the text below in place.
> Each correction is marked **"Corrected in SAGE-006"**. The qualitative
> conclusions did not change: the 3.3-3.9x gap is real, it reproduces without
> INT8, and both candidate optimisations are still rejected. The formal gate
> **passed** (42 passed, exit 0).

| item | value |
| --- | --- |
| clone | `/workspace/agent-workspaces/sage-fused-sdpa-study-5` |
| base commit | `087a3621bbfeb843d9efe2445b67faa7f1ed46ec` |
| branch | `agent/enflame-sage-fused-sdpa-study-5` |
| device | GCU 5 only, PCI `0000:83:00.0`, SN `A0D5460520864`, 42976 MiB |
| host CPUs | `taskset -c 160-191` |
| stack | torch 2.10.0+cpu, torch_gcu 2.10.0+3.7.20260408, Triton 3.6.0 (`/opt/flagtree`), Python 3.12.13 |
| device props | `major=3 minor=0`, `multi_processor_count=2`, `sip_count=24`, arch `gcu300` |

GCU 6 (`84:00.0`) and GCU 7 (`a3:00.0`) were never touched. Every script
resolves the assigned ordinal through `efsmi` and aborts unless the PCI address
*and* serial match, because ordinals are container-local.

**Machine was not quiet.** Load average stayed 26-36 with a foreign tenant on
CPU 176 and `clang++`/`llc` running. Every number below is a diagnostic from
interleaved same-process runs, not a formal performance claim. The
qualitative ordering is stable across all repeats and both shapes.

---

## 1. The reproduction is real, and it is not the host fallback

This was the first thing to rule out, because the repository's
`F.scaled_dot_product_attention` is *not* the device op.

Measured with `sage005_dispatch.py` (one call profiled, retried on empty census;
residency also proved with a GCU event pair, which cannot elapse without device
work):

| path | shape | status | device-event ms | host copies |
| --- | --- | --- | ---: | ---: |
| `aten::_scaled_dot_product_flash_attention` | 1x2x4096x64 | device_resident | 0.2745 | 0 |
| `F.scaled_dot_product_attention` | 1x2x4096x64 | **host_fallback** | 24.3287 | 9 |
| Sage full call | 1x2x4096x64 | device_resident | 1.1416 | 0 |
| Sage attention only | 1x2x4096x64 | device_resident | 1.0389 | 0 |
| `aten::_scaled_dot_product_flash_attention` | 4x32x1024x128 | device_resident | 1.1767 | 0 |
| `F.scaled_dot_product_attention` | 4x32x1024x128 | **host_fallback** | 171.7145 | 9 |
| Sage full call | 4x32x1024x128 | device_resident | 5.3281 | 0 |
| Sage attention only | 4x32x1024x128 | device_resident | 3.8993 | 0 |

Observed facts:

- The device op is a single `GCU::_scaled_dot_product_flash_attention` with zero
  host copies.
- The repository `F.sdpa` runs
  `GCU::wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention`
  then `aten::_to_cpu` then `aten::_scaled_dot_product_flash_attention_for_cpu`:
  it copies all three operands to the host and computes on the CPU. On these
  shapes that costs 24-172 ms, ~90-145x the device op. **It must never be used
  as the device baseline**, and no number in this report does.
- The Sage full call is device-resident and equals the attention-only kernel
  plus 0.10 ms (D64) / 1.43 ms (D128) of INT8 quantization.
- The Kineto GCU activity stream is unreliable on this image: identical repeated
  profiles either return the full runtime list or only two `aten::empty`
  allocations. Launch counts from that stream are therefore not used as
  evidence anywhere in this report; the CPU op chain and GCU event timings are.

Reproduced gap (median of 20 event-timed calls, interleaved):

| shape | native | Sage full | ratio |
| --- | ---: | ---: | ---: |
| 1x2x4096x64 | 0.2703 ms (31.6 TF) | 1.0431 ms (8.2 TF) | **3.86x** |
| 4x32x1024x128 | 1.1696 ms (58.2 TF) | 3.9062 ms (17.6 TF) | **3.34x** |

The formal alternating full-call A/B (`sage004_ab.py`, 3 rounds x 24
alternating pairs, `status: PASS`, exit 0) reproduces the same ordering against
the legacy control:

| shape | candidate | legacy | candidate/legacy | rounds faster |
| --- | ---: | ---: | ---: | ---: |
| 1x2x4096x64 | 1.1478 ms | 1.5809 ms | 0.7235 | 3/3 |
| 4x32x1024x128 | 5.3441 ms | 12.0691 ms | 0.4428 | 3/3 |

## 2. The gap survives deleting INT8, quantization and the scale math

This is the result that reframes the whole question. A hand-written pure-FP16
Triton flash-attention kernel — same tile geometry, same `num_warps=1`,
`num_stages=1`, `exp2` with `1.4426950408889634` scaling, same `acc/l_i`
epilogue, **no INT8, no quantizer, no per-block scales** — reproduces the gap:

| shape | pure-FP16 Triton | native | ratio | max abs vs native |
| --- | ---: | ---: | ---: | ---: |
| 1x2x4096x64 (BM=256, grid=32) | 1.0137 ms (8.47 TF) | 0.2640 ms | **3.84x** | 1.22e-04 |
| 4x32x1024x128 (BM=128, grid=1024) | 4.2598 ms (16.13 TF) | 1.1742 ms | **3.63x** | 9.77e-04 |

So the deficit is **not** Sage's INT8 pipeline, **not** the quantizer, and
**not** the dequantize/scale arithmetic. It is a Triton-versus-vendor
code-generation gap on identical FP16 QK/PV math.

**Evidence status (corrected in SAGE-006).** The script that produced this
table was never written to a JSONL; the numbers above come from the session's
stdout, recovered verbatim into
`raw/sage005_recovered/fp16attn.log`. The per-sample records do not exist and
are not claimed. SAGE-006 re-ran the same probe and committed the raw data
(`raw/sage006_recheck_fp16.jsonl`, exit 0): 1.0124 ms vs 0.2705 ms = **3.74x**
and 4.2508 ms vs 1.1730 ms = **3.62x**, within 0.3 % of the values quoted here.
The conclusion is unchanged; only the provenance is now honest.

SAGE-004's "online-softmax bookkeeping" diagnosis was a correct measurement of
where the *time* goes, but not the cause of the *gap*.

## 3. Triton can saturate this device, but not with a per-iteration tile feed

`tl.dot` throughput with both operands resident in shared memory and no
per-iteration loads — the compute-only ceiling:

| tile | nCTA=1 | nCTA=24 | nCTA=96 |
| --- | ---: | ---: | ---: |
| 256x128x64 | 3.03 TF | **67.78 TF** | 77.86 TF |
| 128x128x128 | 3.80 TF | **82.98 TF** | 97.63 TF |

The same dots with a per-iteration tile load in the layout the Sage kernel uses:

| tile | nCTA=1 | nCTA=24 | nCTA=96 |
| --- | ---: | ---: | ---: |
| 256x128x64 (qk+pv) | 1.38 TF | **28.31 TF** | 40.02 TF |
| 128x128x128 (qk+pv) | 1.40 TF | **28.73 TF** | 39.33 TF |

Adding nothing but a streaming tile load costs **2.4-2.9x** of the achievable
matmul rate at the occupancy that matters (24 CTAs = 1 CTA per SIP). Repeated
runs of these two probes agree within ~1 % (e.g. 67.78-68.00 TF resident at
24 CTAs), so the ordering is not a single-run artefact. That is
the mechanism: with `num_stages=1` the load cannot overlap the dot, so each
iteration serialises `global load -> matmul -> row reduce -> matmul`.

This is not a tuning miss. A `num_stages` sweep 1-4 on the tile-major K loop
never improves it (28.68 / 28.18 / 24.64 / 24.06 TF at 24 CTAs, D128 — stages
1-2 are a wash and stages 3-4 degrade), and manual double-buffered K prefetch
in the loop microbenchmark is **slower**, not faster: 187.3 / 188.1 / 186.9 /
189.0 us against 176.4 / 174.7 / 177.6 / 174.6 us plain, i.e. **+6-8 %**.
(An earlier draft of this report had the sign of that comparison backwards; it
has been corrected against the recovered stdout and a SAGE-006 re-run.) The
compiled GCUIR contains **no `gcu.dma` and no `memref.copy`**, 51 `memref.alloc`
/ 42 `memref.dealloc`, and **4 `gcu.mfence` around 4 `gcu.matmul`** for a single
tile loop — i.e. every matmul goes through a freshly allocated buffer bracketed
by fences, with no cross-iteration overlap.

The LLIR *does* contain async copies — 24 `tops_*_async` call sites, 7 of them
`tops_memcpy_async` — so "no async copy at all" would be wrong. But each is
followed by a `tops_wait_dte` after only 1-12 LLVM operations: 19 sites wait
after 1 op, 2 after 2, 1 after 9, 2 after 12. The pipeline is opened and closed
inside the same basic block, so the transfer is synchronous in effect. "No software
pipeline for this loop on `gcu300`" therefore stands as an **inference** from
the IR shape plus the two negative timing results, not as a backend-source
proof.

## 4. Where the per-iteration time actually goes

Because query length and iteration count move together, a segment slope
conflates CTA count with loop depth. Measured two ways.

**(a) Single-wave marginal cost**, one call per point, `nCTA <= 24` so no second
wave quantisation, per additional 128-wide KV tile:

| D | path | SAGE-005 (session stdout) | SAGE-006 re-run (raw JSONL) | ratio |
| --- | --- | --- | --- | --- |
| 64 | native | 8.02 (7.12 .. 8.88) | 8.26 (7.61 .. 9.00) | — |
| 64 | Sage | **15.21** (14.88 .. 15.47) | **14.91** (14.25 .. 15.25) | **1.81x** |
| 128 | native | 10.49 (9.31 .. 12.33) | 9.84 (9.33 .. 10.73) | — |
| 128 | Sage | 15.38 (10.60 .. 24.10) | **11.06** (10.72 .. 11.36) | **1.12x** |

**Corrected in SAGE-006.** The D128 figure of 15.38 us is wrong. The original
`/tmp/sage005_marg2.py` swept to `T=3584/4096`, where `nCTA` is 28/32 — already
into a second wave — and those two points (24.63 and 24.10 us) dominate the
mean. Restricting the sweep to `nCTA <= 24`, as the D64 sweep already did, gives
11.06 us/tile, i.e. **1.12x** the native slope, not 1.5-1.9x. D64 is unaffected
(1.81x) and *is* flat at ~15 us/tile regardless of depth. The claim that the
Sage slope is "essentially flat at ~15 us regardless of depth" therefore holds
for D64 only.

**(b) Per-variant slope with launch, prologue and epilogue differenced out**
(`sage005_micro.py --probe slope`; each ablation rewrites exactly one source
fragment of the shipped kernel, and a companion record proves the dot ops
survived — 10 `gcu.matmul` for every variant except `no_pv_dot` (9) and
`qk_matmul_only` (6)):

| variant | us per KV iteration | vs full | gcu.matmul |
| --- | ---: | ---: | ---: |
| full | **14.91** | 1.00 | 10 |
| no_row_max | 10.87 | 0.73 | 10 |
| no_sub (max kept, broadcast subtract removed) | 10.53 | 0.71 | 10 |
| both_dots (all softmax bookkeeping removed) | **9.03** | 0.61 | 10 |
| no_pv_dot | 13.37 | 0.90 | 9 |
| qk_matmul_only | 0.16 | 0.01 | 6 |

The `gcu.matmul` column is `sage005_micro.py`'s own discriminator: it counts
occurrences of the string `gcu.matmul` in the compiled GCUIR, which also matches
the `triton_gcu.matmul` border attributes. It is therefore **2.5x the op-level
count** (10 here vs 4 `"gcu.matmul"` ops in the committed GCUIR) but is used
only to prove the ablation did not delete the dots, and it discriminates
correctly: full/no_row_max/no_sub/both_dots all keep 10, `no_pv_dot` drops to 9,
`qk_matmul_only` to 6. SAGE-006 left the probe's semantics unchanged and
recorded the distinction rather than silently rescaling the column.

Observed facts:

- Softmax bookkeeping is **39 %** of the steady-state per-iteration cost
  (14.91 -> 9.03 us).
- Almost all of that is the **row-max broadcast subtract**: dropping the
  subtract alone (`no_sub`, max retained) recovers as much as dropping the max
  entirely (10.53 vs 10.87 us). Standalone, a 256x128 FP32 tile pass costs
  1.53 us for the reduce but **5.91 us** for reduce + broadcast subtract, at
  `num_warps=1`; `num_warps=2/4` does not help (5.90 / 9.10 us) because extra
  warps serialise the dot path (warps 1/2/4/8 -> 18.24 / 9.71 / 5.56 / 1.95 TF).
- `qk_matmul_only` is 0.16 us/iteration because `acc` no longer depends on
  `qk`; the compiler deletes the QK loop entirely. That variant measures
  nothing useful and the SAGE-004 "QK-only 72 TFLOPS" figure must not be read as
  a QK cost.
- **The `both_dots` floor is not Sage overhead.** Per CTA per iteration the loop
  is 8.39 MFLOP for D64. The tile-fed Triton ceiling at 24 CTAs is 28.31 TF,
  i.e. 1.18 TF/CTA, i.e. **7.11 us/iteration**. The `both_dots` ablation
  measures 9.03 us — 1.27x of what Triton can do for that loop shape at all.
  SAGE-004's 2.36-2.48x "structural floor" is therefore Triton's non-pipelined
  load->dot, not something Sage adds.

## 5. Memory traffic, occupancy and wave quantisation

Analytic model of the shipped kernel (per CTA: one Q tile, one O tile, and per
iteration a pre-dequantized FP16 K tile plus an FP16 V tile):

| shape | nCTA | waves | iters | GFLOP | MiB | FLOP/byte | achieved | % of tile-fed ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x2x4096x64 | 32 | 1.33 | 32 | 8.59 | 82.0 | 99.9 | 8.24 TF | 22 % |
| 4x32x1024x128 | 1024 | 42.7 | 8 | 68.72 | 576.0 | 113.8 | 17.59 TF | 45 % |

Observed facts:

- The kernel is **not** DRAM-bound. 82.0 MiB in 1.0431 ms is ~82 GB/s (D128:
  576.0 MiB in 3.9062 ms, ~155 GB/s), and Triton
  streaming reads reach 450-459 GB/s and copies 425 GB/s on this device
  (16 KiB blocks; 2 KiB blocks only reach 103-116 GB/s, which is why tile load
  width matters). The limit is the serialised per-CTA load->dot dependency, not
  bandwidth.
- **Wave quantisation is real and large.** At fixed per-CTA work, only whole
  waves help: 16 CTAs (0.67 wave) 0.565 ms; 32 CTAs (1.33 waves) 1.044 ms;
  48 CTAs (2.00 waves) 1.043 ms; 96 CTAs (4.00 waves) 2.001 ms. A partial second
  wave nearly doubles the time (1.33 waves costs the same as 2 waves). This
  explains the shape dependence of the headline ratio (2.12x at 0.67 wave,
  3.90x at 1.33 waves) but is a tile-geometry problem, which the task
  explicitly excluded from a warp/block sweep.
- Peak device memory, two separate measurements that must not be merged:
  - Whole public path with operands resident (formal A/B harness): Sage
    **11.00 vs 10.03 MiB** (D64) and **352.04 vs 320.52 MiB** (D128) for native.
  - *Marginal* allocation of the attention call alone, counters reset after
    Q/K/V exist: Sage kernel **1.00 vs 1.03 MiB** and **32.0 vs 32.5 MiB**, i.e.
    the kernel itself allocates no more than the vendor op.
  The whole-call difference is therefore the INT8 quantization temporaries
  (Q_int8, K_int8, two scale tensors), not the kernel.
- Cold compile with an empty `TRITON_CACHE_DIR` in a child process:
  Sage **8862 ms / 7940 ms** vs native **5.5 ms / 8.5 ms**. The native op ships
  a precompiled SASS-equivalent in `libkernel_flash_attn_fwd.so`; Sage lowers
  through Triton on first call. This is a real product cost for short jobs but
  is not part of the steady-state gap.
- The vendor INT8/FP8 Sage kernel is **unusable on this device**:
  `/lib/libkernel_sage_attention.so` contains 26 `gcu400` strings and zero
  `gcu300`, while `libkernel_flash_attn_fwd.so` is `gcu300`. There is no
  vendor Sage kernel for S60 to compare against.

## 6. What the vendor kernel does differently (artifact inspection, label: inference)

Disassembling `flash_attn_fwd_kernel<__bf16,256>` from
`libkernel_flash_attn_fwd.so` (73,600-byte symbol, 18,861 instruction slots)
via `llvm-objcopy` -> `clang-offload-bundler --unbundle` -> `llvm-objdump`:

- **47** distinct opcodes (not 173 — see below), dominated by **padding**:
  `m.nop` 4593, `v.nop` 4470, `c.nop` 3830, `s.nop16` 837 — 13,730 of 18,861
  slots are scheduling nops, i.e. a statically scheduled, fully pipelined SIMT
  program with bubbles already filled by the compiler, plus scalar loads/stores
  (`l.st.w` 1136, `l.ld.w` 725). 17,364 of the 18,861 address lines decode;
  1,497 do not (data interleaved in the text section). The other three
  instantiations behave the same way: `bf16_128` 18,920 / 17,422 / 1,498 with
  46 opcodes, `fp16_256` 17,722 / 16,483 / 1,239 with 43, `fp16_128`
  17,554 / 16,330 / 1,224 with 43.
- Triton's output instead emits 4 `gcu.matmul` intrinsics into freshly
  `memref.alloc`'d `memref<256x128xf32>` / `memref<256x64xf32>` buffers, with 4
  `gcu.mfence`, and broadcasts via `memref_ext.broadcast_start`.

**Corrected in SAGE-006.** The "173 distinct opcodes" and "10 `gcu.matmul` /
8 `gcu.mfence`" figures were produced by counting substrings rather than
operations. The opcode regex took the first token after the encoded bytes, so
lines without a tab contributed operand hex (`44`, `18`, `00`) as if they were
mnemonics; taking the token after the tab gives 47. Likewise `10`/`8` count the
*m entions* of `gcu.matmul`/`gcu.mfence`, which also appear in the
`triton_gcu.matmul` border attributes and the `gcu.mfence_type` attribute; the
op-level counts are **4 and 4**. Both are recomputed from the committed
artefacts by `scripts/sage006_artifacts.py` (`raw/sage006_artifacts.jsonl`).

Inference (not observable, so labelled): the vendor kernel overlaps tile loads
with matmul via explicit static double buffering encoded in those nops, whereas
Triton on `gcu300` emits a fence-synchronised, allocation-per-op sequence whose
async copies are waited on immediately and therefore cannot overlap. This is
consistent with every measurement above (no `num_stages` effect, manual
prefetch 6-8 % *slower*, load+dot = 2.4-3.0x cost, async copies only 1-12 ops
ahead of their wait) but the vendor source is not available, so the scheduling
claim itself is inferred.

## 7. Experiment decision: no change shipped (negative result)

The task allowed exactly one opt-in experiment if a concrete Sage change was
plausible. I tested the two candidates that the evidence actually supported, and
both are dead:

1. **Fuse/eliminate the row-max broadcast subtract** — the single largest
   softmax term (5.94 us/pass vs 1.54 us for the reduce alone; 4.3 us/iteration
   of the 14.91 total). Two rewrites of the shipped line
   (`p32 = qk - m_ij[:,None]` reused, and in-place `qk -= m_ij[:,None]`):
   base **1.0316 ms**, fused_expr **1.0340 ms**, inplace_sub **1.0292 ms** at
   D64. Within noise. The compiler already emits one broadcast; there is no
   duplicated subtract to remove. SAGE-006 re-ran this and added the check the
   original run crashed before printing: both rewrites are **bitwise identical**
   to base (maxabs 0.0, D64 and D128), confirming there is nothing to fuse.
   **Rejected.**
2. **Feed INT8 K and dequantize in-kernel** to halve K bytes per iteration,
   instead of the shipped `predequantize_k=True` FP16 temp. Swept
   `predequantize_k` x `tile_major_predequant_k` x `qk_scale_before_dot`.

   **Corrected in SAGE-006.** The earlier "< 1.5 % spread, D64 1.0274-1.0431 ms,
   D128 3.9031-3.9062 ms" quoted a range that silently excluded every
   `qk_scale_before_dot=False` row. The full sweep (re-run, committed as
   `raw/sage006_recheck_flags.jsonl`) is:

   | shape | `qk_scale_before_dot=True` | `=False` | cost | max output diff |
   | --- | ---: | ---: | ---: | ---: |
   | D64 | 1.0248-1.0335 ms | 1.3461-1.3499 ms | **+31.4 %** | 3.66e-04 |
   | D128 | 3.8880-3.8924 ms | 4.0986-4.1011 ms | **+5.4 %** | 9.77e-04 |

   The two K-layout switches (`predequantize_k`, `tile_major_predequant_k`)
   really are free: with `qk_scale_before_dot=True`, all four of their
   combinations land within 0.85 % at D64 and 0.11 % at D128. But
   `qk_scale_before_dot` is **not** a free knob: turning it off costs 31 % at
   D64, 5.4 % at D128, and changes the numerics. The rejection still stands —
   no default should move — but the "< 1.5 % spread" justification does not.
   Since the kernel is not bandwidth-bound (82 GB/s of 450 available), halving K
   bytes buys nothing. **Rejected.**

Because neither candidate improves on the shipped path (one is bit-identical,
the other only gets slower), and because the gap is reproduced by
a quantization-free FP16 kernel, no routing change and no new default is
justified. The original route stays the control. **This is a negative result:**
Sage's fused SDPA on gcu300 is 3.3-3.9x slower than the vendor op, and no
opt-in change within the Triton kernel closes it, because the loss is in code
generation, not in the operator.

## 8. Numerical checks (unchanged path, for the record)

The shipped path was verified against the device op over 6 shapes including
partial blocks, in both output dtypes:

| shape | out | max abs | rel L2 | cosine | finite |
| --- | --- | ---: | ---: | ---: | --- |
| 1x2x4096x64 | fp16 | 2.20e-03 | 1.24e-02 | 0.999918 | yes |
| 1x2x4096x64 | bf16 | 2.09e-03 | 1.25e-02 | 0.999898 | yes |
| 4x32x1024x128 | fp16 | 1.07e-02 | 1.22e-02 | 1.001566 | yes |
| 4x32x1024x128 | bf16 | 1.22e-02 | 1.23e-02 | 1.001345 | yes |
| 1x2x4090x64 (partial M) | fp16 | 2.17e-03 | 1.24e-02 | 0.999915 | yes |
| 1x2x4090x64 (partial M) | bf16 | 2.20e-03 | 1.25e-02 | 0.999898 | yes |
| 1x2x257x64 (partial N, 3 CTAs) | fp16 | 6.84e-03 | 1.19e-02 | 0.999927 | yes |
| 1x2x257x64 (partial N, 3 CTAs) | bf16 | 7.02e-03 | 1.20e-02 | 0.999928 | yes |
| 2x3x1000x128 (partial both) | fp16 | 4.94e-03 | 1.21e-02 | 0.999920 | yes |
| 2x3x1000x128 (partial both) | bf16 | 4.88e-03 | 1.22e-02 | 0.999888 | yes |
| 1x1x129x128 (minimal, masked) | fp16 | 6.84e-03 | 1.10e-02 | 0.999939 | yes |
| 1x1x129x128 (minimal, masked) | bf16 | 7.32e-03 | 1.11e-02 | 0.999940 | yes |

Relative L2 is ~1.2e-02 in every case, which is the INT8 per-block quantization
error floor, identical to SAGE-002/003/004 and independent of shape, partial
blocks and output dtype. No NaNs, no infinities, no shape-dependent blow-up.

The two D128 rows report a cosine of 1.0016 / 1.0013. Cosine similarity cannot
exceed 1; this is a normalisation artefact of `F.cosine_similarity` on flattened
tensors of 16.7 M elements in float32 (the dot product of `a` with `b` slightly
exceeds `|a||b|` after rounding), not a gate failure. The metric that matters,
relative L2 ~1.2e-02 and max abs ~1.1e-02, is in line with every other row. The
per-shape records are committed as `raw/sage005_recovered/sage005_acc.json`;
the generating script `sage005_acc.py` is in `scripts/recovered/`.

## 9. Ranked next steps

1. **Fix the Triton `gcu300` pipeline, not the kernel.** The largest and only
   well-supported win: teach the backend to emit async copy / double buffering
   so a tile load can overlap a `gcu.matmul`. Evidence: resident dots hit
   68-83 TF but a per-iteration feed drops them to 28 TF, `num_stages` does
   nothing, manual prefetch is 6-8 % slower, GCUIR has no `gcu.dma`/`memref.copy`,
   and the 24 LLIR async copies are all waited on within 1-12 operations.
   Expected ceiling if fixed: the `both_dots` floor and possibly the full kernel
   move toward the 28.3 TF tile-fed rate, i.e. the current 3.3-3.9x could
   plausibly halve. **That halving is an extrapolation, not a measurement.**
   This is upstream work, outside this operator, and should be raised as a
   backend issue rather than smuggled into Sage.
2. **Reproduce the vendor scheduling by hand for one shape.** A hand-written
   SIMT kernel avoiding `gcu.matmul`/`memref.alloc` in favour of statically
   scheduled matrix ops with explicit double buffering, benchmarked against
   `libkernel_flash_attn_fwd.so` on D64 only. High effort, and the disassembly
   shows why it is hard (13,700 scheduling nops encode the vendor's pipeline).
3. **Wave-quantisation-aware persistent CTAs.** A partial second wave nearly
   doubles latency (1.33 waves == 2 waves). A persistent-CTA scheduler that
   splits work into whole waves would help the *ratio* on shapes like
   1x2x4096x64 (32 CTAs) without touching kernel math. This was out of scope
   here (tile/wave geometry sweep) but is cheap and shape-specific.
4. **Drop the row-max broadcast subtract via a different softmax
   parameterisation** (e.g. keep `m_i` in the exponent and scale `acc` only).
   Worth ~4.3 us/iteration in isolation but needs a genuinely different
   algorithm rather than a source-level rewrite; the two source-level rewrites
   tried here were bit-identical to base and therefore cannot help.
5. **Do not pursue**: INT8 K feed (not bandwidth-bound; the K-layout switches
   are within 0.85 %, `qk_scale_before_dot=False` is 5-31 % *slower*), extra
   warps (monotonically worse: 18.24 -> 1.95 TF for warps 1 -> 8), extra stages
   (< 2 % effect), and the vendor Sage kernel (no `gcu300` build exists).

## 10. Reproduction

All scripts pin GCU 5 by PCI address and serial before the first device action
and abort otherwise. See `agent-results/scripts/README.md`.

SAGE-006 added `agent-results/scripts/sage005_recheck.py`, which re-runs the
probes whose original output was never written to JSONL, and
`sage006_artifacts.py`, which recomputes every IR and disassembly count from the
committed artefacts. The verbatim SAGE-005 session stdout and the original
`/tmp` scripts are under `raw/sage005_recovered/` and `scripts/recovered/`.
The `/tmp` JSONL never existed for those probes; the report no longer implies
that it did.

```
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_micro.py --probe resident,tilemajor --jsonl ...
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_micro.py --probe slope --jsonl ...
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_micro.py --probe epilogue --jsonl ...
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_dispatch.py --jsonl ...
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_cold_memory.py --jsonl ...
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage004_ab.py --shapes 1x2x4096x64 4x32x1024x128 \
    --output-dtype float16 --rounds 3 --pairs 24 --warmup 5 --min-samples 40 \
    --max-used-mib 40000 --jsonl ... --summary ...
```

Raw evidence: `agent-results/raw/sage005_*.jsonl` with sibling `.exit` files
(all 0). Peak memory and cold compile in `sage005_cold_memory.jsonl`; formal A/B
in `sage005_ab_ab.jsonl` + `sage005_ab_ab_summary.json` (`status: PASS`).
SAGE-006 re-runs: `raw/sage006_recheck_{fp16,fuse,flags,stages,waves,bw_marginal}.jsonl`
and `raw/sage006_artifacts.jsonl`, all with `.exit` 0. Recovered SAGE-005 session
stdout: `raw/sage005_recovered/*.log` (provenance level C — verbatim stdout, not
per-sample JSONL).

## 11. Formal accuracy gate

**This section was wrong in the original report and is corrected here.**

SAGE-005's formal check is executed by the orchestrator after the session, and
it **passed**:

| item | value |
| --- | --- |
| check id | `20261116T054641Z-38b71c5c` |
| kind / name | `correctness` / `accuracy` |
| commit | `167c360085ffe46371626eeb108d17ea66fb92bf` |
| device / pci / cpu | `5` / `83:00.0` / `160-191` |
| exit code | **0** |
| stdout tail | `42 passed, 1 warning in 55.69s` |
| evidence | `raw/sage006_formal_gate_record.json`, `raw/sage006_formal_gate_stdout.log` |

The `command` array in `record.json` holds only the `timeout ...` line below.
The `cd` and the `export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"` are the working
directory and environment the orchestrator sets on the child
(`orchestrator/core.py:405`), so they are not in the array. The record also
carries baseline `087a3621bbfeb843d9efe2445b67faa7f1ed46ec` and branch
`agent/enflame-sage-fused-sdpa-study-5`.

```
cd <workspace> && export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH" && \
  timeout --signal=TERM --kill-after=10s 1800s taskset -c 160-191 \
  /flagos/bin/python -c "import os, torch, torch_gcu, pytest; \
  os.environ['SAGE002_TEST_DEVICE']='gcu:5'; torch.gcu.set_device(5); \
  raise SystemExit(pytest.main(['tests/flag_attn/test_sage_attention.py', \
  'tests/flag_attn/test_sage_bottleneck.py','-q']))"
```

Three corrections against the original text:

1. The "exit 1, 40 passed, 2 failed" result was a **local manual pre-check**
   (`raw/sage005_formal_gate.txt`), not the orchestrator's formal check. Its
   command line omitted `PYTHONPATH`, so the two subprocess-spawning tests could
   not import `flag_attn`. It was never the acceptance result. That file is kept
   for audit and now labelled *superseded*.
2. The conclusion drawn from it — "the orchestrator sets no `PYTHONPATH`" — is
   false: `orchestrator/core.py:405` exports `PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"`
   before every check command. The formal run above therefore had the package on
   its path and passed.
3. The original text said the check ran on commit `6b2487b`. The formal record
   says `167c360`, which is this report's own commit.

SAGE-005 changed no file under `src/` or `tests/`; `git diff --name-only
087a362 HEAD` lists only `agent-results/*`.

**This report is still not acceptance.** Acceptance is the orchestrator's formal
check, and on SAGE-005 that check already passed with 42 passed / exit 0.
