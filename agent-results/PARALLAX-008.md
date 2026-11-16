# PARALLAX-008: query-row tile widening (BT) for dense D64

Branch `agent/enflame-parallax-kernel-profile-1`, baseline
`9eca4bbb856a6a323ed651864adc672e958e1ca1`. All measurements on **GCU 1**
(PCI `33:00.0`), CPUs `32-63`, verified with `torch.gcu.set_device(1)` +
`assert current_device() == 1` in every script. GCU 6/7 were never used.

## TL;DR

PARALLAX-004..007 swept `BS / num_warps / num_stages` and froze them. The
query-row tile **`BT` was never in any sweep** — it was pinned to `_BLOCK_T=64`
by construction. Widening it is a different axis and, at this shape, the
dominant one:

| stage (FP16) | shipped | row-tile | delta |
|---|---|---|---|
| DKV | 367.2 us | 214.5 us | **-41.6 %** |
| DQR | 221.7 us | 194.3 us | -12.4 % |
| FWD | 193.0 us | 181.9 us | -5.7 % |
| **full fwd+bwd** | **776 us** | **620 us** | **-20.1 %** |
| full fwd+bwd (BF16) | 775 us | 621 us | -19.9 % |

Opt-in via `PARALLAX_ROW_TILE_D64=1`, default **off**, gated to the exact shape.
Not a replacement for `PARALLAX_STATIC_D64` and does not touch the default route.

## Why this is not another warp/stage sweep

The previous task's negative result was that static tuning cut cold time but gave
no steady-state gain, with unstable config ranking. This experiment changes a
dimension those sweeps structurally could not see. Two measured consequences,
both re-derived from the kernel source rather than assumed:

* **DKV loops over query row blocks serially inside the GQA group loop.** At
  `BS=BT=64` the four column CTAs together walk 10 row blocks per group (40 per
  `(b,h)`); at `BS=64/BT=256` each CTA walks exactly one (16 per `(b,h)`) with
  4x wider MMAs. That is **1.6x more row x col element work** (65536 vs 40960
  per group, because the wide tile also walks the masked-out upper triangle) and
  still 1.71x faster. So DKV here is bound by serial iteration count and MMA
  width, not by element count.
* **FWD/DQR are grid-parallel over tiles**, so their gain is small and comes
  from launch width, *not* mask avoidance. FWD goes from 4 CTAs x 6 column
  iterations to 1 CTA x 2 at 4x wider MMAs, while its unmasked "safe" fraction
  *falls* 0.33 -> 0.00 — at `BT=256` every column block is causally masked and
  it is still faster. This is the direct evidence that the win is launch shape.

An earlier draft of this comment claimed the opposite (that widening moved more
blocks into the unmasked branch). It was wrong for both kernels and is corrected
in the code comment; the arithmetic is in "Failure modes" below.

## Method

Three separate measurement problems, kept separate:

1. **Cold attribution** — fresh `TRITON_CACHE_DIR` per `(dtype, route)`, one
   process each, recording wall time, compile count, autotune `_bench` call
   count and per-kernel launch counts.
2. **Warm per-stage** — one kernel instrumented *per pass*. Wrapping all three
   kernels in one pass inflates and even reorders the middle stage; that mistake
   produced a spurious FP16 DQR inversion (see Failure modes).
3. **Warm full-call A/B** — same process, same tensors, routes alternating every
   round, 9 rounds x 30 iters, medians. Peak memory measured separately with
   `reset_peak_memory_stats()` and a warm allocator.

## Results

### Cold: first call, fresh cache

| dtype | route | wall s | compiles | autotune `_bench` calls | launches fwd/dqr/dkv |
|---|---|---|---|---|---|
| fp16 | shipped | 389.42 | 30 | 30 (386.6 s) | 73 / 97 / 73 |
| fp16 | row-tile | **43.85** | 3 | 0 | 1 / 1 / 1 |
| bf16 | shipped | 388.47 | 30 | 30 (385.7 s) | 73 / 97 / 73 |
| bf16 | row-tile | **43.84** | 3 | 0 | 1 / 1 / 1 |

The cold cost is **entirely autotune benchmarking**: 386.6 of 389.4 s shipped vs
0 s row-tile. With a warm Triton cache both routes are 2.8 s, so the row-tile
route removes a *cold-cache* cost that is otherwise paid once per cache state,
not a per-call cost.

### Warm per-stage (one kernel instrumented per pass, 3 rounds, median of 30)

| dtype | stage | shipped | row-tile | delta |
|---|---|---|---|---|
| fp16 | fwd | 192.97 | 181.91 | -5.73 % |
| fp16 | dqr | 221.65 | 194.28 | -12.35 % |
| fp16 | dkv | 367.21 | 214.50 | **-41.59 %** |
| bf16 | fwd | 188.92 | 181.37 | -4.00 % |
| bf16 | dqr | 222.45 | 194.21 | -12.69 % |
| bf16 | dkv | 368.19 | 214.03 | **-41.87 %** |

Stage sum vs full call: fp16 781.8 / 590.7 us sum against 794.7 / 609.3 us
measured, bf16 779.6 / 589.6 against 793.3 / 607.8. Ratios 0.98 / 0.97 — stages
account for the full call without the instrumentation inflating it.

### Warm full-call A/B (9 alternating rounds x 30 iters)

Three full 9-round runs of the same code, each after a forced recompile:

| run | dtype | shipped | row-tile | ratio |
|---|---|---|---|---|
| 1 | fp16 | 776.15 | 619.86 | 0.7986 |
| 1 | bf16 | 774.87 | 620.80 | 0.8012 |
| 2 | fp16 | 779.81 | 619.55 | 0.7945 |
| 2 | bf16 | 779.00 | 621.36 | 0.7976 |
| 3 | fp16 | **813.77** | **619.23** | **0.7609** |
| 3 | bf16 | 775.32 | 620.56 | 0.8004 |

The row-tile route is the stable one: **619.2-621.4 us in all six measurements**
(0.35 % total spread). The shipped autotuned baseline is not: fp16 moved
776 -> 780 -> 814 us across identical runs, so run 3's -23.9 % is *not* a better
row-tile result, it is autotune having picked a worse fp16 configuration that
run. This is the same autotune instability PARALLAX-007 reported, now visible as
a 4.8 % swing in the baseline between runs with the candidate fixed.

So the honest headline is the **range**: -20 % (runs 1-2, and all three bf16
measurements) to -24 % (run 3 fp16). The conservative figure to quote is
**-20 %, ratio 0.79**, since it is what both dtypes produced reproducibly.
Per-round deltas never flip sign in any run (9/9 in all six), and within a run
the spread is under 1 point.

### Choice of table (interleaved variant probe, 5 rounds)

| variant | fp16 | bf16 |
|---|---|---|
| shipped (BT=64) | 797.27 | 807.30 |
| **bt128-dqr (shipped table)** | **610.93** | **608.62** |
| bt64-dqr | 644.43 | 639.60 |
| dqr 64x128 | 635.05 | 631.19 |
| bt64-dqr, fwd not widened | 661.93 | 652.54 |

`dqr: (128,128,1,1)` is best in both dtypes, so the committed table is not a
per-dtype compromise. `dkv` must be `BT=256`: `BS=128/BT=256` and
`BS=256/BT>=128` both raise `OutOfResources(..., 'local memory')` on GCU 1.

## Correctness

* `tests/flag_attn/test_parallax_row_tile_d64.py` (30) +
  `test_parallax_optin_precedence.py` (7) — **37 passed** in 58.0 s: gate
  accept/reject, exact launch tuples and grids, output + dq/dr/dk/dv numerics,
  nearby shapes not gated, and the STATIC/FIXED precedence rules.
* Nearby/fallback shapes with the flag **on vs off**, both dtypes: bit-identical
  (max abs diff exactly 0 for output and all four gradients):

| shape | fp16 rel-RMS vs FP32 | bf16 rel-RMS vs FP32 |
|---|---|---|
| B1/T256 | 3.3e-4 … 3.6e-4 | 2.7e-3 … 3.0e-3 |
| B2/T128 | 3.5e-4 … 3.9e-4 | 2.8e-3 … 3.2e-3 |
| B1/T63 | 2.9e-4 … 4.9e-4 | 2.3e-3 … 3.6e-3 |

  All inside tolerance (FP16 0.005, BF16 0.02) and all finite. The gate is exact,
  so these shapes are unchanged by construction — this confirms it rather than
  assuming it.
* Formal suite `pytest tests/flag_attn/test_parallax.py -q`: **32 passed,
  1 warning in 6386.39 s (1:46:26)**, wall 6394.09 s, RC 0, on the committed
  tree (`formal-pytest-final.log`). An earlier run of the same suite on the
  pre-precedence tree took 3186.77 s (0:53:06); the final run is slower only
  because host load average was 25 throughout from unrelated agents, not
  because of any test change. Both runs are recorded.

## Memory

Peak allocated is **identical** between routes: 5.33 MB allocated / 6.0 MB
reserved on a warm allocator, both dtypes, both routes; same 5.33 MB in a fresh
process per `(dtype, route)`. Row tiling only changes the launch geometry, not
the allocation set — the gain is not bought with memory.

## Failure modes and what they cost

* **Instrumenting all three kernels in one pass corrupts the middle stage.** It
  produced a FP16 "DQR BT=128 is worse than BT=64" result (238.8 vs 228.1 us)
  that the full call contradicted. Measuring one kernel per pass, and an
  interleaved dedicated cross-check, both give BT=128 better: 194.1 vs 221.8 us,
  full call 610.8 vs 642.1 us. The original per-stage inversion was an artifact
  and is not reported as a finding.
* **`del outs` inside a route loop** aborted a run after 2 records; peak memory
  moved to an isolated mode.
* **Autotune `_bench` replays were being timed**, giving
  `stage_sum_over_full` 1.03-1.06. Fixed with a `bench_depth` guard.
* **Warm-up polluted the launch-config assertion** (fp16 shipped warm-up launched
  BT=64, then candidate BT=256). Fixed by warming both routes before measuring.
* **One variant round contained a 4.79 ms outlier** (fp16 `bt64-dqr` round 2,
  against ~644 us in its other four rounds; visible in `variant.jsonl`). It is
  why per-round medians are used and why the reported deltas are medians rather
  than means; the outlier does not change any ranking.
* **`torch.gcu` eager `fill_`/`add` are ~200x slower than copy** (1.4-2.7 GB/s vs
  428-434 GB/s for copy at 64-256 MB). They are an unfused eager path, not a
  bandwidth limit, so they must not be quoted as a roofline. Only copy is used.
* **The autotuned baseline drifts between runs** (fp16 776 -> 780 -> 814 us,
  see the A/B table). Any speed claim here has to be a range with the candidate
  side quoted as the stable number, never a single ratio.
* **The priority test caught a real mixed-shape bug.** FWD checked
  `PARALLAX_ROW_TILE_D64` before `PARALLAX_STATIC_D64` and BWD checked them in
  the opposite order, so with both exported the forward pass launched
  `(BS=128, BT=256)` while DQR launched the static `(BS=128, BT=64, stages=2)`.
  Both are now static-first, and `PARALLAX_FIXED_D64` suppresses row-tile.
  3 precedence tests in `test_parallax_optin_precedence.py` cover it.
* **Device vs host quiet.** GCU 1 was idle throughout (0 % DUsed, ~55 C, 134 W),
  but host load average was 25-52 from unrelated agents on GCU 4/5. Timing is
  same-process alternating A/B, so drift hits both arms; the 9/9 sign stability
  and <1 point spread are the evidence that this did not distort the result.
  Still, these are single-host numbers, not a claim about a quiet cluster.

## Files

* `src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py` — +172/-1.
  `_ROW_TILE_D64_ENV`, `_row_tile_d64_enabled()`, `_row_tile_d64_config()`, the
  `_ROW_TILE_D64_CONFIGS` table, and dispatch in fwd/DQR/DKV. `_FIXED_D64_ENV`
  and `_STATIC_D64_CONFIGS` untouched, default off.
* `tests/flag_attn/test_parallax_row_tile_d64.py` — 30 tests.
* `tests/flag_attn/test_parallax_optin_precedence.py` — 7 tests covering
  STATIC > ROW_TILE and FIXED > ROW_TILE precedence on real launch tuples.
* `agent-results/scripts/` — `parallax008_{final,stage,ab_final,variant,dqrcheck,nearby,bw}.py`,
  `cold_driver.sh`, `hot_driver.sh`, `mem_driver.sh`.
* `agent-results/raw/PARALLAX-008/` — cold attribution, stage-clean, ab-final,
  variant, dqrcheck, nearby-final, mem, bw, both pytest logs.

## Reviewer notes

* The declared formal command is `pytest tests/flag_attn/test_parallax.py -q`,
  which does **not** pin a device: `torch.gcu.current_device()` defaults to 0.
  Our runs add `-p gcu1_plugin` (`experiments/gcu1_plugin.py`) to pin GCU 1 for
  the whole session, so the formal reviewer's run on the default device 0 is the
  first time this suite exercises that device. Nothing here reads or asserts a
  device index, so it should be device-agnostic, but it has not been observed.
* `PARALLAX_ROW_TILE_D64` is default **off**. A reviewer run without it set
  exercises only the unchanged autotuned route, i.e. the same behaviour as
  baseline `9eca4bb`. The new tests set the flag explicitly per test via
  `monkeypatch.setenv`, so they are meaningful either way.
* No push, no merge, no change to the default production route.

## Recommended next task

The remaining serial axis is DKV's GQA group loop: `i_g in range(0, G)` is walked
serially per CTA with 4 heads at this shape, and the measured 40->16 iteration
drop shows this kernel responds to serial-length reduction far more than to
element count. Folding the group loop into the grid (one CTA per
`(col_block, b, h, g)` with a split-K accumulation) is the next structural
candidate, and it attacks the same quantity this task found to be binding. It is
a real kernel change, not a launch-shape one, so it needs its own correctness
work.

A cheaper follow-up: `fwd` at `BT=256` lost all its safe-middle iterations yet
still improved. Checking whether `BS=256` recovers the safe branch for fwd is a
small, self-contained launch-shape question with the harness already in place.
