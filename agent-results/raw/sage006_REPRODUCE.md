# SAGE-006 raw evidence index

Every file here was produced on GCU 5 (PCI `0000:83:00.0`, SN `A0D5460520864`)
with `taskset -c 160-191`. GCU 6 and GCU 7 were never touched.

## Provenance levels

| level | meaning |
| --- | --- |
| A | per-sample JSONL committed during SAGE-005 (`sage005_*.jsonl`) |
| B | per-sample JSONL from a SAGE-006 re-run (`sage006_recheck_*.jsonl`) |
| C | verbatim stdout recovered from the SAGE-005 session log (`sage005_recovered/*.log`) |

The `/tmp` scripts and IR dumps survived on the worker and are committed under
`scripts/recovered/` and `sage005_artifacts/`. The per-sample JSONL for the C
files never existed — SAGE-005 printed medians only — so it is not claimed.

## SAGE-006 re-runs (level B), all exit 0

| file | rows | script | command |
| --- | ---: | --- | --- |
| `sage006_recheck_fp16.jsonl` | 3 | `sage005_recheck.py` | `--probe fp16` |
| `sage006_recheck_fuse.jsonl` | 11 | `sage005_recheck.py` | `--probe fuse` |
| `sage006_recheck_flags.jsonl` | 17 | `sage005_recheck.py` | `--probe flags` |
| `sage006_recheck_stages.jsonl` | 25 | `sage005_recheck.py` | `--probe stages` |
| `sage006_recheck_waves.jsonl` | 11 | `sage005_recheck.py` | `--probe waves` |
| `sage006_recheck_bw_marginal.jsonl` | 42 | `sage005_recheck.py` | `--probe bw,marginal` |
| `sage006_artifacts.jsonl` | 10 | `sage006_artifacts.py` | `--ir ... --vendor --sha256` |

Full command line, e.g.:

```bash
cd /workspace/agent-workspaces/sage-review-fix-5
export PYTHONPATH=/opt/flagtree:$PWD/src
taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe fp16 \
  --jsonl agent-results/raw/sage006_recheck_fp16.jsonl
```

Every run starts with an `env` record carrying the commit, host, stack versions,
load average and the `--probe` target, and every measurement carries
`device_index` and `pci`. The command lines above are the ones actually used;
the JSONL does not record `argv` itself (`sage006_artifacts.jsonl` records the
input paths instead), so the only authoritative record of a run's provenance is
the `env` record plus the `.exit` file.

`raw/sage006_recheck_fp16.jsonl` is the clean `--probe fp16`-only run that the
report cites. `raw/sage006_recheck_fp16_fuse.jsonl` is an earlier combined run
kept for audit; it aborted after the FP16 probe because the variant builder
resolved the repo root from `__file__`, which points outside the tree when the
script is staged in `/tmp`. That is also why its `env` record has an empty
`commit`: `git rev-parse HEAD` was run in `/tmp`, which is not a repository. The
fix (resolve from cwd / `SAGE006_REPO`) is in the committed `sage005_guard.py`;
that run's `.exit` is 1 and it is not cited. Every other `sage006_*` file has
`commit = 167c360…` in its `env` record, `device_index = 5` and
`pci = 0000:83:00.0` on every measurement row.

## Recovered SAGE-005 artefacts (level C)

| file | contents |
| --- | --- |
| `sage005_recovered/fp16attn.log` | the 1.0137 / 4.2598 ms pure-FP16 stdout |
| `sage005_recovered/fuse.log` | base/fused_expr/inplace_sub timings, then the dict-key crash |
| `sage005_recovered/flags.log` | the full 8-combination sweep per shape |
| `sage005_recovered/stage2.log` | tile-major `num_stages` x warps sweep |
| `sage005_recovered/stages_qk.log` | row-major plain vs manual-prefetch sweep |
| `sage005_recovered/marginal.log` | single-wave sweep, including the D128 overflow points |
| `sage005_recovered/bw.log` | copy/read bandwidth by block size |
| `sage005_recovered/bh.log` | per-BH wave quantisation |
| `sage005_recovered/waves.log`, `waves_failed.log` | first failed run and the fixed run |
| `sage005_recovered/acc.log`, `sage005_acc.json` | 6 shapes x 2 dtypes accuracy records |
| `sage005_recovered/*.exit` | the shell exit code of each recovered run |

Each `.log` begins with the `callId` of the SAGE-005 session event it came from,
so it can be diffed against the run's `events.jsonl`.

The `.exit` value is the **shell pipeline's** exit code, so it is only
meaningful where the command did not end in `| tail`/`| grep`. Two recovered runs
failed even though the wrapper reported success:

- `fuse.exit` is 0, but the script died right after the timings: `outs[n]` holds
  the tuple returned by `attn.forward(...)`, so `outs[n].float()` raised
  `AttributeError: 'tuple' object has no attribute 'float'`. The equality check
  it was about to print never ran. The truncated traceback tail is in
  `fuse.log`; the full stderr was filtered by the session's own `grep`.
- `waves_failed.exit` is 1 (`ValueError: block_m must be 64 or 128`); the
  corrected run is `waves.log`.

Both failures are kept rather than deleted, and `sage005_recheck.py` fixes both
(index `[0]` on the forward result, and the `block_m` mapping).

## Formal gate

| file | contents |
| --- | --- |
| `sage006_formal_gate_record.json` | orchestrator `record.json`: exit 0, device 5, PCI `83:00.0`, CPUs 160-191, commit `167c360` |
| `sage006_formal_gate_stdout.log` | `42 passed, 1 warning in 55.69s` |
| `sage006_formal_gate.exit` | `0` |

`sage005_formal_gate.txt` is the **superseded local pre-check** (exit 1, 40
passed, 2 failed) whose command line omitted `PYTHONPATH`. It is kept for audit
and is not the acceptance result.

## Artefact hashes

| file | sha256 |
| --- | --- |
| `sage005_artifacts/sage005_attn_gcuir.txt` | `01cccdc0af5da18986cc9ec3474c266eebdf885bc0ca32331ed504382458c740` |
| `sage005_artifacts/sage005_attn_llir.txt` | `d137f2405ce66962b71dec8c180d0e989ae57033c7d692a9a1e031ecc289e2d6` |
| `sage005_artifacts/sage005_attn_ttgir.txt` | `2e5956467d0f3b96bc2d4e5282852a19d6033531a79e5d3563791da487b3a46e` |
| `sage005_artifacts/fa_bf16_256.s` | `9d0d38e683d5d6c30f6456f89c79a28a785396a2ab929ecef3b00c63aa461082` |

Vendor library digests, recomputed by `sage006_artifacts.py --sha256`:
`libkernel_flash_attn_fwd.so` `4d77dfd7...`, `.tops_fatbin` `1552b748...`,
`fa.kernel` `b85ab534...`, `fa.s` `bfef3add...`.
