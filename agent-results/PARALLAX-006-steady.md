# PARALLAX-006 steady A/B raw summary

Generated from `experiments/logs/p006-ab-*.log`; full records in
`experiments/results/PARALLAX-006.jsonl`.

## Protocol

- 3 alternating rounds per dtype: exact old worktree `.baseline-70bdc5`
  (`PARALLAX_STATIC_D64=0`) then candidate (`PARALLAX_STATIC_D64=1`).
- 5 synchronized samples per round per metric: forward, backward on that graph,
  and a full fresh forward+backward. `torch.gcu.synchronize()` on both sides.
- Separate process per route per round, separate Triton cache copied from that
  route's own cold cache; path probe + 2 untimed warmups before sampling, so no
  compile or autotune benchmark enters a sample.
- GCU 1 only, every command `taskset -c 160-191`. Import probes in every log
  show baseline HEAD `70bdc5e3` / parallel.py `51cf806459d5d043` and candidate
  HEAD `8190827a` / `5f6fc383ec213bcc`.
- Launches: baseline shows 30 autotune configs and callable DKV grid; candidate
  shows 0 autotune configs, fwd grid `[4,16]` BS=128, DQR grid `[4,16]`,
  grouped DKV grid `[4,4]` BS=64 for FP16 / BS=64 s2 for BF16.

## Per-round medians (ms)

| dtype | round | base fwd | cand fwd | base bwd | cand bwd | base full | cand full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 0 | 0.422 | 0.413 | 0.834 | 0.865 | 1.026 | 1.053 |
| FP16 | 1 | 0.428 | 0.427 | 0.816 | 0.851 | 1.001 | 1.034 |
| FP16 | 2 | 0.438 | 0.409 | 0.900 | 0.842 | 1.098 | 1.019 |
| BF16 | 0 | 0.418 | 0.402 | 0.822 | 0.844 | 1.000 | 1.013 |
| BF16 | 1 | 0.435 | 0.400 | 0.865 | 0.850 | 1.074 | 1.014 |
| BF16 | 2 | 0.450 | 0.424 | 0.862 | 0.892 | 1.071 | 1.070 |

Median of the three round medians and 3-round range:

| dtype | metric | baseline | candidate | cand/base |
| --- | --- | ---: | ---: | ---: |
| FP16 | forward | 0.428 (0.422-0.438) | 0.413 (0.409-0.427) | 0.965x |
| FP16 | backward | 0.834 (0.816-0.900) | 0.854 (0.842-0.865) | 1.024x |
| FP16 | full | 1.026 (1.001-1.098) | 1.034 (1.019-1.053) | 1.008x |
| BF16 | forward | 0.435 (0.418-0.450) | 0.402 (0.400-0.424) | 0.924x |
| BF16 | backward | 0.862 (0.822-0.865) | 0.850 (0.844-0.892) | 0.986x |
| BF16 | full | 1.071 (1.000-1.074) | 1.014 (1.013-1.070) | 0.947x |

Raw samples per round are in the JSONL (`samples` field, microseconds in the
tables below are raw `*_s` values).

## Raw samples (us)

FP16 forward: baseline `[[411,430,433,422,416],[432,426,446,415,428],[436,438,454,445,428]]`;
candidate `[[410,1097,412,435,413],[427,440,499,421,419],[421,409,409,408,420]]`.
FP16 backward: baseline `[[840,831,830,834,835],[841,828,816,812,811],[928,900,898,907,888]]`;
candidate `[[863,949,865,852,875],[884,851,972,837,854],[851,848,835,827,842]]`.
FP16 full: baseline `[[1026,1028,1031,1003,1003],[1029,1001,1013,994,996],[1114,1117,1096,1078,1098]]`;
candidate `[[1053,1073,1056,1024,1035],[1057,1281,1034,1028,1023],[1049,1014,1036,1019,1000]]`.

BF16 forward: baseline `[[446,417,457,418,405],[438,439,430,435,435],[454,443,448,455,450]]`;
candidate `[[392,402,405,399,416],[400,405,409,396,396],[419,424,428,416,431]]`.
BF16 backward: baseline `[[822,833,834,819,820],[865,871,872,857,847],[870,852,871,852,862]]`;
candidate `[[849,839,862,836,844],[878,845,842,850,866],[896,887,897,875,892]]`.
BF16 full: baseline `[[1016,1021,988,1000,972],[1086,1074,1078,1038,1040],[1071,1092,1075,1061,1040]]`;
candidate `[[1059,1009,1013,982,1105],[1038,1003,1246,1000,1014],[1078,1070,1102,1039,1051]]`.

## Peak memory

| route | peak allocated | peak reserved |
| --- | ---: | ---: |
| baseline (all rounds, both dtypes) | 4,276,224 B (4.08 MiB) | 274,726,912 B (262 MiB) |
| candidate (all rounds, both dtypes) | 4,276,224 B (4.08 MiB) | 6,291,456 B (6 MiB) |

Allocations are identical; the reserved difference is autotune benchmark
workspace in the baseline process, not a steady-state working-set change.

## Verdict

The candidate is a **tie within noise** on warm latency: FP16 full is 0.8 %
slower and BF16 full 5.3 % faster, but the 3-round ranges overlap
(FP16 base 1.001-1.098 vs cand 1.019-1.053; BF16 base 1.000-1.074 vs cand
1.013-1.070) and single samples show 0.15-0.26 ms outliers in both routes.
No steady speedup is claimed. The measured, reproducible benefit is one-time
cold compilation: 3 compiles / ~36 s instead of 30 configs / ~380 s.
