
### float16 inputs, float16 output steady-state medians

| shape | device fused ms | Sage full ms | Sage/device | device peak MiB | Sage peak MiB |
|---|---|---|---|---|---|
| 1x2x4096x64 | 0.2900 | 1.1484 | 3.96x | 9.03 | 10.00 |
| 4x32x1024x128 | 1.2149 | 5.5010 | 4.53x | 288.52 | 320.04 |

### float16 inputs, float16 output stage attribution (per round, ms)

| shape | round | quant tuned | quant default launch | speedup | attention | attention/device | quant share |
|---|---|---|---|---|---|---|---|
| 1x2x4096x64 | 0 | 0.1694 | 0.6201 | 3.66x | 1.0409 | 3.58x | 15% |
| 1x2x4096x64 | 1 | 0.1694 | 0.6538 | 3.86x | 1.0342 | 3.64x | 15% |
| 1x2x4096x64 | 2 | 0.1747 | 0.6089 | 3.48x | 1.0365 | 3.57x | 15% |
| 4x32x1024x128 | 0 | 1.4966 | 7.7247 | 5.16x | 3.9025 | 3.21x | 28% |
| 4x32x1024x128 | 1 | 1.4906 | 7.6687 | 5.14x | 3.9127 | 3.25x | 26% |
| 4x32x1024x128 | 2 | 1.5031 | 7.6720 | 5.10x | 3.9055 | 3.21x | 27% |

### float16 inputs, float16 output accuracy against the device fused op

| shape | round | cosine | relative L2 | max abs | finite |
|---|---|---|---|---|---|
| 1x2x4096x64 | 0 | 0.9999245 | 0.01165 | 0.002319 | True |
| 1x2x4096x64 | 1 | 0.9999208 | 0.01185 | 0.002686 | True |
| 1x2x4096x64 | 2 | 0.9999197 | 0.01213 | 0.002380 | True |
| 4x32x1024x128 | 0 | 0.9999293 | 0.01208 | 0.007080 | True |
| 4x32x1024x128 | 1 | 0.9999315 | 0.01227 | 0.005920 | True |
| 4x32x1024x128 | 2 | 0.9999326 | 0.01227 | 0.006836 | True |

### float16 inputs, float16 output cold compile (fresh Triton cache)

| shape | scope | first call s | steady ms | compile overhead s | cache files after |
|---|---|---|---|---|---|
| 1x2x4096x64 | full | 8.81 | 1.1400 | 8.81 | 24 |
| 1x2x4096x64 | stages | 4.01 | 0.1865 | 4.01 | 24 |
| 4x32x1024x128 | full | 7.97 | 5.7466 | 7.96 | 24 |
| 4x32x1024x128 | stages | 4.10 | 1.5309 | 4.10 | 24 |

### float16 inputs, float16 output device path evidence

- device baseline ops: `['GCU::_scaled_dot_product_flash_attention', 'aten::_scaled_dot_product_flash_attention', 'topsDeviceSynchronize', 'topsDeviceSynchronize', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsLaunchKernel']`
- reference F.sdpa ops: `['GCU::_copy_from', 'GCU::_copy_from', 'GCU::_copy_from', 'GCU::_copy_from', 'GCU::wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention', 'aten::_copy_from', 'aten::_copy_from', 'aten::_copy_from', 'aten::_copy_from', 'aten::_scaled_dot_product_flash_attention_for_cpu', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_cpu', 'aten::copy_', 'aten::copy_', 'aten::copy_', 'aten::copy_', 'aten::scaled_dot_product_attention', 'topsDeviceSynchronize', 'topsDeviceSynchronize', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsStreamSynchronize', 'topsStreamSynchronize', 'topsStreamSynchronize', 'topsStreamSynchronize']`
- host CPU per call: device 0.162 ms, F.sdpa 32.01 ms
- device baseline is a GCU kernel: True
- reference F.sdpa is the CPU fallback: True

### float16 inputs, bfloat16 output steady-state medians

| shape | device fused ms | Sage full ms | Sage/device | device peak MiB | Sage peak MiB |
|---|---|---|---|---|---|
| 1x2x4096x64 | 0.2894 | 1.1490 | 3.97x | 9.03 | 10.00 |
| 4x32x1024x128 | 1.2079 | 5.7545 | 4.76x | 288.52 | 320.04 |

### float16 inputs, bfloat16 output stage attribution (per round, ms)

| shape | round | quant tuned | quant default launch | speedup | attention | attention/device | quant share |
|---|---|---|---|---|---|---|---|
| 1x2x4096x64 | 0 | 0.1827 | 0.6178 | 3.38x | 1.0373 | 3.56x | 16% |
| 1x2x4096x64 | 1 | 0.1734 | 0.6538 | 3.77x | 1.0342 | 3.59x | 15% |
| 1x2x4096x64 | 2 | 0.1745 | 0.6073 | 3.48x | 1.0348 | 3.58x | 15% |
| 4x32x1024x128 | 0 | 1.4911 | 7.6570 | 5.14x | 3.9033 | 3.22x | 20% |
| 4x32x1024x128 | 1 | 1.4917 | 7.6694 | 5.14x | 3.8974 | 3.24x | 26% |
| 4x32x1024x128 | 2 | 1.8041 | 7.6612 | 4.25x | 3.8931 | 3.22x | 34% |

### float16 inputs, bfloat16 output accuracy against the device fused op

| shape | round | cosine | relative L2 | max abs | finite |
|---|---|---|---|---|---|
| 1x2x4096x64 | 0 | 0.9998858 | 0.01188 | 0.002075 | True |
| 1x2x4096x64 | 1 | 0.9998826 | 0.01208 | 0.002686 | True |
| 1x2x4096x64 | 2 | 0.9998801 | 0.01236 | 0.001953 | True |
| 4x32x1024x128 | 0 | 0.9998489 | 0.01231 | 0.007812 | True |
| 4x32x1024x128 | 1 | 0.9998466 | 0.01249 | 0.005859 | True |
| 4x32x1024x128 | 2 | 0.9998499 | 0.01249 | 0.006836 | True |

### float16 inputs, bfloat16 output cold compile (fresh Triton cache)

| shape | scope | first call s | steady ms | compile overhead s | cache files after |
|---|---|---|---|---|---|
| 1x2x4096x64 | full | 8.86 | 1.1390 | 8.86 | 24 |
| 1x2x4096x64 | stages | 4.05 | 0.1858 | 4.05 | 24 |
| 4x32x1024x128 | full | 8.02 | 9.0942 | 8.01 | 24 |
| 4x32x1024x128 | stages | 4.16 | 1.5172 | 4.16 | 24 |

### float16 inputs, bfloat16 output device path evidence

- device baseline ops: `['GCU::_scaled_dot_product_flash_attention', 'aten::_scaled_dot_product_flash_attention', 'topsDeviceSynchronize', 'topsDeviceSynchronize', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsLaunchKernel']`
- reference F.sdpa ops: `['GCU::_copy_from', 'GCU::_copy_from', 'GCU::_copy_from', 'GCU::_copy_from', 'GCU::wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention', 'aten::_copy_from', 'aten::_copy_from', 'aten::_copy_from', 'aten::_copy_from', 'aten::_scaled_dot_product_flash_attention_for_cpu', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_cpu', 'aten::copy_', 'aten::copy_', 'aten::copy_', 'aten::copy_', 'aten::scaled_dot_product_attention', 'topsDeviceSynchronize', 'topsDeviceSynchronize', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsStreamSynchronize', 'topsStreamSynchronize', 'topsStreamSynchronize', 'topsStreamSynchronize']`
- host CPU per call: device 0.349 ms, F.sdpa 31.93 ms
- device baseline is a GCU kernel: True
- reference F.sdpa is the CPU fallback: True

### float16 additional shapes steady-state medians

| shape | device fused ms | Sage full ms | Sage/device | device peak MiB | Sage peak MiB |
|---|---|---|---|---|---|
| 1x2x2048x64 | 0.1323 | 0.4369 | 3.30x | 4.52 | 5.00 |
| 1x2x8192x64 | 0.8289 | 2.7967 | 3.37x | 18.07 | 22.00 |
| 2x8x512x128 | 0.1088 | 0.3779 | 3.47x | 18.03 | 20.00 |
| 4x32x2048x128 | 5.1101 | 30.8630 | 6.04x | 577.05 | 640.07 |

### float16 additional shapes stage attribution (per round, ms)

| shape | round | quant tuned | quant default launch | speedup | attention | attention/device | quant share |
|---|---|---|---|---|---|---|---|
| 1x2x2048x64 | 0 | 0.1336 | 0.3433 | 2.57x | 0.3307 | 2.45x | 30% |
| 1x2x2048x64 | 1 | 0.1422 | 0.3583 | 2.52x | 0.3296 | 2.54x | 33% |
| 1x2x2048x64 | 2 | 0.1314 | 0.3710 | 2.82x | 0.3292 | 2.49x | 31% |
| 1x2x8192x64 | 0 | 0.2621 | 1.1494 | 4.39x | 3.3438 | 4.03x | 9% |
| 1x2x8192x64 | 1 | 0.2600 | 1.1835 | 4.55x | 2.5447 | 3.09x | 9% |
| 1x2x8192x64 | 2 | 0.2630 | 1.1657 | 4.43x | 2.5522 | 3.08x | 10% |
| 2x8x512x128 | 0 | 0.1759 | 0.5788 | 3.29x | 0.2480 | 2.28x | 47% |
| 2x8x512x128 | 1 | 0.1745 | 0.5544 | 3.18x | 0.2334 | 2.19x | 43% |
| 2x8x512x128 | 2 | 0.1758 | 0.5727 | 3.26x | 0.2392 | 2.14x | 48% |
| 4x32x2048x128 | 0 | 2.8632 | 15.2417 | 5.32x | 14.4209 | 2.82x | 9% |
| 4x32x2048x128 | 1 | 2.9053 | 15.2370 | 5.24x | 14.4388 | 2.85x | 12% |
| 4x32x2048x128 | 2 | 2.8987 | 15.2223 | 5.25x | 14.4072 | 2.82x | 9% |

### float16 additional shapes accuracy against the device fused op

| shape | round | cosine | relative L2 | max abs | finite |
|---|---|---|---|---|---|
| 1x2x2048x64 | 0 | 0.9999245 | 0.01169 | 0.003052 | True |
| 1x2x2048x64 | 1 | 0.9999205 | 0.01200 | 0.002716 | True |
| 1x2x2048x64 | 2 | 0.9999183 | 0.01231 | 0.004150 | True |
| 1x2x8192x64 | 0 | 0.9999250 | 0.01245 | 0.005249 | True |
| 1x2x8192x64 | 1 | 0.9999309 | 0.01160 | 0.001984 | True |
| 1x2x8192x64 | 2 | 0.9999291 | 0.01223 | 0.001476 | True |
| 2x8x512x128 | 0 | 0.9999301 | 0.01211 | 0.011230 | True |
| 2x8x512x128 | 1 | 0.9999340 | 0.01204 | 0.008301 | True |
| 2x8x512x128 | 2 | 0.9999334 | 0.01208 | 0.013672 | True |
| 4x32x2048x128 | 0 | 0.9999281 | 0.01249 | 0.004883 | True |
| 4x32x2048x128 | 1 | 0.9999280 | 0.01235 | 0.006104 | True |
| 4x32x2048x128 | 2 | 0.9999297 | 0.01204 | 0.009399 | True |

### float16 additional shapes cold compile (fresh Triton cache)

| shape | scope | first call s | steady ms | compile overhead s | cache files after |
|---|---|---|---|---|---|

### float16 additional shapes device path evidence

- device baseline ops: `['GCU::_scaled_dot_product_flash_attention', 'aten::_scaled_dot_product_flash_attention', 'topsDeviceSynchronize', 'topsDeviceSynchronize', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsLaunchKernel']`
- reference F.sdpa ops: `['GCU::_copy_from', 'GCU::_copy_from', 'GCU::_copy_from', 'GCU::_copy_from', 'GCU::wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention', 'aten::_copy_from', 'aten::_copy_from', 'aten::_copy_from', 'aten::_copy_from', 'aten::_scaled_dot_product_flash_attention_for_cpu', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_copy', 'aten::_to_cpu', 'aten::copy_', 'aten::copy_', 'aten::copy_', 'aten::copy_', 'aten::scaled_dot_product_attention', 'topsDeviceSynchronize', 'topsDeviceSynchronize', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsGetDevice', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsMemcpyAsync', 'topsStreamSynchronize', 'topsStreamSynchronize', 'topsStreamSynchronize', 'topsStreamSynchronize']`
- host CPU per call: device 0.361 ms, F.sdpa 31.90 ms
- device baseline is a GCU kernel: True
- reference F.sdpa is the CPU fallback: True

### attention kernel launch sweep (quantization excluded)

| shape | device ms | best config | best ms | best/device | worst/device |
|---|---|---|---|---|---|
| 1x2x4096x64 | 0.2780 | bm=128 bn=128 nw=1 ns=2 | 1.0526 | 3.79x | 12.02x |
| 4x32x1024x128 | 1.1775 | bm=128 bn=128 nw=1 ns=1 | 3.9014 | 3.31x | 17.72x |

### full-call latency modes (FP16, all rounds and shapes)

| shape | dtype | median ms | fast mode ms | slow mode ms | ratio | slow calls |
|---|---|---|---|---|---|---|
| 1x2x4096x64 | fp16 | 1.148 | - | - | - | unimodal |
| 1x2x4096x64 | fp16 | 1.152 | - | - | - | unimodal |
| 1x2x4096x64 | fp16 | 1.144 | - | - | - | unimodal |
| 4x32x1024x128 | fp16 | 5.369 | 5.329 | 8.980 | 1.69x | 30% |
| 4x32x1024x128 | fp16 | 5.708 | 5.699 | 9.080 | 1.59x | 30% |
| 4x32x1024x128 | fp16 | 5.501 | - | - | - | unimodal |
| 1x2x4096x64 | bf16 | 1.151 | 1.150 | 1.778 | 1.55x | 20% |
| 1x2x4096x64 | bf16 | 1.149 | - | - | - | unimodal |
| 1x2x4096x64 | bf16 | 1.142 | 1.138 | 2.003 | 1.76x | 10% |
| 4x32x1024x128 | bf16 | 7.340 | 5.682 | 9.009 | 1.59x | 50% |
| 4x32x1024x128 | bf16 | 5.755 | 5.542 | 9.082 | 1.64x | 40% |
| 4x32x1024x128 | bf16 | 5.360 | 5.360 | 9.067 | 1.69x | 10% |
| 1x2x2048x64 | fp16 | 0.438 | - | - | - | unimodal |
| 1x2x2048x64 | fp16 | 0.437 | - | - | - | unimodal |
| 1x2x2048x64 | fp16 | 0.426 | - | - | - | unimodal |
| 1x2x8192x64 | fp16 | 2.797 | - | - | - | unimodal |
| 1x2x8192x64 | fp16 | 2.799 | - | - | - | unimodal |
| 1x2x8192x64 | fp16 | 2.746 | - | - | - | unimodal |
| 2x8x512x128 | fp16 | 0.378 | 0.363 | 0.482 | 1.33x | 40% |
| 2x8x512x128 | fp16 | 0.411 | 0.362 | 0.467 | 1.29x | 50% |
| 2x8x512x128 | fp16 | 0.365 | 0.361 | 0.473 | 1.31x | 20% |
| 4x32x2048x128 | fp16 | 30.863 | 18.115 | 30.883 | 1.70x | 70% |
| 4x32x2048x128 | fp16 | 24.128 | 17.221 | 30.969 | 1.80x | 50% |
| 4x32x2048x128 | fp16 | 30.968 | 17.677 | 30.974 | 1.75x | 80% |

Bimodal rounds exist: **True**. The ratio is medians of the two modes; `slow calls` is the share falling in the slower mode.

### negative controls (must be nonzero)

| control | expected | observed |
|---|---|---|
| sage003_negative_numeric.exit | 3 | NEG_NUMERIC_EXIT=3 |
| sage003_negative_samples.exit | 3 | NEG_SAMPLES_EXIT=3 |
| sage003_ab_float16.exit | 0 | float16_EXIT=0 |
| sage003_ab_bfloat16.exit | 0 | bfloat16_EXIT=0 |
| sage003_attn_sweep.exit | 0 | SWEEP_EXIT=0 |
| sage003_tests.exit | 0 | TESTS_EXIT=0 |

Headline: Sage is 3.96-4.53x slower than the device fused op on the formal shapes.
Quantization is up to 28% of the Sage call; the SAGE-003 launch change is 3.48-5.16x faster than the default launch with bitwise identical int8/scale outputs. Even with quantization free the attention kernel alone would still be 3.21x the device op.
