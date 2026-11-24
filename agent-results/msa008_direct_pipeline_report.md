# MSA-008 · MSA 长 prefill direct kernel 流水（组成拆解）

- 任务：MSA-008（operator = msa orchestrator lock）
- 分支：`agent/enflame-msa-long-prefill-direct-pipeline`
- 基线提交：`984069d7e0e57791f3ffcbe2639fff1f10905423`
- 工作副本：`/workspace/agent-workspaces/msa-stage-profile/.cache/agent-workspaces/msa-long-prefill-direct-pipeline-984069d`
- 硬件：Enflame S60（GCU300 / ZIXIAOC200），只用 **GCU 5**
- 产物：`agent-results/msa008_direct_pipeline.py`（可复跑）、`agent-results/raw/*.json` 与 `*.log`（原始样本）

---

## 1. 结论摘要

固定用例（B/Q/KV/Hkv/Hq = 1/1024/16384/1/8，D=128，page=128，TopK=16，stock 构造 + seed 0 + 随机 page 映射）：

| 量 | 数值（本卡实测，median of 7 event 样本） |
|---|---|
| 整链（index_score → index_topk → sparse_attn） | **6.781 ms** |
| `minimax_m3_sparse_attn` 单调用（stock，含 host 编排） | **5.427 ms** |
| 其中 `_gqa_sparse_common_prefix`（grid 64,1,1） | **0.372 ms** |
| 其中 `_gqa_sparse_fwd_direct`（grid 1024,1,1，USE_SHARED_PREFIX=True） | **4.999 ms** |
| direct kernel 不变、仅把 prefix 关掉（每 query 跑满 16 页） | **5.908 ms** |
| direct 每页边际成本 d(t)/d(pages)（线性拟合 r²=0.99999，三次运行 0.3439/0.3443/0.3446） | **0.3439–0.3446 ms/页** |
| direct 与页数无关的固定成本（拟合截距） | **0.393 ms** |

四个核心事实：

1. **direct 的 4.945 ms（本次 4.999 ms）几乎全部是「每页串行链」×页数**，不是某个大常数。页数扫描 1→16 页得 0.742/1.763/3.146/4.511/5.900 ms，线性拟合斜率 0.3439 ms/页、截距 0.393 ms、r²=0.99999。即 16 页里约 **5.50 ms 是 16×0.344**，只有 **0.39 ms** 是与页数无关的启动/收尾。
2. **prefix 复用只吃掉了 3 页**（`shared_count` 直方图：64 个 group 全部 = 3，均值 3.0），所以 prefix 打开后 direct 只跑 13 页而非 16 页：13×0.3439+0.393 ≈ 4.86 ms，与实测 4.999 ms 一致；关掉 prefix 是 16 页 → 5.90 ms。**prefix 的真实贡献 = 3 页 × 0.344 ≈ 1.03 ms**，而不是 0.318 ms 的 prefix kernel 时长；净收益约为 1.03 − 0.372 ≈ **0.66 ms**（实测 5.908 − 4.999 = 0.909 ms，含 page 分布变化）。
3. **每页 0.344 ms 里，PV 的矩阵乘几乎不花时间，成本压在「K 的搬运+转置」这一拍。** 同页数下的边际成本（4↔16 页斜率）：只有 K 载入+转置+QK = 0.2237；再加 online softmax = 0.2928；完整 = 0.3444；把 V 的 DMA 换成复用 K = 0.2950。逐项做差得每页预算：**K 载入 + 转置 + QK ≈ 0.2241 ms（65.1%）**、**online softmax 算术 ≈ 0.0688 ms（20.0%）**、**V 的 DMA + DTE 等待 ≈ 0.0493 ms（14.3%）**、**PV matmul ≈ 0.0022 ms（0.6%）**。四项之和 0.34437 ms 与实测 full 0.34437 ms **残差 0.00000**（无拟合残差，纯恒等式）。

   这里最反直觉的一点：**PV 的矩阵乘只有 0.002 ms/页，基本免费**；把 p（8×128）乘进 v（128×128）在 convgen 上不是成本。成本压在 K 那一拍：一次 128×128 bf16 的载入 + 一次 `tops_transpose_async_2_bf16` + 一次 QK convgen **合计** 0.224 ms，占 2/3 —— 这三者本次未能再分开（§8.6）。
4. **这不是 DRAM 带宽问题，是「每页一拍」的延迟链问题。** 把 block_table 全部指向 page 0（指令、掩码、计算完全不变，只有 KV 地址变成 256 KB 热区，必然全命中）只从 5.364 降到 5.323 ms（**−0.8%**）。也就是说 KV 随机寻址的 DRAM 侧成本几乎可以忽略；同时 `num_warps=2` 反而退化到 19.16 ms（GCU 上 warp 数必须与 12 lane 的 warp_size 匹配，1 warp 才是对的），`num_stages=2` 与 `tl.range(num_stages=2)` 软件流水都只带来 −0.1% 的噪声级变化。**这个 kernel 在 GCU 上已经是「一页一拍、DTE 串行」的形态，编译器没有、也无法在本形状下重叠页间搬运。**

一句话：**4.945 ms ≈ 0.39 ms 固定 + 13 页 × 0.344 ms/页；每页 0.344 ms 中 65% 是 K 的载入+转置+QK 这一拍，20% 是 online softmax 算术，14% 是 V 的 DMA，PV 的矩阵乘只有 0.6%。prefix 复用值 3 页 ≈ 1.03 ms。**

---

## 2. 环境与设备记录

`--verify` / `--benchmark` 开头自动写入 JSON 的 `env` 段（本次 `verify_*.json` / `bench_20261124T125346Z.json`）：

| 项 | 值 |
|---|---|
| `torch.gcu.current_device()` | 5（脚本内显式 `torch.gcu.set_device(5)`） |
| PCI | `0000:83:00.0`（efsmi PCIE 段 Domain 0000 / Bus 83 / Dev 00 / Func 0；与 `/sys/bus/pci/devices/0000:83:00.0` 一致） |
| 设备名 / 驱动 / AP FW / SP FW | ZIXIAOC200 / 1.10.6 / 1.2.7 / 3.4.21 |
| Mem clock | 7000 MHz |
| 卡前状态 | `efsmi pmon` = no process running；`-d USAGE` = 0.0%；`busy_before=false`，`foreign_pids=[]` |
| 显存 | total 43878764544 B（≈40.9 GiB），测量时约 41.8 GB free |
| CPU 亲和 | `taskset -c 160-191`（记录在 `cpu_affinity_after_pin`） |
| torch / triton / python | 2.10.0+cpu / 3.6.0 / 3.12.13 (`/flagos/bin/python`) |
| GCU 属性 | major 3 / minor 0 / `sip_count` 24 / `max_threads_per_multi_processor` 12 / `multi_processor_count` 2 |

占用检查是脚本硬门：`env_record()` 解析 `efsmi pmon`，凡出现不属于本进程及其祖先进程的 PID 就 `FATAL: device busy` 并退出，不做任何测量。本次三次完整正式运行（12:53:46 / 12:55:26 / 12:56:56 UTC）`foreign_pids` 均为空，测量结束后卡已释放（`pmon` 恢复 `no process running`、usage 0.0%）。

---

## 3. 启动形态与 IR 指令对应

### 3.1 精确 launch/grid（由 `capture_launches()` 在**计时区外**抓取）

| kernel | grid | 关键 constexpr | shared |
|---|---|---|---|
| `_index_block_score_kernel` | (16, 1) | BLOCK_SIZE_Q=64, BLOCK_SIZE_K=128, warps=1 | — |
| `_prefill_topk_rows_enflame` | (128, 1, 1) | TOPK=16, BLOCK_SIZE_N=128, SPARSE_BLOCK=128, BLOCK_SIZE_Q=8, warps=1, stages=1 | 0 |
| `_gqa_sparse_common_prefix` | (64, 1, 1) | GROUP_Q=16 → 1024/16=64, warps=1, stages=1 | 163840 |
| `_gqa_sparse_fwd_direct` | (1024, 1, 1) | BLOCK_SIZE_Q=1, K=128, D=128, H=QH=8, USE_FP8=False, KV_SCALE_MODE=0, warps=1, stages=1, GROUP_Q=16, NUM_GROUPS=64, **USE_SHARED_PREFIX=True** | 65536 |

即 direct 是 **1 query × 1 kv_head × 1 batch = 一个 program**，1024 个 program 完全独立、各自 1 warp（GCU warp_size=12 lane），shared 64 KiB。topk 路径为 streaming 的 `_prefill_topk_rows_enflame`（`row_tile=8`：`max_blocks=128 ≤ 512` 且 `topk=16 ≤ 32`），grid.x = cdiv(1024, 8) = 128。

### 3.2 两个 dot → IR → 源码行

编译缓存（`~/.triton/cache/L7O6KQKMDMNKSZCV7EMAU56OXDH6TEJMBU2JCQT74O2DTAEPANRA`，`num_stages=1`、`shared=65536`，即发布配置；同源码另有 `num_stages=2` 的诊断构建，已在 `cache_candidates` 中列出并排除）：

| 源码 | ttir | gcuir | llir | 机器指令 |
|---|---|---|---|---|
| `sparse_attn.py:286:34` `qk_dot = tl.dot(q_qk, tl.trans(k))` | `%qk_dot_40 = tt.dot %q, %qk_dot` `tensor<8x128xbf16> * tensor<128x128xbf16> -> tensor<8x128xf32>` | `"gcu.matmul"` 第 298 行，**无累加器**，`(memref<8x128xf32>, memref<8x128xbf16>, memref<128x128xbf16>)` | 第 456 行 `llvm.call callee=@tops_convgen_m8_n128_k128_bf16_fp32` | **`tops_convgen_m8_n128_k128_bf16_fp32`** |
| `sparse_attn.py:366:39` `acc_o += tl.dot(p_dot, v)` | `%acc_o_53 = tt.dot %p_dot, %v, %acc_o_50` `tensor<8x128xbf16> * tensor<128x128xbf16> -> tensor<8x128xf32>` | `"gcu.matmul"` 第 671 行，**有累加器**（`operandSegmentSizes = 1,1,1,1,0`，第 4 个操作数 `memref<8x128xf32>`） | 第 952 行 `llvm.call callee=@tops_convgen_m8_n128_k128_bias_bf16_fp32` | **`tops_convgen_m8_n128_k128_bias_bf16_fp32`** |

这就是题目里「两个 dot 生成 `tops_convgen_m8_n128_k128_*`」的完整对应：QK 走非 bias 版本（结果写新 memref），PV 走 `_bias` 版本（把 `acc_o` 作为第 4 个累加操作数直接读入）。两者都是 m=8、n=128、k=128 的 bf16×bf16→fp32 convgen。

### 3.3 页循环（page loop）里的数据搬运

gcuir 外层 `"scf.for"` 在第 168 行（per-query），内层页循环在第 211 行，页体 469 行，其中：

| 类别 | 页体内计数 | 说明 |
|---|---|---|
| `memref.dma_start` | 3 | K 块、V 块、以及页尾 8×f32 状态回写 |
| `memref.dma_wait` | 12 | DTE 等待点，页内串行链的主体 |
| `memref_ext.slice_start` | 3 | 源切片（K/V/索引） |
| `memref_ext.transpose_start` | 1 | **K 的 128×128 转置**（`tl.trans(k)`） |
| `memref_ext.deslice_start` | 1 | 转置结果回填 |
| `memref_ext.broadcast_start` | 4 | 标量/向量广播 |
| `gcu.matmul` | 2 | 上面两个 convgen |
| `math.exp2` | 9 | online softmax 的 `exp2`（`p`、`alpha`） |

整 kernel（含 1024 次外层循环体与收尾）：`memref.dma_wait` 20、`dma_start` 8、`slice_start` 4、`deslice_start` 2、`transpose_start` 1、`gcu.matmul` 2、`math.exp2` 9；llir 侧对应 `tops_wait_dte`、`tops_memcpy_async` 8、`tops_slice_async_2_bf16` 4、`tops_transpose_async_2_bf16` 1、`tops_deslice_async_2_bf16` 2、`tops_syncthreads` 2、`tops_exp2_v4_fp32` 9、`tops_redmaxnumf_v4_fp32` 8、`tops_mac_v4_fp32` 16、`tops_convgen_m8_n128_k128_{,bias_}bf16_fp32` 各 1。

**注意：这些是静态指令条数，不是时间。** 一节里的所有时间结论都来自 event 计时与消融斜率，IR 只用来解释「时间花在哪些指令上」。

---

## 4. 测量方法

- **`--verify`**：stock 输出 vs 本地复制 kernel（MODE 0）vs 关 prefix vs torch fp32 参考；再统计 `shared_count` 直方图。
- **`--benchmark`**：`bench_samples()` 轮转交替顺序，每次调用取一个 `device_fn.Event` 样本（中位数 + **保留全部原始样本**），7 轮。`stock` 与各诊断用同一份输入（同一 `topk_idx`、同一 `block_table`、同一 seed）。
- **绝不混用**：instrumented（monkeypatch / trace）与 uninstrumented 时间分开；`capture_launches()`、`probe_counters()`、`ir_report_for_source()` 全部在计时区外执行。
- **页数扫描**：同一份编译产物，只改 `topk_idx[..., :k]`，k ∈ {1,4,8,12,16}，prefix 关闭使每页都真正走循环体；对每个消融变体同样做 4↔16 页两点斜率，得到该变体的每页边际成本。

---

## 5. 数值一致性

| 对比 | max abs | 非零元素 | 判定 |
|---|---|---|---|
| stock vs 复制 kernel（MODE 0，prefix on） | **0.0** | 0 / 1048576 | 位级一致 |
| 复制 kernel prefix on vs off | 2.44e-4 | 7 / 1048576 | 仅 7 个元素有 1 ulp 级差异，来自不同的 page 加载顺序/累加起点 |
| stock vs torch fp32 参考（12 个代表性行） | 1.45e-4 | — | 与 bf16 精度相符 |
| `num_stages=2` / `tl.range(num_stages=2)` | **0.0** | 0 | 位级一致（纯调度变化） |
| `num_warps=2` | 2.44e-4 | 5 | 数值仍正确但**慢 3.6 倍** |
| `copy_pin`（block_table→0） | 0.2194 | 1048092 | 探针，故意错误 |
| `abl_qk_only` / `abl_no_pv` / `abl_no_softmax` / `abl_no_v_dma` | 0.392 / 6.77 / 0.342 / 0.0835 | 几乎全部 | 探针，故意错误 |

所有输出 `all_finite=true`。`copy_on` 与 stock 位级相同 ⇒ 后续所有消融的相对差都只归因于被删掉的那部分工作。

---

## 6. 组成拆解（本报告的核心）

### 6.1 页数扫描：direct 时间 = 固定 + 页数 × 每页

| k 页 | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| ms | 0.742 | 1.763 | 3.146 | 4.511 | 5.900 |

最小二乘拟合：**斜率 0.34386 ms/页，截距 0.39280 ms，r² = 0.999991**。斜率在 1→16 页范围内完全线性，说明这个 kernel 是**严格串行、一页一拍**的，没有随页数摊销的并行度，也没有随页数增长的额外开销（如 workspace 争用）。

代入实测 13 页（prefix 之后）：13×0.34386 + 0.3928 = **4.863 ms**，与 `direct_reuse_13p` 实测 **5.000 ms** 相差 0.14 ms（2.8%，来自 13 页时 page 分布与 16 页不同、以及 kernel 尾部的归约/写回）。

### 6.2 每页 0.344 ms 的构成（同页数边际成本，不是绝对时间差）

| 变体 | 保留的工作 | 每页边际 (ms) | 相比 full 省下 (ms/页) | 占每页 |
|---|---|---|---|---|
| `full`（MODE 0） | 全部 | **0.3444** | — | 100% |
| `qk_only`（MODE 1） | K 载入 + 转置 + QK dot | 0.2237 | 0.1207 | 35% |
| `qk_softmax`（MODE 2） | QK + 在线 softmax，无 V 载入/PV | 0.2928 | 0.0516 | 15% |
| `no_v_dma`（MODE 4） | 两个 dot + softmax，V 换成已载入的 K | 0.2950 | 0.0494 | 14% |
| `both_dots`（MODE 3） | 两个 dot + V 载入，无 softmax 算术 | 0.3030 | 0.0414 | **不参与推导**（见下） |

由此得到每页增量预算（以 full 为基准，各探针只删一项）：

- **K 载入 + 转置 + QK convgen：0.2241 ms / 页 = 65.1%**
- **online softmax 算术（`max`/`exp2`/`alpha` 缩放/`l_acc`）：0.0688 ms / 页 = 20.0%**
- **V 的 DMA + 对应 DTE 等待：0.0493 ms / 页 = 14.3%**
- **PV convgen：0.0022 ms / 页 = 0.6%**
- 合计 **0.34437 ms**，与实测 full 的 0.34437 ms **残差 0.00000**。

推导用的恒等式（同页数、同编译产物、同 page 分布）：

```
softmax      = qk_softmax − qk_only   = 0.29283 − 0.22399 = 0.06884
V DMA + PV   = full       − qk_softmax = 0.34437 − 0.29283 = 0.05154
PV           = no_v_dma   − qk_softmax = 0.29500 − 0.29283 = 0.00215
V DMA        = full       − no_v_dma   = 0.34437 − 0.29500 = 0.04931
K载入+转置+QK = qk_only                 = 0.22399
```

两条独立路径交叉验证：(V DMA + PV) 由 `full − qk_softmax` 得 0.05154，由 `V + PV` 相加得 0.05146，一致到 8e-5 ms。

**关键判断 1：PV 的矩阵乘只有 0.6%，可以视作免费；成本压在 K 那一拍。** 注意 QK 的 convgen 与 K 的载入、转置被合并在 0.2241 ms 里**没有分开**（§8.6），所以不能说「两个 MMA 都很便宜」——能说的是 **PV 便宜到可忽略**，而 K 路径即使全部算成搬运也有 0.224 ms。因此这不是「算力不够」，而是**数据搬运 + 一次强制转置**的串行链。

**关键判断 2：`no_v_dma`/`both_dots` 一类探针不能简单相减。** `both_dots`（0.30266）比 `qk_softmax`（0.29283）高 0.0098，若按「保留两个 dot、删 softmax」直读会得出 PV 约 0.01 ms；实际它删掉 softmax 后 `p` 变成 raw logit 的 cast，PV 的输入分布已不同，因此**不满足 telescoping**，残差 0.0271 ms（见 JSON `mode3_residual_ms`、`mode3_used=false`）。本报告只用上述四条恒等式，`both_dots` 仅作旁证。

### 6.3 与题面 4.945 ms 的对照

题面参考值：整链 6.945 / Attention 5.275 / prefix 0.318 / direct 4.945 ms。本次同卡同 shape 实测：整链 6.781 / sparse_attn 5.427 / prefix 0.372 / direct 4.999 ms。整链低 0.16 ms、prefix 高 0.05 ms、direct 高 0.05 ms，量级与离散度一致（`stock` 的 7 个样本有 5.41–5.62 的抖动，首样本偏慢，故 `copy_on` 5.364 比 `stock` 5.427 低 0.06 ms —— 这 0.06 是 host 编排/首样本效应，不是 kernel 差异；两者输出位级相同）。拆解结论对 4.945 与 4.999 同样成立：**都是 13 页 × 约 0.344 + 约 0.39 固定**。

---

## 7. 对照实验：prefix 打开/关闭到底回答了什么

- **回答了什么**：跨 query 的 prefix 复用**当前只值 3 页**。`shared_count` 直方图 64 个 group 全部为 3（`{0:0, 1:0, 2:0, 3:64, 4:0}`，均值 3.0）。因为 `_gqa_sparse_common_prefix` 里 `for slot in range(min(4, K))` 只吸收最多 4 页，而本用例严格顺序前缀下前 3 页（384 token）在所有 query 间字节相同。于是 prefix on 让 direct 从 16 页降到 13 页，省 3×0.344 ≈ 1.03 ms，代价是 prefix kernel 0.372 ms，**净收益约 0.66 ms**；实测端到端 5.908 → 4.999（−0.909 ms），差值主要来自 13 页时 page 分布改变（少了 3 页的 DRAM 访问）。
- **没回答什么**（必须在报告里说清）：
  1. 它**不能**把「K/V 的 DRAM 流量」「转置」「DTE 等待」「matmul」「softmax」分开——它只是改变有多少页进入循环。
  2. 它**改变 page 加载分布**（少 3 页 ⇒ 少 3 次随机寻址），所以 on/off 的差不是纯「复用收益」。
  3. 因此真正的分解靠 6.2 的 MODE 消融（同页数下的边际成本），prefix on/off 只用来量化「复用值几页」。
- **`copy_pin`**：`block_table` 全置 0，指令流、掩码、计算完全不变，只把 KV 地址压进 256 KB 热区。结果 **5.364 → 5.323 ms（−0.8%）**。这直接说明**随机页寻址的 DRAM 侧代价在本形状下几乎为零**，从而排除「direct 慢在 DRAM 带宽/随机访问」这一假设；也说明 `copy_off` 的 +0.55 ms 不是 DRAM 带宽造成的，而是多了 3 页串行链。
- **`num_stages=2` 与 `tl.range(num_stages=2)` 软件流水**：都属于「零成本/零收益」——0.0 数值差、−0.1% 时间差（5.364 → 5.359/5.364）。这是重要负面结论：**在当前 GCU 后端 + 本形状下，编译器无法把页间 K/V 预取与上一页计算重叠**。原因与 6.2 的 65% 集中在「K 载入+转置+QK」这一串行拍一致：转置（`tops_transpose_async_2_bf16`）与 QK convgen 之间有 `dma_wait` + `gpu.barrier` 依赖，页内已经串行，页间没有多余 buffer（shared 64 KiB 已用满）。
- **`num_warps=2`**：19.16 ms（慢 3.6×）。GCU 的 `warp_size=12`、`max_threads_per_multi_processor=12`，1 warp 正好铺满 12 lane；2 warp 会让 24 lane 争 12 个线程槽，反向印证「本 kernel 已按 12 lane 最优配置」。

---

## 8. 未解释/未覆盖的部分（诚实边界）

1. **没有 GCU kernel 级硬件计数器。** 本卡 `efsmi dmon --gpm-metrics 2,3,18,19,30,31,32,33` 在**阳性对照**（4096×4096×4096 bf16 matmul 持续打满 6 秒）下 12 个采样点全为 0.0，MSA 全链同样全 0（`hardware_counters.positive_control.nonzero_samples = 0`）。⇒ 计数器在本卡/本驱动（1.10.6）下不工作，**没有被用来做任何归因**。这是本次最大的证据缺口：无法直接读出 SIP 利用率、DRAM 读写字节数。
2. **clock 内建函数不可用。** `gcu.begin_clock/end_clock` 能 lower 成 `tops_clock_begin/end`，但运行时返回值恒为 0（试过 N=1/200/2000 与 int32 截断），无法做 kernel 内分段计时。
3. **`torch.profiler` 无 kernel 事件。** `ProfilerActivity.PrivateUse1` 可用但 trace 里没有 kernel duration，只能拿到 host 侧。
4. **静态 IR 条数不等于时间。** 3.3 的计数只说明「页内有 12 个 DTE 等待点、1 次转置、2 个 convgen」，不说明各占多少毫秒；时间结论全部来自 event 计时 + 消融斜率。
5. **消融探针改变工作量，不是「免费开关」。** MODE 1/2/4 都会让编译器删掉下游消费者、可能改变调度与寄存器压力，因此它们的**绝对**时间不可直接相减；本报告只用**同页数下的斜率差**（4↔16 页），这在同一编译产物、同一页面分布下才可比。斜率法的残余误差由 r²=0.99999 与各变体 `fixed_ms` 的稳定性（0.360–0.405 ms）佐证。特别地 `both_dots`（MODE 3）**不满足 telescoping**（残差 0.0271 ms），已在 §6.2 说明并排除出预算推导——这是「探针本身会改变被测量」的一个实例，而不是可以忽略的噪声。
6. **未拆到「转置」单项。** 没有单独做「去掉 `tl.trans`」的探针（去掉会改变 QK 的语义/形状，无法保持同 kernel 可比），所以「K 载入」与「K 转置」在 0.224 ms 里仍未分开；只能说二者合计 65%。
7. **未做跨卡/跨驱动验证。** 全部数据来自 GCU 5 单卡；同卡同 shape 三次**完整独立**运行的重复性如下（各自 7 轮 event 采样，全流程重新编译/重新构造输入）：

   | 运行（UTC） | direct 13 页 | 斜率 ms/页 | 每页预算（K / softmax / V / PV） | 残差 | stock |
   |---|---|---|---|---|---|
   | 12:53:46 | 4.9997 | 0.34386 | 0.2241 / 0.0688 / 0.0493 / 0.0022 | 0.00e+00 | 5.4268 |
   | 12:55:26 | 4.9980 | 0.34432 | 0.2241 / 0.0688 / 0.0493 / 0.0022 | 0.00e+00 | 5.4322 |
   | 12:56:56 | 5.0067 | 0.34455 | 0.2236 / 0.0690 / 0.0494 / 0.0024 | 0.00e+00 | 5.4263 |

   斜率离散度 0.7‰、预算各项离散度 <2%，重复性足以支撑 §6 的结论。但没有第二张卡或第二个驱动版本对照，**不能外推**到其他 S60 机器或驱动。

---

## 9. 模型接入状态

**未接入 / 仅实验。** 本任务的所有 kernel 变体（MODE 1–4、`PIPELINE`、`num_warps=2`）都只存在于 `agent-results/msa008_direct_pipeline.py` 这份独立脚本里，通过 `make_dispatch()` 本地复刻 host 编排来启动；`src/flag_attn/**` 与 `benchmark/`、`tests/` **一行未改**，基线默认 dispatch 未动。`copy_on` 与 stock 位级一致只证明「我的复刻是忠实的」，不构成任何性能改动建议。

从本次数据能给出的、**仍需正式 gate 验证**的方向（不在本任务落地）：
- 页间软件流水在当前后端无效（§7），若要走这条路需要改 shared 预算或改用双缓冲 DMA，属于 kernel 重写，风险高。
- 每页 65% 在「K 载入+转置+QK」，其中转置是独立的一拍 `tops_transpose_async_2_bf16` 且紧跟 `dma_wait`+`barrier`；把 K 以「已转置」布局写入 paged cache（即 `[page, head, d, pos]`）可省掉这次转置，但会改动 KV cache 布局约定，属接口级改动。
- prefix 只吸收 3 页（`min(4, K)` 与顺序前缀共同决定）；若把吸收页数上限提高，direct 页数可继续下降，每多吸收 1 页净收益约 0.344 − （prefix kernel 边际成本）。prefix kernel 当前 0.372 ms / 3 页 ≈ 0.124 ms/页，**理论上吸收更多页仍划算**，但需验证 prefix kernel 的扩展性与数值稳定性。

---

## 10. 复现

```bash
ssh -i /home/dsagent/.ssh/id_ed25519 -p 22226 gcuagent@127.0.0.1
cd /workspace/agent-workspaces/msa-stage-profile/.cache/agent-workspaces/msa-long-prefill-direct-pipeline-984069d

# 数值校验（stock vs 复制 kernel vs 关 prefix vs torch fp32）
PYTHONPATH="$PWD/src:$PWD:/opt/flagtree:$PYTHONPATH" \
  taskset -c 160-191 /flagos/bin/python agent-results/msa008_direct_pipeline.py --device 5 --verify

# 非侵入式成对采样 + 分段 profile + IR 普查 + 启动形态
PYTHONPATH="$PWD/src:$PWD:/opt/flagtree:$PYTHONPATH" \
  taskset -c 160-191 /flagos/bin/python agent-results/msa008_direct_pipeline.py --device 5 --benchmark --rounds 7
```

JSON 落在 `agent-results/raw/bench_<UTC>.json` 与 `verify_<UTC>.json`，包含 `env`、`launch_meta`、`end_to_end_ms`（每个变体的全部样本）、`full_chain_ms`、`staged_ms`、`per_page_marginal`、`page_sweep`、`numeric_diffs_vs_copy_on`、`hardware_counters`（含阳性对照）、`ir_census`（含 `dot_mapping`、`landmarks`、`cache_candidates`）。

**证据边界一句话**：本报告的每一个时间数字都来自 `device_fn.Event` 的成对/交替采样，硬件计数器与 kernel 内 clock 在本卡均不可用（已用阳性对照证明），静态 IR 只用于解释、不用于计时；所有自测均为诊断，最终裁决以 orchestrator 正式 benchmark 的独占 gate 为准。
