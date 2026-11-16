# SAGE-006：SAGE-005 审查问题的证据修复与报告纠正

**任务范围。** 在独立克隆 `/workspace/agent-workspaces/sage-review-fix-5`（基线提交
`167c360085ffe46371626eeb108d17ea66fb92bf`，分支 `agent/enflame-sage-review-fix-5`）中
修复 SAGE-005 的审查问题：补齐或降级缺失的原始证据、纠正错误的数值与正式检查状态、
区分观察与推断。**未修改 `src/` 下任何文件，未改动默认路由，未推送、未合并。**

| 项目 | 值 |
| --- | --- |
| 克隆 | `/workspace/agent-workspaces/sage-review-fix-5` |
| 基线提交 | `167c360085ffe46371626eeb108d17ea66fb92bf` |
| 交付提交 | 本分支末端提交（`git log -1`），父提交即上述基线 |
| 分支 | `agent/enflame-sage-review-fix-5` |
| 设备 | 仅 GCU 5，PCI `0000:83:00.0`，SN `A0D5460520864` |
| CPU | `taskset -c 160-191` |
| 栈 | torch 2.10.0+cpu / torch_gcu 2.10.0+3.7.20260408 / Triton 3.6.0 / Python 3.12.13 |

GCU 6（`84:00.0`）与 GCU 7（`a3:00.0`）全程未触碰。所有脚本在第一次设备操作前用
`efsmi` 同时校验 **PCI 地址与序列号**，不匹配即退出，不回退到别的卡。

---

## 0. 审查意见与处理结论（逐条）

| # | 审查意见 | 处理 | 结果 |
| --- | --- | --- | --- |
| 1 | 纯 FP16 的 3.84x / 3.63x 复现只有 `/tmp` 脚本和事件日志，原始 JSONL 未入库，报告却声称原始证据完整 | `/tmp` 证据仍在，脚本已恢复入库；并用 `sage005_recheck.py --probe fp16` 重新产出逐样本 JSONL | **已修复**：数值复现（3.74x / 3.62x），原始数据入库，来源如实标注 |
| 2 | 两个被否决的优化同样缺原始数据 | 恢复 `sage005_fuse.py` / `sage005_flags.py`，并重跑为 `sage006_recheck_fuse.jsonl` / `sage006_recheck_flags.jsonl` | **已修复**，其中 flags 一节原结论**有误**，见 §3.3；fuse 一节当年在打印等价性前就崩了（`outs[n]` 是 tuple），重跑补上 |
| 3 | §11 称 orchestrator 准确性检查失败，而 SAGE-005 正式检查实为 42 passed、exit 0 | 用 `checks/20261116T054641Z-38b71c5c/record.json` 替换；旧的本地预检标注为「已作废」 | **已纠正**，见 §6 |
| 4 | 要求区分观察与推断 | 全文新增观察/推断标签，§5 单列推断清单 | **已落实** |
| 5 | 优先证据修复，不重复长时间性能实验 | 只做短复跑（单项 0.5-3 分钟，共 7 组），未重跑 A/B、冷编译、长序列实验 | **已遵守** |

**结论没有翻转。** 修复后仍然成立的核心结论是：SageAttention 在 GCU S60 上比设备原生
fused SDPA 慢约 3.3-3.9x，差距可被一个不含 INT8 的纯 FP16 Triton kernel 复现，因此
瓶颈在 Triton `gcu300` 的代码生成，而不在 Sage 的算子逻辑；两项候选优化仍应否决。
被**纠正**的是五处：手动预取的方向写反了、D128 单波斜率偏高、opcode 数与
算子数被子串统计放大、`qk_scale_before_dot` 的代价被剔除后低估、以及 §11 把
本地预检当成了正式门（后者结论完全相反：正式门是通过的）。逐条见 §4。

---

## 1. 证据分级

本报告对每一条数值标注来源等级，不再使用「原始证据完整」这类笼统表述。

| 等级 | 含义 | 文件 |
| --- | --- | --- |
| **A 已入库原始数据** | SAGE-005 当时即写入仓库的逐样本 JSONL，带 `.exit` | `raw/sage005_ab_ab.jsonl`、`sage005_dispatch.jsonl`、`sage005_cold_memory.jsonl`、`sage005_micro_ceiling.jsonl`、`sage005_micro_slope.jsonl` |
| **B 本次复跑原始数据** | SAGE-006 重新执行并写入的逐样本 JSONL，带 `.exit` 与脚本 | `raw/sage006_recheck_*.jsonl`、`raw/sage006_artifacts.jsonl` |
| **C 会话日志恢复** | 从 SAGE-005 会话 `events.jsonl` 逐字恢复的 stdout（脚本同时恢复），**不是**当年的逐样本 JSONL | `raw/sage005_recovered/*.log`、`scripts/recovered/*.py` |
| **D 不可恢复** | 无 | — |

`/tmp` 证据的实际状态：SAGE-005 结束后 `/tmp/sage005_*.py`、IR dump 与反汇编仍在
worker 上，已全部取回。**唯一确实丢失的是当年的逐样本 JSONL**（脚本只在 stdout 打印
中位数，未落盘），这部分用等级 C 的逐字 stdout + 等级 B 的复跑来补，而不是假装它存在。

### 1.1 报告断言 → 证据对照

| 报告断言 | 数值 | 等级 | 证据 |
| --- | --- | --- | --- |
| 纯 FP16 复现 3.84x / 3.63x | 1.0137 / 4.2598 ms | C | `sage005_recovered/fp16attn.log`；复跑 B：`sage006_recheck_fp16.jsonl` |
| 正式 A/B 0.7235 / 0.4428 | 3 轮 x 24 对 | A | `sage005_ab_ab.jsonl` + `_summary.json`（`status: PASS`） |
| 派发路径与 host 拷贝 | 0.2745 ms / 9 次拷贝 | A | `sage005_dispatch.jsonl` |
| `num_stages` 扫描 | 28.05 … 24.10 TF | C | `sage005_recovered/stage2.log`；复跑 B |
| 手动双缓冲预取 | 186-187 us vs 173-175 us | C | `sage005_recovered/stages_qk.log`；复跑 B，**原文方向写反** |
| 单波边际斜率 | D64 8.02 / 15.21 us | C | `sage005_recovered/marginal.log`；复跑 B |
| 带宽 450-459 / 425 GB/s | 16 KiB 块 | C | `sage005_recovered/bw.log`；复跑 B |
| 波量化 | 0.565 / 1.044 / 1.043 / 2.001 ms | C | `sage005_recovered/bh.log`；复跑 B |
| 微基准天花板/斜率 | 67.78 / 82.98 TF | A | `sage005_micro_ceiling.jsonl`、`sage005_micro_slope.jsonl` |
| 冷编译 8862 / 7940 ms | 对比 5.5 / 8.5 ms | A | `sage005_cold_memory.jsonl` |
| 精度 12 行表 | rel L2 ~1.2e-2 | C（JSON 已恢复） | `raw/sage005_recovered/sage005_acc.json` |
| 反汇编统计 | 18,861 / 173 opcodes | C + B | `raw/sage005_artifacts/fa_bf16_256.s`；`sage006_artifacts.jsonl` |
| 正式准确性门 | 42 passed, exit 0 | A（orchestrator 记录） | `raw/sage006_formal_gate_record.json` + `_stdout.log` |

---

## 2. 这次补进仓库的东西

### 2.1 脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/sage005_guard.py` | 共用件：PCI+SN 双重校验、事件计时、JSONL 写出、仓库根解析 |
| `scripts/sage005_recheck.py` | SAGE-006 短复跑，`--probe fp16,fuse,flags,stages,waves,bw,marginal`，`--jsonl` 必填 |
| `scripts/sage006_artifacts.py` | 从 IR/反汇编**重新计算**算子计数、async 等待距离、符号大小与哈希；不加载设备 |
| `scripts/recovered/` | SAGE-005 当年 `/tmp` 下的 49 个探针脚本（其中 3 个是 fuse 的 kernel 变体），逐字保留 |
| `scripts/recovered/sage005_micro_variants/` | `sage005_micro.py` 生成的 9 个文件：3 个微基准 kernel（resident / tilemajor / epilogue）+ 6 个 shipped kernel 的消融变体（full、no_row_max、no_sub、both_dots、no_pv_dot、qk_matmul_only） |

`recovered/` 里的脚本保留原始形态（硬编码 `set_device(5)`、无 JSONL 写出），用于审计
当年到底跑了什么；它们**不建议直接复用**，可复现入口是 `sage005_recheck.py`。

### 2.2 原始数据

| 文件 | 内容 | 行数 |
| --- | --- | ---: |
| `raw/sage006_recheck_fp16.jsonl` | 两个 headline 形状的 FP16 Triton vs 原生 | 3 |
| `raw/sage006_recheck_fp16_fuse.jsonl` | 首次合并运行的残留（`.exit` 为 1，**不引用**，见 `raw/sage006_REPRODUCE.md`） | 3 |
| `raw/sage006_recheck_fuse.jsonl` | 两种行最大值改写 + 逐位相等性 | 11 |
| `raw/sage006_recheck_flags.jsonl` | 3 个布尔开关的全组合 | 17 |
| `raw/sage006_recheck_stages.jsonl` | row-major / tile-major x stages 1-4 | 25 |
| `raw/sage006_recheck_waves.jsonl` | 16-192 CTA 固定工作量 + `block_m` 对比 | 11 |
| `raw/sage006_recheck_bw_marginal.jsonl` | 带宽 + 单波边际斜率 | 42 |
| `raw/sage006_artifacts.jsonl` | IR 算子计数、async 等待距离、反汇编直方图、哈希 | 9 |
| `raw/sage005_recovered/*.log` | 当年 stdout 逐字恢复（含命令行），11 次运行 | 11 |
| `raw/sage005_artifacts/*` | GCUIR/TTGIR/LLIR 全文 + 18,863 行反汇编 | 4 |
| `raw/sage006_formal_gate_*` | 正式准确性门的 record / stdout / exit | 3 |

---

## 3. 复跑结果（等级 B）

所有复跑都在 GCU 5、CPU 160-191 上完成，每次运行的 `env` 记录都带当时负载
（1 分钟负载 25.3-44.1，有外部租户），因此下列数值只作**一致性核对**，
不作为新的性能主张。

### 3.1 纯 FP16 复现（审查点 1）

| 形状 | SAGE-005 (C) | SAGE-006 (B) | 原生 (B) | 比值 (B) | 最大绝对误差 (B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x2x4096x64, BM=256, grid=32 | 1.0137 ms | 1.0124 ms | 0.2705 ms | **3.74x** | 6.10e-05 |
| 4x32x1024x128, BM=128, grid=1024 | 4.2598 ms | 4.2508 ms | 1.1730 ms | **3.62x** | 9.77e-04 |

同一脚本、同一 tile 几何、同样 `num_warps=1, num_stages=1`，两次结果相差 < 0.3 %，
比值差异来自原生 op 的 2 % 抖动。**「不含 INT8 也能复现差距」这一结论成立。**

### 3.2 `num_stages` 与手动预取（审查点：§3 措辞）

原报告写「manual double-buffered K prefetch 把 186-187 us 移到 173-175 us，约 7 %」。
恢复的 stdout 显示方向相反，复跑同样相反：

| K 布局 | 循环 | stages=1 | 2 | 3 | 4 |
| --- | --- | ---: | ---: | ---: | ---: |
| tile-major 128x128x128, 24 CTA | plain | **28.68 TF** | 28.18 | 24.64 | 24.06 |
| tile-major 128x128x128, 96 CTA | plain | **39.23 TF** | 39.74 | 33.50 | 32.53 |
| row-major 256x128x64, 24 CTA | plain | 176.4 us | 174.7 | 177.6 | 174.6 |
| row-major 256x128x64, 24 CTA | manual_prefetch | 187.3 us | 188.1 | 186.9 | 189.0 |

**纠正**：手动双缓冲预取不是快 7 %，而是**慢 6-8 %**；`num_stages` 1→2 无变化，
3→4 反而退化。结论「Triton 在这条循环上没有可用流水线」不变，但原报告把预取的方向
写反了，已修正。

### 3.3 两个被否决的优化（审查点 2）

**行最大值广播减法改写（D64）**：

| 变体 | 中位数 | 相对 base | 与 base 逐位相等 |
| --- | ---: | ---: | --- |
| base | 1.0333 ms | 1.00 | — |
| `fused_expr`（复用 `p32`） | 1.0285 ms | 0.995 | 是 |
| `inplace_sub`（`qk -=`） | 1.0277 ms | 0.995 | 是 |

D128 同样在 ±0.1 % 内且逐位相等。**否决结论保留**：改动只值 0.5 %（噪声级），
而且逐位相等说明编译器本来就只发一次广播减法，没有可消除的重复计算。

**三个布尔开关的全组合（原报告此处有实质性错误）**：

| 形状 | `qk_scale_before_dot=True` | `=False` | 代价 | 输出最大差 |
| --- | ---: | ---: | ---: | ---: |
| D64 | 1.0248-1.0335 ms | 1.3461-1.3499 ms | **+31.4 %** | 3.66e-04 |
| D128 | 3.8880-3.8924 ms | 4.0986-4.1011 ms | **+5.4 %** | 9.77e-04 |

原报告写「D64 1.0274-1.0431、D128 3.9031-3.9062，即 < 1.5 % 摆幅」，是把
`qk_scale_before_dot=False` 的四行**剔除后**再取区间。真实情况是：`predequantize_k` 与
`tile_major_predequant_k` 确实几乎无影响（≤ 0.85 %），但 `qk_scale_before_dot=False`
会让 D64 慢 31 %，并且改变数值结果。

**这不改变否决结论**（三个开关都不该动默认值），但原报告「< 1.5 % 摆幅」的说法
必须撤回，已改为「两个 K 布局开关 ≤ 0.85 %；`qk_scale_before_dot` 不是免费开关」。

### 3.4 单波边际斜率（原报告 D128 数字偏高）

原脚本对 D128 采到 `T=3584/4096`，此时 `nCTA=28/32` 已进入第二波，斜率被波量化污染，
得到 24.63 / 24.10 us 两个离群点，把均值抬到 15.38。复跑把两点剔除（`nCTA ≤ 24`）：

| D | 路径 | SAGE-005 (C) | SAGE-006 (B) | 每 128-KV tile 倍数 |
| --- | --- | ---: | ---: | ---: |
| 64 | native | 8.02 (7.12-8.88) | 8.26 (7.61-9.00) | — |
| 64 | Sage | 15.21 (14.88-15.47) | **14.91** (14.25-15.25) | **1.81x** |
| 128 | native | 10.49 (9.31-12.33) | 9.84 (9.33-10.73) | — |
| 128 | Sage | 15.38 (10.60-24.10) | **11.06** (10.72-11.36) | **1.12x** |

**纠正**：D128 的「每 tile 1.5-1.9x」不成立，真实是 1.12x；D64 的 1.81x 成立。
报告 §4 的「Sage 斜率与深度无关、稳定在 ~15 us」只对 D64 成立。

### 3.5 波量化与带宽（与原文一致）

| nCTA | 波数 | 复跑中位数 | 原文 |
| ---: | ---: | ---: | ---: |
| 16 | 0.67 | 0.5613 ms | 0.5651 |
| 32 | 1.33 | 1.0280 ms | 1.0438 |
| 48 | 2.00 | 1.0440 ms | 1.0427 |
| 96 | 4.00 | 1.9775 ms | 2.0009 |
| 192 | 8.00 | 3.8879 ms | 3.8519 |

带宽：`torch copy_` 431.8 GB/s；2 KiB 块 115.9-116.3（copy）/ 102.4-103.2（read）；
16 KiB 块 424.3-425.6 / 430.3-450.7；32 KiB 块 423.4-426.8 / 431.0-459.7 GB/s。
原报告「读 450-459、拷贝 425、2 KiB 仅 103-116」成立。

---

## 4. 报告纠正清单

| 位置 | 原文 | 订正 | 依据 |
| --- | --- | --- | --- |
| §2 | 3.84x / 3.63x | 保留为等级 C；新增复跑 3.74x / 3.62x | §3.1 |
| §3 | 「预取把 186-187 us 移到 173-175 us，约 7 %」 | 方向写反：预取**慢** 6-8 % | §3.2 |
| §3 | 「8 `gcu.mfence` 围绕 10 `gcu.matmul`」 | 算子级是 **4 / 4**；10/8 是子串计数（`triton_gcu.matmul` 边界、`gcu.mfence_type` 属性） | `sage006_artifacts.jsonl` |
| §4 | 微基准的 `gcu.matmul` 列 10/9/6 | 保留，但注明是子串计数（约为算子级的 2.5x），仅用于证明消融未删掉 dot | 同上 |
| §3 | 「**没有** async copy」 | LLIR 有 24 个 async 调用点、24 个 `tops_wait_dte`；但每次拷贝后仅 1-12 条指令即等待，**没有跨迭代重叠** | 同上 |
| §4 | D64 15.21 / D128 15.38 us per tile | D64 14.91（1.81x）；D128 **11.06**（1.12x，剔除越波点） | §3.4 |
| §5 | 波量化、带宽 | 复跑一致，保留 | §3.5 |
| §6 | 「173 distinct opcodes」 | **47**；173 来自正则把操作数十六进制当成助记符。18,861 个指令槽中 17,364 条可解码、1,497 条 `<unknown>`；nop 13,730 | 同上（`bf16_256`；另三个实例 46/43/43） |
| §7 | 「D64 1.0274-1.0431、D128 3.9031-3.9062，< 1.5 %」 | 该区间剔除了 `qk_scale_before_dot=False`；该开关实际代价 +31.4 % / +5.4 % 并改变数值 | §3.3 |
| §7 | 改写「within noise」但无等价性数据 | 补 `bitwise_equal=true`、maxabs 0 | §3.3 |
| §11 | 「exit 1, 40 passed, 2 failed」「orchestrator 不设 PYTHONPATH」 | **exit 0, 42 passed**；orchestrator 确实导出 `PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"`（`orchestrator/core.py:405`） | §6 |

---

## 5. 观察与推断

**观察（有原始数据支撑）**

- 纯 FP16 Triton kernel 与含 INT8 的 Sage kernel 落在同一性能档位，二者都显著慢于原生 op。
- GCUIR 里每个 tile 循环含 4 个 `gcu.matmul`、4 个 `gcu.mfence`、51 次 `memref.alloc`、
  42 次 `memref.dealloc`，`gcu.dma` 与 `memref.copy` 为 0。
- LLIR 的 24 个 async 调用点在 1-12 条指令内就跟一个 `tops_wait_dte`
  （19 处仅隔 1 条、2 处隔 2 条、1 处隔 9 条、2 处隔 12 条）。
- tile-major QK 循环在 24 CTA 下 28.7 TF，而双操作数常驻的纯 `tl.dot` 达 67.8 TF，
  加入每迭代 tile 装载后掉到 28.3 TF。
- `num_stages` 1-4 无效，手动双缓冲预取反而慢 6-8 %。
- 波量化明显：1.33 波与 2.00 波耗时相同。

**推断（不可直接观察，明确标注）**

- 原生 kernel 用静态调度的 nop 槽位编码了双缓冲流水线，因此能把 tile 装载藏在
  matmul 之后。
  依据是 13,730 个 nop 与 1,136 条 `l.st.w` 的分布，但**厂商源码不可得**，该调度结论是推断。
- Triton `gcu300` 后端没有为这条循环生成跨迭代流水线；async 拷贝存在但立即等待，
  等价于同步拷贝。这是从 IR 形态 + `num_stages` 无效 + 预取变慢三者**共同推断**，
  不是后端源码级的证明。
- 「修好后当前 3.3-3.9x 可能腰斩」是**外推**，没有任何测量支撑，不应作为承诺。

---

## 6. 正式准确性检查（纠正 §11）

SAGE-005 的正式检查由 orchestrator 在会话结束后自动执行，**已通过**：

| 项 | 值 |
| --- | --- |
| check id | `20261116T054641Z-38b71c5c` |
| kind / name | `correctness` / `accuracy` |
| commit | `167c360085ffe46371626eeb108d17ea66fb92bf` |
| device / pci / cpu | 5 / `83:00.0` / `160-191` |
| exit code | **0** |
| stdout 末尾 | `42 passed, 1 warning in 55.69s` |
| 证据 | `raw/sage006_formal_gate_record.json`、`raw/sage006_formal_gate_stdout.log` |

`record.json` 里逐字记录的 `command` 数组只有下面第二行的 `timeout ...`；
`cd` 与 `export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"` 是 orchestrator 在
启动子进程前设置的工作目录与环境，**不在数组内**（`orchestrator/core.py:405`，
见本节纠错第 2 条）。`record.json` 同时记录基线
`087a3621bbfeb843d9efe2445b67faa7f1ed46ec` 与分支
`agent/enflame-sage-fused-sdpa-study-5`。完整形式为：

```
cd <workspace> && export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH" && \
  timeout --signal=TERM --kill-after=10s 1800s taskset -c 160-191 \
  /flagos/bin/python -c "import os, torch, torch_gcu, pytest; \
  os.environ['SAGE002_TEST_DEVICE']='gcu:5'; torch.gcu.set_device(5); \
  raise SystemExit(pytest.main(['tests/flag_attn/test_sage_attention.py', \
  'tests/flag_attn/test_sage_bottleneck.py','-q']))"
```

原报告 §11 的三处错误：

1. 把**本地手工预检**（`raw/sage005_formal_gate.txt`，exit 1、40 passed、2 failed）
   当成了 orchestrator 的正式检查。该预检的命令行没有 `PYTHONPATH`，两个测试因此
   无法 import `flag_attn`；它从来不是验收结果，现标注为「已作废的本地预检」。
2. 由此推出的「orchestrator 不设 PYTHONPATH」不成立——`core.py:405` 明确导出。
3. 声称检查跑在 `6b2487b`，与正式记录的 `167c360` 不符。

`src/` 与 `tests/` 在本任务中零改动（`git diff --name-only 167c360 HEAD` 仅列出
`agent-results/*`）。**本报告仍不是验收本身；验收以 orchestrator 的正式检查为准，
而该检查在 SAGE-005 上已经以 42 passed / exit 0 通过。**

---

## 7. 复现方式

```bash
# 设备预检
bash agent-results/scripts/preflight_gcu5.sh

# 短复跑（本次入库数据的来源）
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe fp16 \
  --jsonl agent-results/raw/sage006_recheck_fp16.jsonl
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe fuse \
  --jsonl agent-results/raw/sage006_recheck_fuse.jsonl
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe flags \
  --jsonl agent-results/raw/sage006_recheck_flags.jsonl
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe stages \
  --jsonl agent-results/raw/sage006_recheck_stages.jsonl
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe waves \
  --jsonl agent-results/raw/sage006_recheck_waves.jsonl
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage005_recheck.py --probe bw,marginal \
  --jsonl agent-results/raw/sage006_recheck_bw_marginal.jsonl

# 产物统计（不占设备）
/flagos/bin/python agent-results/scripts/sage006_artifacts.py \
  --ir agent-results/raw/sage005_artifacts/sage005_attn_gcuir.txt \
       agent-results/raw/sage005_artifacts/sage005_attn_ttgir.txt \
       agent-results/raw/sage005_artifacts/sage005_attn_llir.txt \
  --vendor --sha256 --jsonl agent-results/raw/sage006_artifacts.jsonl
```

反汇编统计需要 `llvm-objcopy` / `clang-offload-bundler` / `llvm-objdump`
（`/opt/tops/bin`）与宿主的 `nm`；脚本会先解包 `.tops_fatbin`，再反汇编，再按
「制表符后的第一个 token」取助记符——这一点是 47 vs 173 的根因。

---

## 8. 没有改变的东西

- `src/` 与 `tests/`：零改动。
- 默认路由与默认参数：零改动。Sage 仍是 opt-in 路径，原生 fused SDPA 仍是设备基线。
- 两项候选优化：仍然否决（行最大值改写 ≤ 0.5 % 且逐位相同；K 布局开关 ≤ 0.85 %，
  且 kernel 并非带宽受限）。
- 性能结论的定性部分：不变。定量部分按 §3、§4 订正。

**这是一个负结果报告**：在 GCU S60 上，SageAttention 的 fused SDPA 比设备原生 op 慢
3.3-3.9x，且没有任何 Triton kernel 内的改动能弥合差距——损失在代码生成，不在算子。
