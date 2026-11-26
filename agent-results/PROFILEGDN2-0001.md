# PROFILEGDN2-0001：GDN2 前向的 K1/K2 时间归属，以及 V tile 并行度与状态读写的分离

## 结论（Verdict）

在**唯一允许使用的 GCU 5（PCI `83:00.0`）**上，以本轮固定基线
`1320878ddada3b3d29738d1e5f5c0bcc9385def7` 为对照，本任务把一次融合前向
`chunk_gdn2` 的耗时拆成可加的三块，并证明了一个此前被误判的结论：

1. **K1 与 K2 的时间归属已定位**。一次前向**恰好发起 2 个 kernel**
   （K1 `_chunk_gdn2_fwd_intra_infer_kernel` + K2 `_chunk_gdn2_fwd_h_o_infer_kernel`），
   在 `B,T,H,K = 2,512,4,128` 的 14 个 (dtype, V) 单元上，默认路径的平均耗时占比是
   **K1 ≈ 180–250 µs、K2 主体 ≈ 140–345 µs、`final_state` 零填充 ≈ 268–506 µs**。
   三者相加与实测整步的残差**恒为 0.0 µs**（分批计时下严格可加），因此这个拆分不是
   估算，而是同一测量窗口内的三条独立测量。
2. **状态零填充是整步最大单项，占了 38–59%**。`chunk_gdn2_fwd_h_o_infer` 在
   `output_final_state=True` 时用 `kg.new_zeros(...)` 分配返回值，但 kernel 会写满
   这个张量的**每一个元素**（grid 覆盖全部 V tile，K 分块链覆盖 `K <= 4*BK`）。
   这次 memset 随即被覆盖，纯属浪费。实测单独一次 `new_zeros` 是
   **196–501 µs**（同样大小的 `new_empty` 只有 **5.84–6.56 µs**）。
3. **把这项浪费去掉后，整步稳定快 1.83x**。3 轮交替 A/B（每轮独立进程 + 独立
   Triton cache），13 个 (dtype, shape) × 3 轮 = **39 次配对全部 > 1**，
   **中位 1.8295x、geomean 1.8376x、区间 1.517–2.206x**。输出 `o` 与最终状态 `ht`
   与基线**逐位相同（39/39 SHA-256 相等）**，峰值显存逐形状相同，
   kernel 路径与 launch 几何（kernel 名、grid、BK/BV/BT）**完全不变**。
4. **V tile 并行度与状态读写已成功分离，并修正了 GDN2-004 的一个解释**。
   split 真正改变 tile 的 **8 个单元**（4 个形状 × 2 dtype）上，K2 主体
   （`h0`/`ht` 都不存）稳定快 **1.259–1.276x（8/8）**；把 `h0_only` 也算进来是
   1.230–1.335x。但一旦打开 `ht`，同一份增益就被 memset 淹没到 **1.023–1.397x**
   （8/8 全部变窄）。也就是说 GDN2-004 观测到的"split 1.07x 整步收益"**没有错**，
   只是它的候选侧仍带着 memset，把 split 的真实贡献稀释了——两者不是同一个量。
5. **V=128/160/256 对照**：split 只在 `128 < V <= 256` 生效，实测该区间
   K2 主体 1.26–1.28x、整步（含 memset）1.83–2.09x；`V=128` 两侧都是 128、
   `V=320` 两侧都是 256，**这两组 `state_bv` 相同、K2 主体 139–147 / 181 µs 两侧持平**，
   作为阴性对照验证了**收益确实来自 V 轴并行度，而不是测量偏差**。

**建议：保留该改动，但保持默认关闭。** 默认路径逐位不变（`new_zeros`），
只有显式设 `FLAG_ATTN_GDN2_STATE_EMPTY_HT=1` 才走实验路径，且带一个
"K 必须被分块链覆盖"的守卫（`_s60_state_all_written`），一旦未来的 BK 策略
不再覆盖全部 K，就自动退回 `new_zeros`，**不可能返回未初始化内存**。

本轮给出了一个可证伪的后续假设（第 8 节），而不是终止理由。

---

## 1. 基线、分支、工作区与提交链

- 基线 SHA：`1320878ddada3b3d29738d1e5f5c0bcc9385def7`
  （GDN2-004 的文档提交；其 `src/` 已包含被接受的 K2 V-tile split）
- 分支：`agent/enflame-profilegdn2-0001`
- 克隆：`/workspace/agent-workspaces/gdn2-irregular-tiles-0/.cache/profile-workspaces/profilegdn2-0001`
- 前序任务：GDN2-004，其报告 `agent-results/GDN2-004.md` 与原始记录
  `GDN2-004-ab-final.jsonl`（290 行）、`GDN2-003-profile.jsonl` 已先读并复用其
  门槛定义与形状集。

| SHA | 说明 |
|---|---|
| `1320878` | 本轮固定基线（不作改动） |
| `9082fae` | `perf(gdn2): make K2 final-state zero fill an explicit opt-in experiment`（实验本体 + 阶段剖析脚本 + A/B 脚本 + 检查脚本） |
| `8c10b1f` | `test(gdn2): pin the final-state coverage guard and harden the check script`（守卫单测 + 检查脚本适配调度器的 6 卡绑定） |

`src/` 的唯一改动是 **+34 / −2 行**（`git diff --numstat 1320878 HEAD -- src/`）：
一个模块级开关 `_S60_STATE_EMPTY_HT_ENABLED`、一个纯函数守卫
`_s60_state_all_written`、`final_state` 分配处改成 `new_empty`/`new_zeros` 二选一，
外加说明性注释。**默认分支行为逐位不变**。

## 2. 设备与环境（只有 GCU 5）

```
$ efsmi -L | sed -n "6,9p"
DEV    SN                Slot    ID           Bus             CPU Affinity     UUID
...
5      A0D5460520864     N/A     1ea0:2a22    0000:83:00.0    128-255          TS9B37120204
```

`dev 5` = `83:00.0`（SN `A0D5460520864`，UUID `TS9B37120204`，驱动报告其 CPU
亲和性为 `128-255`），与任务书一致。GCU 6/7 与 GCU 0–4 全程未触碰。

进程绑定有两处需要如实区分：

- **A/B（§6）**：78/78 条 meta 记录 `cpu_affinity` = `160-191`（外层显式
  `taskset -c 160-191`）、`pci_gcu5 = 83:00.0`、`device_count_visible = 1`、
  `TOPS_VISIBLE_DEVICES=5`，即物理 GCU 5 是进程内序号 0。
- **阶段剖析（§5）**：脚本本身没套 `taskset`，1 条 meta 记录的是 `cpu_affinity=[0,255]`，
  即进程可在 256 个逻辑 CPU 上被调度。它的 PCI/设备可见性与 A/B 相同，但**没有固定
  CPU**——这是阶段剖析数字（尤其那个 5.8 ms 野值）比 A/B 更易受宿主抢占影响的一个
  已知原因，也是 §4 要分批计时的直接动机。

| 项 | 值 |
|---|---|
| 主机 | `vm-dhrj-gd-zone7-d-s60-48g-1-17` |
| Python | 3.12.13（`/flagos/bin/python`） |
| torch / torch_gcu | `2.10.0+cpu` + torch_gcu |
| Triton | 3.6.0（`/opt/flagtree`） |
| 驱动 / FW | 1.10.6 / AP 1.2.7, SP 3.4.21 |
| 链路 | Gen5 x16 |
| GCU 5 显存 | 41846 MiB（`total_memory` = 43878764544 B） |
| SM / SIP | `multi_processor_count=2`、`sip_count=24` |
| 主机负载（运行期间） | loadavg **26.3–45.1**（阶段剖析的 14 个单元逐单元记录；有其它 agent 的无关进程，**非独占**） |
| GPU 占用 | 每次运行前 `Used Size = 0 MiB`，运行后回到 0 MiB |

与 GDN2-003/004 的可比性说明：那两轮在 **GCU 0 / `23:00.0` / CPU `0-31`** 上测得，
本轮在 **GCU 5 / `83:00.0` / CPU `160-191`**。同机同型号、同 torch/Triton/驱动版本、
同测量口径（`torch.gcu.Event` 包住单次调用并同步、取中位）。**因此量级可比、绝对值
不可逐位对齐**；下文凡引用前序数字都注明来源，且不与本轮数字混算比值。

## 3. 复现命令

```bash
cd /workspace/agent-workspaces/gdn2-irregular-tiles-0/.cache/profile-workspaces/profilegdn2-0001
export PYTHONPATH="$PWD/src:$PWD:/opt/flagtree"     # 必须含 /opt/flagtree，否则 triton 不可导入
export TOPS_VISIBLE_DEVICES=5                       # 只让物理 GCU 5 可见 => 进程内序号 0

# (a) 阶段归因：K1 / K2 主体 / 状态零填充，V=96,128,160,192,256,320 × bf16,fp16
/flagos/bin/python benchmark/gdn2_stage_profile.py \
    --out agent-results/PROFILEGDN2-0001-stage.jsonl --dtype both

# (b) 整步 A/B：基线 1320878 vs 同树 + FLAG_ATTN_GDN2_STATE_EMPTY_HT=1，3 轮交替
/flagos/bin/python benchmark/gdn2_final_state_ab.py drive --rounds 3 \
    --out agent-results/PROFILEGDN2-0001-ab.jsonl

# (c) 离线复核（不重跑、只校验已有 JSONL）
/flagos/bin/python benchmark/gdn2_final_state_ab.py check \
    --out agent-results/PROFILEGDN2-0001-ab.jsonl --rounds 3
```

调度器正式入口（`config.json` 中的命令）调的是
`agent-results/profilegdn2-0001_check.py --verify` / `--benchmark`；该脚本本身只做
"跑仓库自带测试 + 调用上面的 A/B 脚本"，不另写一套数值参考。

它接受**两种合法绑定**：调度器 wrapper 的"6 卡可见、`set_device(5)`"（进程内序号 5），
以及本报告用的"`TOPS_VISIBLE_DEVICES=5`、序号 0"。其余组合一律拒绝。这条判定矩阵
用桩对象逐个验证过（7/7 正确）：合法 `(83:00.0, 1 卡可见, 序号 0)` → 接受；
合法 `(83:00.0, 6 卡可见, 序号 5)` → 接受；而 `(23:00.0, …)`（错误物理卡）、
`(83:00.0, 6 卡, 序号 0)`（卡在但绑到了别的卡）、`(83:00.0, 2 卡, 序号 1)`
（歧义绑定）、`(83:00.0, 1 卡, 序号 1)`、`(83:00.0, 6 卡, 序号 6)` 全部返回 4。
后者尤其重要：序号 6 是**另一租户的卡**，绝不能被当成"6 卡可见"放行。

## 4. 测量方法，以及为什么必须分批

单次调用的 event 采样在本机**不可靠**：一次宿主调度抢占会落在 `begin.record()` 与
launch 之间，表现为一个 ~5.8 ms 的样本（与本机一次调度时间片同量级——这是推断，
不是我实测到的调度事件），而该算子连 memset 都没有、状态只有 0.375 MiB 时也会出现。
前几轮剖析里 5.8 ms 的"野值"就是它，**不是 kernel 时间，也不是分配成本**；
上面那个 0.364 的单元里 2.5–5.8 ms 的 4 个样本属于同一现象。

因此阶段剖析改为**一次 event 包住 8 次连续调用、再除以 8**：抢占被摊薄到 1/8，
同时保留全部原始样本、min/p25/median/max 与"超过 3 倍最小值"的样本占比。

K1 的该占比在 14/14 单元里**为 0.0**；K2 的 168 个已测子单元里**只有 1 个为 0.364**
（fp16 `(2,512,4,128,128)` 的 `split_0_bv128/ht_only`）。它的 11 个原始样本是：

```
[424.99, 413.43, 419.89, 416.62, 405.28, 433.72, 566.18, 2462.37, 3248.69, 3141.64, 5794.26]
```

即 **6 个在 405–434 µs 的紧密带内、1 个 566 µs、4 个跳到 2.5–5.8 ms**（4/11 = 0.364）。
我没有删掉或改写这些样本，而是原样保留在 JSONL 里。这个单元在本报告里**只出现在
"可信度"讨论中，不参与第 7 节的任何比值计算**（第 7.2 节该形状用的是 `h0_ht` 而非
`ht_only`）。`ht_new_zeros`/`ht_new_empty` 的占比全为 0.0。

同时记录**宿主墙钟**：`new_zeros` 的 device/host 中位数比接近 1（例如
299.98 vs 302.23 µs、316.84 vs 318.59 µs、382.27 vs 383.79 µs），说明它确实是设备
memset，而不是 host 侧的分配排队。

**与官方口径的差异（如实说明）**：A/B 脚本沿用 GDN2-004 的单次采样口径
（15 个稳态样本 + 事件对 + 同步），与阶段剖析的 8 次批量口径不同。两者在
"candidate 更快、且幅度与预测一致"上互相印证（见第 6 节末的对照），但**不能把
A/B 的 1.83x 与阶段剖析的 2.0x 当作同一个数**。

## 5. 阶段归因：K1 / K2 主体 / 状态零填充

形状 `B,T,H,K = 2,512,4,128`，只推 V；`chunk_size=64`，`BT=64`。K1 `BK=128`，
`BV` = 128（V≤128）或 256（V≥160，`_s60_tile` 向上取 2 的幂）；K2 `BK=128`（K=256 的形状
则为 256），`BV` = 128（split 生效）或 256（不生效）。完整 launch 见 §5.1。
单位 µs，**中位**，batch=8。

| dtype | V | K1 | K2 主体 | 状态零填充 | K1+K2 | 实测整步 | 残差 |
|---|---|---|---|---|---|---|---|
| bf16 | 96 | 195.3 | 143.8 | 330.9 | 339.1 | 670.0 | 0.0 |
| bf16 | 128 | 186.9 | 139.4 | 327.2 | 326.2 | 653.4 | 0.0 |
| bf16 | 160 | 197.3 | 142.5 | 268.0 | 339.8 | 607.8 | 0.0 |
| bf16 | 192 | 191.1 | 141.3 | 336.8 | 332.4 | 669.2 | 0.0 |
| bf16 | 256 | 193.3 | 142.7 | 315.3 | 336.0 | 651.3 | 0.0 |
| bf16 | 320 | 208.2 | 181.0 | 474.7 | 389.2 | 863.9 | 0.0 |
| bf16 | 256 (K=256) | 250.6 | 344.8 | 473.9 | 595.4 | 1069.3 | 0.0 |
| fp16 | 96 | 196.2 | 143.4 | 270.7 | 339.7 | 610.4 | 0.0 |
| fp16 | 128 | 184.8 | 139.8 | 308.7 | 324.6 | 633.3 | 0.0 |
| fp16 | 160 | 196.5 | 143.6 | 323.5 | 340.1 | 663.5 | 0.0 |
| fp16 | 192 | 193.5 | 141.4 | 394.2 | 334.9 | 729.0 | 0.0 |
| fp16 | 256 | 192.9 | 141.4 | 478.4 | 334.4 | 812.8 | 0.0 |
| fp16 | 320 | 205.1 | 181.7 | 497.1 | 386.8 | 883.9 | 0.0 |
| fp16 | 256 (K=256) | 246.9 | 344.1 | 370.3 | 591.0 | 961.4 | 0.0 |

**结论一：K1 是固定成本，与 V 几乎无关。** V=96→256 时 K1 只在 **184.8–197.3 µs**
之间波动（<7%），因为 K1 的 grid 是 `(NT, B·H)`、**不含 V 维度**，V 只在 kernel 内部
以 `for i_v in range(tl.cdiv(V, BV))` 串行遍历，而 V≤256 时该循环只有 1 次迭代
（BV=256；V≤128 时 BV=128，同样 1 次）。V=320 时循环变成 2 次迭代，K1 升到
**205.1–208.2 µs**（bf16 +7.7%、fp16 +6.3%）。K=256 时 K1 升到
**246.9–250.6 µs**（bf16 +29.6%、fp16 +28.0%），与 `BK` 由 128 变 256、
每次 dot 面积翻倍一致。

**结论二：K2 主体是并行度受限，不是访存受限。** 同样 `B·H = 8`，V=96/128 时
K2 主体 139–144 µs，V=160/192/256（split 生效、2 个 V tile）时 141–149 µs，
**几乎持平**；而 split 关闭（BV=256、只有 1 个 V tile）时升到 179–187 µs。
即 K2 主体在“2 个 V tile”下已达到约 1.23–1.34x，之后加 V 不再变慢。

**结论三：状态零填充是最大单项。** 它占整步 **38.5%–58.9%**（14 个单元），
比 K1 和 K2 主体都大。单独测同形状 `new_zeros` 为 195.6–500.8 µs，而
`new_empty` 只有 **5.84–6.56 µs**——差两个数量级，说明成本全在 memset，不在分配。

**结论四：`ht` 打开时 K2 内部的时间不是 memset 的简单叠加。** `ht_only` 比
`neither` 多 293.9–514.5 µs，而独立 memset 只有 195.6–500.8 µs；两者同量级但
不完全相等，因为 kernel 在 `STORE_FINAL_STATE` 分支里多做 1–4 次 block store，
且 memset 与 kernel 的写在同一段内存上产生额外往返。这正是"状态**读写**"必须与
"V tile 并行度"分开测量的原因：只看 `h0_ht` 会把两者混在一起。

### 5.1 派发核对（每个单元都记录实际 launch）

实际 launch（从每个单元的 `launches` 字段读出，不是按代码推断）：

| dtype | V | K1 grid | K1 BK | K1 BV | K2 BV (split 关/开) | K2 BK |
|---|---|---|---|---|---|---|
| bf16/fp16 | 96, 128 | `[8, 8]` | 128 | 128 | 128 / 128（不拆） | 128 |
| bf16/fp16 | 160, 192, 256 | `[8, 8]` | 128 | 256 | 256 / **128** | 128 |
| bf16/fp16 | 320 | `[8, 8]` | 128 | 256 | 256 / 256（不拆） | 128 |
| bf16/fp16 | K=256, V=256 | `[16, 4]` | 256 | 256 | 256 / **128** | 256 |

K2 的 grid 在 trace 里记为 `callable`（脚本按 `V//BV × B·H` 现算），所以表里列的是
决定 grid 的 `BV`。每次调用的 launch 数：K1 恒为 1.0；K2 **仅 ht 路径 1.0、
整步（带 ht）2.0**——即整步恒为 K1+K2 两个 kernel，实验路径没有新增或减少任何
launch，只是把 K2 之前那次 memset 去掉。

**这里有一个必须纠正的细节**：K1 的 grid 是 `(NT, B·H)`，**不含 V 维度**；
K1 内部用 `for i_v in range(tl.cdiv(V, BV))` 串行遍历 V。
因此 K1 的 `BV` 在 V=160/192/256/320 时是 256（`_S60_INTRA_BV` 向上取 2 的幂），
V=96/128 时是 128。这解释了第 5 节的"结论一"：**K1 的耗时对 V 不敏感，是因为
V 只在 K1 内部串行循环，而 V≤256 时该循环只有 1 次迭代**；V=320 时变成 2 次迭代，
K1 从 ~195 µs 升到 ~206 µs（+5.6%），与"多一轮 V 循环"一致。

## 6. 整步 A/B：3 轮交替，39 次配对

`drive` 用 `git archive` 从两个不可变提交各导出一棵树：**base = `1320878`**；
**cand = 同树 + `FLAG_ATTN_GDN2_STATE_EMPTY_HT=1`**。每个 (轮, 侧, dtype, shape)
都是独立进程 + 独立 `TRITON_CACHE_DIR`，因此 `cold_compile_s` 是真实首次编译。
**轮次之间交替方向**（从 `invocation` 记录复核：轮 1 = base 块→cand 块，轮 2 =
cand→base，轮 3 = base→cand），因此"先跑的那一侧"在三轮里不固定，
可以抵消随时间的单向热漂移。注意这是**块级**交替（一轮内 26 次调用全走同一侧），
不是每个形状内部交替；单侧连续 26 次调用的好处是每侧只付一次温升。

| dtype | shape (B,T,H,K,V) | base ms | cand ms | 加速 | 三轮 | state BV |
|---|---|---|---|---|---|---|
| bf16 | (1,1024,4,256,256) | 1.0701 | 0.6686 | 1.601x | 1.517/1.599/1.748 | 128 |
| bf16 | (2,512,4,96,128) | 0.7520 | 0.4240 | 1.774x | 1.782/1.716/1.768 | 128 |
| bf16 | (2,512,4,128,96) | 0.7147 | 0.4173 | 1.713x | 1.712/1.805/1.709 | 128 |
| bf16 | (2,512,4,128,128) | 0.7692 | 0.4102 | 1.875x | 1.783/1.951/1.877 | 128 |
| bf16 | (2,512,4,128,160) | 0.7527 | 0.4123 | 1.826x | 1.821/1.828/1.835 | 128 |
| bf16 | (2,512,4,128,192) | 0.8348 | 0.4081 | 2.045x | 1.885/2.058/2.043 | 128 |
| bf16 | (2,512,4,128,256) | 0.8476 | 0.4051 | 2.092x | 2.098/2.086/2.206 | 128 |
| bf16 | (2,512,4,128,320) | 0.9272 | 0.4647 | 1.995x | 2.204/2.009/1.946 | 256 |
| bf16 | (2,512,8,64,64) | 0.7688 | 0.4837 | 1.589x | 1.589/1.654/1.578 | 64 |
| fp16 | (1,1024,4,256,256) | 1.0753 | 0.6628 | 1.622x | 1.634/1.601/1.608 | 128 |
| fp16 | (2,512,4,128,128) | 0.7656 | 0.4124 | 1.856x | 1.856/1.796/1.889 | 128 |
| fp16 | (2,512,4,128,160) | 0.7606 | 0.4110 | 1.851x | 1.829/1.985/1.851 | 128 |
| fp16 | (2,512,4,128,256) | 0.8357 | 0.4098 | 2.039x | 1.921/2.039/2.190 | 128 |

**39 次配对：中位 1.8295x、geomean 1.8376x、min 1.5172x、max 2.2062x，39/39 > 1。**

### 正确性与不变量（全部由 `check` 从原始 JSONL 重算）

| 项 | 结果 |
|---|---|
| `o`、`ht` 逐位相同 | **39/39 SHA-256 相等** |
| 有限性 | `o`/`ht` 全部 finite（78/78 条 shape 记录） |
| Gate A / Gate B（`o`） | **39/39 两侧全 PASS** |
| Gate A / Gate B（`ht`） | 两侧同为 15 PASS / 24 FAIL-A（**改动前后完全一致**） |
| kernel 路径 / launch 几何 | 39/39 完全相同（kernel 名、grid、BK、BV、BT） |
| 峰值显存 | 13/13 形状两侧完全相同（7.12–16.00 MiB） |
| 冷编译 | 中位 base 16.52 s vs cand 16.57 s（同量级，无回退） |
| 环境 | 78/78 条 meta 与 78/78 条 meta_end：`pci_gcu5=83:00.0`、`device_count_visible=1`、`cpu_affinity=160-191` |
| 阶段剖析的 2×2 交叉 | 14/14 单元 × 4 个 (split, empty) 组合，`o`/`ht` 的 SHA-256 各自只有 1 个值，即 **56/56 组合逐位相同**；全部 finite |
| 开关状态 | base 侧 `FLAG_ATTN_GDN2_STATE_EMPTY_HT` 未设、模块无此属性；cand 侧 `=1`、模块为 `True` |

`ht` 的 Gate A 失败是**基线既有**的 BF16 dtype 精度问题（GDN2-004 已记录同样的
5/8），本轮的关键点是**两侧 24/24 完全一致**，因此不是本改动引入的。

### 交叉印证：阶段剖析的预测 vs A/B 的实测

阶段剖析预测（同一树、同一 memset、同一 K2 主体，仅去 memset）：
整步比值 = (K1+K2+memset)/(K1+K2)。对 (2,512,4,128,160) bf16 为
(197.3+142.5+268.0)/(197.3+142.5) = **2.00x**；A/B 实测 **1.826x**。
对 (2,512,4,128,256) bf16 预测 2.15x、实测 2.092x；fp16 预测 2.16x、实测 2.039x。
**预测略高于实测、方向与量级一致**，差异来自 A/B 的单次采样口径与剖析的分批口径
（第 4 节），以及极小形状上固定开销占比更高。两条独立路径给出同一结论。

## 7. V tile 并行度 vs 状态读写：分离实验

这是本任务假设的核心。同一份输入、同一进程内交替 2×2 单元，只动两个旋钮：
`split ∈ {off,on}`（K2 的 V tile 256 vs 128）与 `ht ∈ {zeros,empty}`。

### 7.1 只动 V tile（K2 主体，`h0`/`ht` 都不存）

| dtype | shape | BV=256 | BV=128 | 收益 |
|---|---|---|---|---|
| bf16 | (2,512,4,128,160) | 181.8 | 142.5 | **1.276x** |
| bf16 | (2,512,4,128,192) | 179.3 | 141.3 | **1.269x** |
| bf16 | (2,512,4,128,256) | 180.2 | 142.7 | **1.263x** |
| bf16 | (1,1024,4,256,256) | 436.4 | 344.8 | **1.266x** |
| fp16 | (2,512,4,128,160) | 180.8 | 143.6 | **1.259x** |
| fp16 | (2,512,4,128,192) | 180.2 | 141.4 | **1.275x** |
| fp16 | (2,512,4,128,256) | 179.9 | 141.4 | **1.272x** |
| fp16 | (1,1024,4,256,256) | 435.4 | 344.1 | **1.265x** |

8/8 单元一致 **1.26–1.28x**。这直接证明 **V tile 拆分本身就有约 1.26x 的并行度收益**，
与 GDN2-004 的结论一致，且这里把它从状态读写里彻底隔离出来了。

### 7.2 打开 `ht` 后，同一份收益被 memset 稀释

| dtype | shape | h0+ht BV=256 | h0+ht BV=128 | 收益 |
|---|---|---|---|---|
| bf16 | (2,512,4,128,160) | 620.2 | 444.0 | 1.397x |
| bf16 | (2,512,4,128,192) | 583.6 | 558.7 | 1.045x |
| bf16 | (2,512,4,128,256) | 569.3 | 532.8 | 1.069x |
| fp16 | (2,512,4,128,160) | 604.9 | 523.9 | 1.155x |
| fp16 | (2,512,4,128,192) | 545.3 | 490.6 | 1.111x |
| fp16 | (2,512,4,128,256) | 614.7 | 600.6 | 1.023x |
| bf16 | (1,1024,4,256,256) | 948.1 | 797.9 | 1.188x |
| fp16 | (1,1024,4,256,256) | 929.3 | 792.5 | 1.173x |

同一形状，收益从 **1.26x 掉到 1.02–1.40x**。原因是 `h0_ht` 里两边各多了一次
~300–500 µs 的 memset，把 ~40 µs 的并行度收益稀释成几个百分点。**这就是"tile 并行度"
与"状态读写"必须分开测量的实证**：混在一起测，会得出"split 收益不稳定/很小"的错误结论。

### 7.3 V=128 / 160 / 256 对照的完整解释

- **V=128**：tile 已经是 128，split 开/关的 `state_bv` 都是 128 → 对照组，
  K2 主体 139–147 µs 两侧相同。任务要求的 V=128 对照**已验证为"无差别"**。
- **V=160/256**：split 生效，BV 256→128，K2 主体 1.26x、整步 1.83–2.09x。
- **V=320**：split 不生效（`V > 256` 回到 256 单 tile），两侧 `state_bv` 都是 256，
  K2 主体 181.0 vs 181.2 µs、整步比 1.995x 全部来自去掉 memset。作为"规则不该生效"的
  阴性对照，它**正确地没有产生 split 收益**，因此 1.26x 不可能来自测量偏差。

## 8. 改动本体、守卫与默认行为

`src/flag_attn/runtime/backend/_enflame/FLA/gdn2/chunk_gdn2.py`：

```python
_S60_STATE_EMPTY_HT_ENABLED = (
    os.environ.get("FLAG_ATTN_GDN2_STATE_EMPTY_HT", "0") == "1"
)

def _s60_state_all_written(K: int, state_bk: int) -> bool:
    """K2 的 K 分块链是否写满整个 final state。"""
    return K <= 4 * state_bk

# ...
    if output_final_state:
        empty_ht = (_S60_STATE_EMPTY_HT_ENABLED
                    and _s60_state_all_written(K, state_bk))
        new_state = kg.new_empty if empty_ht else kg.new_zeros
        if state_v_first:
            final_state = new_state(N, H, V, K, dtype=torch.float32)
        else:
            final_state = new_state(N, H, K, V, dtype=torch.float32)
```

守卫不是装饰：kernel 用 `b_h1..b_h4` 四块累加并以 boundary check 存储，
只有 `K <= 4*BK` 时四块覆盖整个 K。S60 的 BK 阶梯是 64/128/256，对 `K<=256`
恒有 `K <= 4*BK`，所以当前 K 范围全部安全；但若未来把 BK 降到更小，守卫会
**自动退回 `new_zeros`**，不会返回未初始化内存。

### 8.1 直接验证"kernel 写满"的两组毒化实验

不靠推理，直接把分配器喂成脏块再跑实验路径：

- 先用一个 8 块的 junk 列表填满同尺寸块再丢弃，让缓存分配器不得不交出一块含
  `NaN` / `+inf` / `1e38` / `-3.5` 的内存；
- 再跑实验路径，要求结果与**同一输入、干净分配**的基线 `torch.equal` 逐位相等。

两轮探针（原始输出已存为 `agent-results/PROFILEGDN2-0001-poison.jsonl`）：

| 探针 | 形状 | 毒化值 | 布局 | 检查数 | 结果 |
|---|---|---|---|---|---|
| 1 | (2,512,4,96,128)、(2,512,4,128,160)、(2,512,4,128,256)、(1,1024,4,256,256)，bf16 | `nan`/`1e38`/`-7.0` | 默认 | **24** | 24/24 `finite=True`、`equal_ref=True`、`nonfinite_elems=0` |
| 2 | 9 个形状，覆盖 `V ∈ {129,160,192,224,257,320,384,512}`、`K ∈ {64,96,128,192}`，bf16+fp16 | `nan`/`inf`/`-3.5`/`1e38` | `state_v_first ∈ {False,True}` | **72** | **72/72 零失败**（`poison checks=72 fails=0`） |

**两轮合计 96 次毒化检验，全部与干净基线逐位相等、且全部 finite。**
这说明 kernel 确实写满每个元素，`new_empty` 不会漏出任何未初始化值。

需要说明的是：探针 1 用的 `(2,512,4,128,160)` 等形状与本报告的主表一致，但它只跑了
默认布局；`state_v_first` 两种布局与 V=129/257 的掩码边界由探针 2 覆盖。

### 8.2 仓库内新增的测试（`tests/flag_attn/test_gdn2_forward_tiles.py`）

| 测试 | 覆盖 |
|---|---|
| `test_empty_ht_switch_is_off_by_default` | 出厂必须 `new_zeros`，开关默认 False |
| `test_empty_ht_switch_reads_the_environment` | `FLAG_ATTN_GDN2_STATE_EMPTY_HT=1` 确实开启，reload 后复原 |
| `test_empty_ht_allocation_is_bit_identical` | 4 个形状（含 K=96、K=256、V=96）开关前后 `o`/`ht` 逐位相同 |
| `test_s60_state_all_written_matches_the_k_block_chain` | 7 组 (K,bk) 钉住守卫的判定 |
| `test_empty_ht_falls_back_when_bk_does_not_cover_k` | 人为把 BK 降到 16，守卫必须退回 `new_zeros` 且结果与默认一致 |

单跑这 14 个相关用例：**14 passed**。全量三文件：**170 passed, 1 skipped**
（见 §9）。

## 9. 正确性验证

```
$ TOPS_VISIBLE_DEVICES=5 PYTHONPATH=$PWD/src:$PWD:/opt/flagtree taskset -c 160-191 \
    /flagos/bin/python -m pytest \
    tests/flag_attn/test_gdn2_forward_tiles.py \
    tests/flag_attn/test_gdn2_backward_reference.py \
    tests/flag_attn/test_gdn2.py -q
170 passed, 1 skipped, 2 warnings in 60.31s (0:01:00)
```

调度器口径（6 卡可见 + `set_device(5)`）下的 `--verify` 也实跑通过：

```
device: GCU 5 PCI 83:00.0, visible=6, ordinal=5, torch 2.10.0+cpu
...
  torch.bfloat16 (2, 512, 4, 128, 160) bit_identical=True gate=True finite=True rms_rel=4.1e-03 ok
  torch.float16  (2, 512, 4, 128, 256) bit_identical=True gate=True finite=True rms_rel=5.2e-04 ok
  torch.bfloat16 (2, 512, 4, 96, 128)  bit_identical=True gate=True finite=True rms_rel=4.2e-03 ok
  torch.bfloat16 (1, 1024, 4, 256, 256) bit_identical=True gate=True finite=True rms_rel=4.1e-03 ok
  torch.float16  (2, 512, 8, 64, 64)   bit_identical=True gate=True finite=True rms_rel=5.3e-04 ok
VERIFY PASS: tests, default-off switch and bit-identical experiment path
```

唯一 skip 是 `test_gdn2_backward_reference.py` 的"设备反向不可用"用例。
**`chunk_gdn2` 只有推理路径，设备反向没有验证，本报告任何地方都不把它写成已验证。**
仓库里的 CPU FP64 伴随只是数值参考，不构成设备反向证据。

### 9.1 一个必须说清的测试环境陷阱

最初用"6 卡可见 + `set_device(5)`"跑 `test_gdn2_backward_reference.py` 时，
12 个用例报 `Kurama Error [TOPS]: topsErrorInvalidValue`，stderr 为
`FindMemObj DeviceId[0] ... is not match for DeviceId[5] of stream`。原因**不是算子
数值问题**，而是该测试文件里硬编码了 `DEVICE = torch.device("gcu:0")`，在 6 卡可见
时 `gcu:0` 是物理 GCU 0，与当前设备 5 冲突。改用 `TOPS_VISIBLE_DEVICES=5`（只有
GCU5 可见，`gcu:0` 即物理 GCU5）后同一文件全过。我把这一点写进报告而不是隐藏它，
因为它会影响任何按"多卡可见"方式复现本任务的人。

## 10. 与 GDN2-003/004 的关系（诚实边界）

- GDN2-004 在 GCU 0 上得出"K2 V-tile split 整步 1.07x"。本轮在 GCU 5 上测得
  **K2 主体 1.26x、整步 1.83x**。两者不矛盾：GDN2-004 的候选侧**仍然带着 memset**，
  所以它的整步收益被 memset 稀释；本轮把 memset 去掉后，split 的真实贡献才显露，
  同时 memset 本身贡献了更大的一块。**但我没有在 GCU 0 上重跑本轮的实验**，
  所以"GCU 0 上 memset 也是 0.3–0.5 ms"只是基于同型号同版本的同机推断，
  **不是本轮实测**。
- 本轮所有数字都来自 GCU 5，**与 GDN2-003/004 的 GCU 0 绝对值不可逐位对齐**。
  可比的是：同测量口径、同门槛定义、同形状集、同 torch/Triton/驱动版本。
- 我没有重新验证 GDN2-004 的 state BK 三档结论（本轮未改 BK）。

## 11. 未测范围

- 只测了 `B,T,H,K = 2,512,4,128` 的 V 扫描 + 一个 `1,1024,4,256,256` 控制形状；
  **`B·H = 8` 全程未变**，因此本报告不能说明 K1/K2 在更大 `B·H` 下的扩展性
  （K2 的并行度直接由 `B·H × V tile` 决定，`B·H` 更大时 V tile 的相对收益可能变小）；
  `T` 更大（4096/8192）与 `H` 更大的情形未扫。
- 只测 bf16 / fp16；fp32、fp8 未测。
- 未测 `return_intermediate_states=True` 路径（那里 `h` 用 `new_empty` 已经是既有行为，
  但本轮的 `ht` 实验不影响它）。
- 未测 varlen（`cu_seqlens`）路径。
- **设备反向未验证**（算子无此路径）。
- 官方正式测量由调度器独占运行；本报告的数字来自我在 GCU 5 上的自跑，
  两者口径与结论一致但**不是同一次执行**。

## 12. 可证伪的后续假设

**假设 H1**：K2 主体的 1.26x 已经接近 V 轴并行度上限，进一步提升应转向
**减少 K2 的串行 chunk 链**，而不是继续调 V tile。

可证伪预测：在 `V=256`、`B·H=8`、`NT=8` 上把 K2 的 V tile 再拆到 4 块（BV=64），
若 K2 主体**不再变快（收益 <1.05x）甚至变慢**，则说明 V 轴已饱和、瓶颈在串行链；
若仍能再快 ~1.2x，则说明并行度尚未饱和，H1 被证伪。

**另一条更省事的假设 H2**（若只允许一次实验）：`ht` 的零填充可以完全消除，
因为 kernel 写满（本轮已用 96 次毒化检验证明，见 §8.1）。下一步应验证它在
**`state_v_first=True` + varlen + 更大 T** 下依然成立——这是本轮唯一没有覆盖的
风险面，也是把实验路径转正前的最后一个门槛。

**为什么不是终止理由**：本轮找到的浪费（占整步 38–59%）已经消除且逐位安全，
但 K2 主体的 1.26x 说明并行度仍有可测的结构性收益，H1 是一个明确、便宜、
可证伪的下一步。

---

## 附录 A：产物清单

| 文件 | 内容 |
|---|---|
| `agent-results/PROFILEGDN2-0001.md` | 本报告 |
| `agent-results/PROFILEGDN2-0001-stage.jsonl` | 阶段归因原始记录（1 meta + 14 cell + 1 end；每单元含全部 11 个原始样本、launch、四个 (split,empty) 组合的 SHA-256） |
| `agent-results/PROFILEGDN2-0001-ab.jsonl` | 整步 A/B 原始记录，314 行 = 78 shape + 78 meta（子进程前）+ 78 meta_end（子进程后，含峰值显存/结束时间）+ 78 invocation（真实 argv/env）+ driver + driver_end |
| `agent-results/PROFILEGDN2-0001-stage.log` | 阶段剖析 stdout |
| `agent-results/PROFILEGDN2-0001-ab.log` | A/B drive+check stdout |
| `agent-results/PROFILEGDN2-0001-poison.jsonl` | 第一轮毒化检验的 24 条原始记录（§8.1） |
| `agent-results/profilegdn2-0001_check.py` | 调度器入口：`--verify` / `--benchmark` |
| `benchmark/gdn2_stage_profile.py` | 阶段归因脚本（本报告 §5） |
| `benchmark/gdn2_final_state_ab.py` | 整步 A/B 脚本（`drive`/`run`/`check`，本报告 §6） |

## 附录 B：原始样本长什么样

`PROFILEGDN2-0001-stage.jsonl` 每个单元都带全部 11 个原始样本，下面是
bf16 `(2,512,4,128,160)` 的 `split_1_bv128` / `full_empty_ht_0` **原样截取**：

```json
"device_us": [629.76, 618.96, 601.28, 612.98, 649.64, 607.76, 603.21,
              605.56, 602.84, 621.21, 604.32]
"host_us":   [618.10, 605.89, 588.06, 599.63, 636.18, 594.17, 589.72,
              592.55, 589.58, 607.73, 590.78]
"median_us": 607.76, "min_us": 601.28, "p25_us": 603.21, "max_us": 649.64
"host_median_us": 594.17, "batch": 8, "warmup_iters": 1671, "outlier_fraction": 0.0
"launches": {"per_call": 2.0, "total_launches": 176, "distinct": [
   {"kernel": "_chunk_gdn2_fwd_intra_infer_kernel", "grid": [8, 8],
    "cfg": {"BK": 128, "BV": 256, "BT": 64, "num_warps": 1, "num_stages": 2}},
   {"kernel": "_chunk_gdn2_fwd_h_o_infer_kernel", "grid": "callable",
    "cfg": {"BK": 128, "BV": 128, "BT": 64, "num_warps": 1}}]}
```

注意这个样本本身就能说明方法：`device_us` 中位 607.76 与 `host_us` 中位 594.17 相差
2.3%，说明测的是设备执行而不是 host 排队；`outlier_fraction=0.0` 表示 11 个样本都
落在最小值的 3 倍以内；`per_call=2.0` 表示每次整步恰好 2 个 launch。

`PROFILEGDN2-0001-ab.jsonl` 每个 shape 记录带 15 个稳态样本、冷编译时间、峰值显存、
launch 列表、`o`/`ht` 的 SHA-256、Gate A/B 判定，以及 CPU FP64 参考的
`max_abs` / `rms_rel` / `ref_rms` / `ref_max`。A/B 的每个子进程日志路径也由 driver
记进 `child_log` 字段（临时目录结束后清理，记录本身保留）。
