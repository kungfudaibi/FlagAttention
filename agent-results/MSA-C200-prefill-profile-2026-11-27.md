# C200 MiniMax MSA prefill 性能分析（2026-11-27）

## 当前基线

固定 BF16 prefill，B=1、Q=1024、KV=16384、8 个 Q head、1 个 KV head、D=128、TopK=16，页大小 128 token。共享前缀先处理 3 页，Direct 处理剩余 13 页。整链时间包含 Score、TopK、共享前缀、K 排版（启用时）和 Direct。主性能对照在 C200 5 号卡（PCI 83:00.0）进行，驻留与页内时钟诊断在 C200 2 号卡（PCI 34:00.0）进行。结果覆盖固定 benchmark 的 prefill 原型。

目前最快原型是 **Direct 2.832 ms、整链 4.445 ms**。相比同轮原地更新版，页索引前移的 20 组配对中，Direct 和整链各有 19 组更快，配对中位数分别省 **19 µs、22 µs**。三组随机输入输出逐位一致。旧 C200 原版的整链存档为 6.794 ms，该值与 4.445 ms 来自独立轮次。本次安排的原版和最快版同轮复测受全机 benchmark 锁占用阻塞。 [最新配对](msa-c200-2026-11-27/paired-measurements.log) · [原版存档](msa-c200-2026-11-26/measurements.log)

## 设备时间线与计数器

在较早的公开入口组合上采全链，源码基点为 062ec84，开启 packed K、Score page-major 和页元数据预解析。无 profiler 的整链中位数为 **4.716 ms**，TopsProf 只开启 operator 事件时得到下表。阶段中位数取预热后 3 轮，2D 效率来自独立的 tcu 采集。

| 阶段 | 设备耗时 | Kernel 时间占比 | SIP/2D_EFFICIENCY |
| --- | ---: | ---: | ---: |
| Index Score | 0.556 ms | 12.0% | 4.364% |
| TopK | 0.625 ms | 13.5% | 0% |
| 共享前缀 | 0.304 ms | 6.6% | 2.185% |
| K 排版 | 0.023 ms | 0.5% | 0% |
| Direct | 3.113 ms | **67.4%** | **0.885%** |

五个 kernel 的间隙每轮合计约 2 µs。同卡 2048×2048 BF16 稠密 GEMM 对照给出 44.1%–45.4%，确认了 2D 指标能区分两类计算负载。稠密 GEMM 实际运行 vendor matmul kernel，MSA Direct 由 Triton 将 QK/PV 分别编译为 BF16 M=8、N=128、K=128 的矩阵调用。2D 指标分母未获得厂商定义，当前使用其同形状相对变化。 [设备时间线](msa-c200-2026-11-27/timeline.log) · [计数器与 GEMM 对照](msa-c200-2026-11-27/hardware-counters.log)

TopsProf 1.9.10 和 tcu 在容器里给出了有效设备时间戳和计数器。普通容器用户创建 profiling context 时出现 DTU 错误 -160，临时在已授权容器的 root 上运行采集成功，去掉全部 capability 的正向对照仍失败。常规计时采用无 profiler 设备 Event。开启 SDTE/CDTE 搬运事件使 Direct 从 3.113 ms 变为 3.342 ms。VPD 里 Direct 时间窗内各 SDTE 路的事件覆盖中位数约 38%，事件缺少字节数、页标识和 kernel 关联。这部分记录可用于定位搬运活动，实际片外带宽和单次等待归因仍待验证。 [采集权限对照](msa-c200-2026-11-27/profiler-control.log) · [开关搬运事件的时间线](msa-c200-2026-11-27/timeline.log)

## Direct 的因果对照

每行耗时来自各自的同轮实验。K 预取、running max 和页索引前移分别保留自己的匹配对照。

| 改动 | Direct 或整链 | 结论 |
| --- | --- | --- |
| 早搬 V，随后引入 packed K | 整链 6.794 → 6.355 → 5.091 ms | 计入了 K 排版，packed K 对应 fb664f8。 |
| Score 按逻辑页连续写回 | Score 0.924 → 0.647 ms | 计入该轮整链后为 5.076 → 4.820 ms，对应 a9fb9f7。 |
| 批量解析逻辑页和物理页 | Direct 3.198 → 3.079 ms | 页表加载提到循环外，循环内仍需选出当前索引，对应 062ec84。 |
| 单向跨 query 预取 K | 3.004 → 2.930 ms | 与双 query 串行比较，20/20 组更快，约省 75 µs。B 的 K 在 A 的 QK 前发起，跨过 A 的 QK/PV。 |
| running max 原地更新叠加单向 K 预取 | 2.921 → 2.850 ms | 20/20 组更快，消除每轮 A/B 各一次拷贝及其 DTE wait，2D 效率 0.982% → 1.020%。 |
| 下一轮索引提前选出 | 2.849 → 2.832 ms | 四个选取移到 A 的 PV 与 B 的 QK 之间，搬运顺序一致，2D 效率约 1.02%。 |
| 双向 K 预取 | 2.924 → 2.940 ms | 相对单向版约慢 17 µs，反向搬运存在但循环回边前仍有 wait。尾轮拆分对照未完成。 |
| 同时提前搬第二条 query 的 K/V | Direct 约慢 0.10 ms | 该方案产生回退。 |

早搬 V 的设备时钟对照在三个 program 上分别抽了首、中、末页。PV 开始提前约 1.00–1.10 µs。QK 前 wait 区间缩短约 0.35–0.47 µs，从页循环入口到 QK 开始基本持平。这个例子说明需要同时记发起点、wait 入口和消费者开始点。 [早搬 V 页内对照](msa-c200-2026-11-27/clock-summary.json) · [早期优化和负结果](C200-progress-2026-11-26.md) · [新原型配对](msa-c200-2026-11-27/paired-measurements.log)

## 搬运上界与页内计算

驻留诊断在相同的 Q、TopK 和页表上，将被测 K/V 页内容设为一致，计入每个 program 的首次加载。输出逐位一致。直接让 K 驻留时编译器会把循环不变的 QK 提到循环外，因此用 GCU IR 调整 QK 位置和缓冲区生命周期，再核对最终 LLIR 中两次 QK、两次 PV 都在循环内。该实验给出固定调度下移除重复搬运的收益，覆盖环境为共享主机上的独占 2 号卡探索性诊断。

| 调度 | Direct | 整链 | Direct 配对中位差 |
| --- | ---: | ---: | ---: |
| 逐页搬 K/V | 2.811 ms | 4.419 ms | — |
| K 驻留 | 2.311 ms | 3.923 ms | −0.500 ms |
| V 驻留 | 2.471 ms | 4.077 ms | −0.341 ms |
| K/V 都驻留 | **2.225 ms** | **3.832 ms** | **−0.586 ms** |

联合驻留的 20 组配对全部更快，Direct 省约 21%。逻辑页访问量与实际片外流量仍需独立计数。合成输入在扣除共享前缀后，相邻 2/4 条 query 同槽复用的页访问节省上限仅 0.4%/1.2%，允许跨槽时为 5.5%/15.1%，且后者还需计入重排与 softmax 状态合并。 [驻留 A/B](msa-c200-2026-11-27/paired-measurements.log) · [QK/PV 与搬运核验](msa-c200-2026-11-27/ir-check.log) · [页重合统计](msa-c200-2026-11-27/page-clock-and-overlap.log)

进一步用设备时钟采三个 program 的首、中、末槽，每处 12 轮。插桩前后逐位一致，整链计时增加约 0.02–0.05 ms。下表为 K/V 驻留版每条 query 的单页指令区间，约扣除 0.15 µs 空打点开销。区间用于比较量级，无法外推成整个 kernel 的硬件占比。

| 页内阶段 | 抽样中位数范围 |
| --- | ---: |
| QK 调用 | 0.47–0.48 µs |
| QK 返回到首条 exp2，包含 mask、缩放和行最大值 | **0.50–0.86 µs** |
| 首条到末条 exp2，包含 alpha | 约 0.14–0.15 µs |
| exp2 后到 PV 前，包含状态更新和 BF16 转换 | 约 0.35–0.39 µs |
| PV 调用 | 通常 0.53–0.56 µs，首槽一个采样点为 0.83 µs |

softmax 与状态更新合计约 **0.99–1.37 µs**。逐页搬运版中，A 的 QK 前 K 等待约 **0.71–1.65 µs**，已有 K 预取的 B 和两条 query 的 PV 前等待在这些采样点接近打点底噪。下一步可先沿生成的 IR 检查完整可见页是否仍物化 8×128 的 mask 或 logits 临时缓冲，以最小改动配对验证缩放、mask 和行最大值之间是否能少一次片上读写。 [页内统计](msa-c200-2026-11-27/stage-summary.json) · [时钟原始样本](msa-c200-2026-11-27/page-clock-and-overlap.log)

## 跨平台参照与复现

同形状 BF16 prefill 的整链存档为天数 BI-V150 1.196 ms、平头哥 810E 0.822 ms、燧原 S60 6.779 ms。各平台的软件、预热和缓存条件有差异。Hopper SM90 0.1358 ms 和海光 C-3000 0.6008 ms 的公开表格未明确是否同时计入 Score 和 TopK，保留作独立参考。 [BI-V150](https://github.com/flagos-ai/FlagAttention/pull/100) · [810E](https://github.com/flagos-ai/FlagAttention/pull/111) · [S60](https://github.com/flagos-ai/FlagAttention/pull/104) · [Hopper](https://github.com/flagos-ai/FlagAttention/pull/85) · [海光](https://github.com/flagos-ai/FlagAttention/pull/94)

本分支包含脚本所用的 26368f0 内核签名。[最快候选脚本](msa-c200-2026-11-27/ahead_collect.py) 和 [输入构造](msa-c200-2026-11-27/public_check.py) 使用现有 benchmark 的 shape、公开入口和 GCU 随机数生成器。将两份脚本复制到同一个可写临时目录，从仓库根目录用已经配置的 torch_gcu/FlagTree 容器运行。脚本默认 GCU 5，设备和 CPU 亲和性按当前映射核对。原型包含 GCU IR 改写。

~~~bash
export PYTHONPATH="$PWD/src:$PWD:/opt/flagtree"
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5
taskset -c 160-191 /flagos/bin/python /path/to/scratch/ahead_collect.py plain
~~~
