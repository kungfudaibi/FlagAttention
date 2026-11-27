# MiniMax MSA 跨平台 prefill Profile

采集于 2026-11-27。对照对象为天数 BI-V150 和平头哥 ZW810E，另引用同形状的燧原 S60 设备 Profile。

## 输入和测量

固定 BF16、B=1、Q=1024、KV=16384、8 个 Q head、1 个 KV head、D=128、128 token/page、TopK=16、init/local=1/2。两台卡使用同一份 CPU 随机输入和物理页表。输入哈希与 TopK 哈希一致。五个 query 的输出对照 FP32，最大绝对误差均约 1.36e-4。

| 设备及工具 | Index Score | TopK | Sparse Attention | 无探针整链 |
| --- | ---: | ---: | ---: | ---: |
| BI-V150 · ixSYS | 0.528 ms | 0.0204 ms | 0.640 ms | 1.180 ms |
| ZW810E · Asight Systems | 0.071 ms | 0.0060 ms | 0.671 ms | 0.762 ms |
| S60 · TopsProf，既有记录 | 0.556 ms | 0.625 ms | direct 3.113 ms，共享前缀 0.304 ms | 4.716 ms |

前两行的单 kernel 时间是厂商时间线中的算术平均值，整链是独立的无探针设备 Event 中位数。S60 行来自另一轮同形状实验，单 kernel 数据采用该报告的中位数。三台使用各自的软件栈。孤立调用的 Event 可能包括提交间隙，尤其是短 TopK，不能替代时间线中的 kernel duration。 [S60 设备时间线](MSA-C200-prefill-profile-2026-11-27.md) · [数值与 Profile 摘录](msa-cross-platform-2026-11-27/evidence.json)

## 实现能解释的差距

810E 和天数的 attention 接近，810E 的整链优势主要来自 Score。810E 同卡改变 KV 切分数，保留同一 BF16 计算、页表和输入，从 1 路到 16 路使 program 数从 16 增至 256，Score **0.214 → 0.075 ms**，整链 **0.893 → 0.756 ms**。所有因果有效的 Score 和 TopK 逐位一致。Score 分配中未写入的未来页没有参与比较。

## 平台差异仍有多大

让两台卡执行同一份 `_index_block_score_kernel` Triton 源码，固定 64 query × 128 KV token tile、4 warps、1 stage 和 16 路切分，BI-V150 **0.418 ms**，810E **0.084 ms**，约 **5×**。两边可见 Score 与各自原生实现逐位相同，TopK 一致。这组差值覆盖硬件、各自的 Triton 编译后端、生成的指令和调度。当前没有等价机器码或统一硬件计数口径来隔离其中的纯硬件份额。

天数 ixKN 对 direct 的一次 replay 测到 **TCU Efficiency 45.84%**、**DRAM Read Efficiency 2.08%**、**Achieved Occupancy 24.66%**。这三个量按 ixKN 的定义读取，S60 的 SIP/2D_EFFICIENCY 使用另一套定义。810E 的 Asight Systems 有有效时间线，Asight Compute 返回 `Device is not ready for profiling`，该轮没有矩阵或访存计数。

对比 S60，已观察到的主要差距在 sparse attention 和 TopK。现有结果没有在 S60 上运行这份共用 Score kernel，也没有掌握三方同定义的内存和矩阵计数。

## 复测

- BI-V150 使用 FlagAttention [abc0837](https://github.com/flagos-ai/FlagAttention/commit/abc0837561b1d52d1d59f7cfe727d8c4000b8136)、CoreX 4.4、Torch 2.7.1 和 FlagTree/Triton 3.6.0。 [BI-V150 实现与基准](https://github.com/flagos-ai/FlagAttention/pull/100)
- 810E 使用 FlagAttention [d01b5dd1](https://github.com/flagos-ai/FlagAttention/commit/d01b5dd1)、PPU SDK 2.1 系列、Torch 2.10.0 和 FlagTree/Triton 3.6.0。 [810E 实现与基准](https://github.com/flagos-ai/FlagAttention/pull/111)
- [profile.py](msa-cross-platform-2026-11-27/profile.py) 使用公开入口构造并测量整链。`--score-split-ab` 只用于 810E，`--common-score` 在两边运行同一个 Score kernel。环境需要设备可见和对应的厂商 FlagTree。适配路径应打印为 `iluvatar` 或 `thead`。

~~~bash
export PYTHONPATH=/path/to/pinned-flagattention/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
python /path/to/profile.py --common-score
python /path/to/profile.py --score-split-ab  # 810E only

# BI-V150
ixsys -t cuda_device --print-gpu-summary python /path/to/profile.py --trace-only
ixkn-cli --kernel-name _gqa_sparse --launch-count 1 --section SpeedOfLight,LaunchStats,Occupancy --csv python /path/to/profile.py --trace-only

# ZW810E
asys profile -t hggc --sample=none -o msa python /path/to/profile.py --trace-only
asys stats --report ppu_op_sum --format csv msa.asysrep
~~~

`evidence.json` 收录本轮观测的汇总值和关键计数。原始厂商二进制 trace 是临时采集产物。
