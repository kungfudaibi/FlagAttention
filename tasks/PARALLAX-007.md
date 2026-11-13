# PARALLAX-007：D64 静态调参生产候选

基于 PARALLAX-006 提交 `dcf487e19e333b796ac7fbb0e535c60aaf7528b2`，独立分支 `agent/enflame-parallax-static-tune-release`，容器 GCU 1。前置是审查静态配置对旧 fused DQR / grouped DKV 的路径保持、冷启动收益、稳态无回退和配置脆弱性；审查前不自动启动。

本任务只研究 D64/T256/G4 的静态调参能否成为默认候选。覆盖 FP16/BF16 目标及邻近形状、T63 回退、输出和全梯度；空缓存冷启动与预热后的完整调用分别记录。安静机器上对精确旧提交交替三轮 A/B，若稳态回退、数值异常或配置依赖不稳，保持实验开关默认关闭并给出停止建议。仅提交最小生产候选改动和 pytest 回归。

另做跨进程磁盘 autotune 缓存对照：当前 FlagTree Triton 的 autotuner 支持 `TRITON_CACHE_AUTOTUNING=1`，但本项目 `libtuner` 包装层尚未确认启用。分别测空缓存首次进程、相同缓存目录的新进程和缓存失效情形，记录编译/调参耗时与缓存命中证据，并与静态配置比较。缓存只可能降低重复进程的调参成本，不应把它记作首次空缓存收益。

