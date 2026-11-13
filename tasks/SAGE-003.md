# SAGE-003：GCU 设备 SDPA 基线与 SageAttention

基于 SAGE-002 实验提交 `3c3634fba806df4e0713e7bce14f901f262bb1f1`，独立克隆 `/workspace/agent-workspaces/sage-sdpa-device-5`、分支 `agent/enflame-sage-sdpa-device-5`。SAGE-002 的正式 A/B 仅覆盖 D64，Sage 当时约比设备 fused SDPA 慢 4.3 倍；PyTorch `F.scaled_dot_product_attention` 在该镜像可能回退 CPU，不能把它冒充设备 SDPA 基线。

只用容器 GCU 5（PCI `83:00.0`，CPU `160-191`），先核对张量、当前 stream 与真实 kernel 的设备归属。分别测 D64、D128 的代表形状，验证输出、有限值、量化误差与设备路径。对确认在 GCU 上执行的 fused SDPA 与 Sage 做同输入、同设备、同步的完整调用三轮交替 A/B；记录冷编译、稳态、峰值显存、实际 kernel 路径和原始样本。若 Sage 仍明显落后，定位量化/softmax/访存或 launch 的主因，只做一项可验证优化；保留无收益结论，不改默认路由。交付测试、复现脚本、报告、原始 JSONL、退出码和提交 SHA。

