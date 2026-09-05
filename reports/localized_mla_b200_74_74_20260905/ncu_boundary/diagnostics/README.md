# L2 越界与计数一致性复核

后续单指标测量及两轮完整对照见 [L2 hit rate 记录](../../../localized_mla_b200_74_74_separate_20260905/L2_HIT_RATE_NOTES.md)。

配置：NVIDIA B200 74/74 SM，localized prefill，B=64、Sq=128、Sk=65,536，seed=42。

| 复核 | raw L2 hit rate | (hit+miss)/total−1 | replay passes |
| --- | ---: | ---: | ---: |
| first_combined | 105.68% | +8.80% | 4 |
| application_replay | 103.56% | +6.89% | 4 |
| l2_hit_miss | 未请求 | 未请求 total | 3 |
| l2_only | 102.59% | +5.82% | 4 |
| repeat_combined | 113.08% | +16.26% | 4 |

`repeat_combined` 重复原 11 指标；`l2_only` 只请求 duration、L2 rate、total、hit、miss；`l2_hit_miss` 只请求 hit/miss，仍需 3 pass，因此未得到单 pass 的一致比率；`application_replay` 使用新进程进行每个 pass，其他 11 项指标与参数保持一致。

多次复核均未消除原始 L2 百分比越界。保留所有原值，不裁剪到 100%，也不以 hit/(hit+miss) 替换原指标。该派生比率可在原始 JSON 中查看，但并未证明跨 pass 一致性，不能视为修复。

[NVIDIA Range and Precision 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision)说明：跨 replay pass 的工作分布变化可能造成越界指标，工具有意保留这些原值；减少同时采集的指标可能改善精度。本次复核符合这种问题的表现，但没有确定更底层的唯一原因。

主矩阵继续采用统一的 kernel replay 协议，仅添加质量标记：raw hit rate 超出 [0,100]%，或 abs((hits+misses)/total−1)>5%，则 hit rate 不进入有效结果解释。5% 是本实验的检查阈值，不是硬件精度保证。其它原始计数、LTC 与 HBM 仍完整保存，NCU 的单次/跨 pass 数据均按诊断测量解释。

## 原始文件与复现

- 首次异常：`../profiles/prefill/sq128_b64_sk65536/attempt_001/localized.*`。
- 诊断复测：`prefill_sq128_b64_sk65536/<实验名>.ncu-rep`、`.csv`、`.log`。
- 每次复测的完整 argv 和环境变量：同目录 `<实验名>_command.json`。
- 所有复测均调用 `benchmarks/profile_cute_dsl_localized_mla_ltc_target.py`。
- 复现时读取对应 command JSON，保留参数，将 `--export` 后的路径改为新文件位置，再执行 argv；不覆盖历史证据。
- 主矩阵使用 `--resume` 续测，原 13 个完整配置未重跑。首次失败的配置创建 `attempt_002`，历史失败保留。
