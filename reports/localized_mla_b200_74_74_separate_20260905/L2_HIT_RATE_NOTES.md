# L2 cache hit rate 测量记录（2026-09-05 UTC）

今天的结果表中，L2 hit rate 存在真实的原始读数越界与计数一致性警告，不能把“采集完成”理解为测量精度已经验证。
**baseline（standard）和实验组（localized）都出现过超过 100% 的 NCU hit rate。**
只请求这一个 metric 的后续测量中，localized 曾越界；baseline 的两个配置各 5 次未越界，但仍有波动。

## 两轮实验如何使用

| 内容 | 首轮 `localized_mla_b200_74_74_20260905` | 重测 `localized_mla_b200_74_74_separate_20260905` |
| --- | --- | --- |
| Decode 性能 | Sq=1/4 各 72 点新测；几何平均加速 1.0382× / 1.0285× | 未重测，引用首轮 |
| Prefill 性能 | 复用此前 74/74、Sq=128 dense 的 72 点 | 72 点全部重新计时，几何平均加速 0.9925× |
| LTC 与 L2/memory | 联合请求 11 项指标，36 份主报告，每份 4 pass | 分开采集；LTC 36 份/1 pass，L2/memory 36 份/4 pass |
| L2 质量标记 | 7/36：4 条 >100%，另 3 条仅计数不一致 | 8/36：3 条 >100%，另 5 条仅计数不一致；全部属于 prefill |

性能汇总使用首轮 decode 和重测 prefill；按分开采集协议引用硬件指标时使用重测目录。
首轮联合报告用于历史对照，不能与重测报告混合当成同一组独立采样。
两轮各覆盖 18 对边界 shape（36 个 mode/shape 记录），并非完整硬件指标矩阵。

## “原始命中率”和质量警告分别是什么

- **原始命中率**：NCU metric `lts__t_sector_hit_rate.pct`，直接取原始 CSV 的百分比，不是本项目根据 CSV 另造的百分比。
- **越界**：原始值不在 [0,100]%；本次越界值均为 >100%。这与真实缓存命中率的逻辑范围矛盾。
- **计数一致性警告**：本项目另行计算 `δ = 100 × ((hit + miss) / total − 1)`，若 `|δ| > 5%` 则标记。
  使用 `lts__t_sectors_lookup_hit.sum`、`lts__t_sectors_lookup_miss.sum`、`lts__t_sectors.sum`。
  **5% 是我们设置的检查阈值，不是 NCU 自带警告或 NVIDIA 精度规范。**
- 原始值保留，不截断到 100%，不删除越界样本，也不以 `hit/(hit+miss)` 替换为“修复值”。
  表中的 † 和图中的 N/A 表示不用于有效命中率比较，原始数据并未缺失。
- 没有触发这两条规则也不等于已经证明稳定或准确。只请求 hit rate 的实验没有单独检查 hit/miss/total 一致性。

## 两轮异常点对照

下表包含任一轮被标记的全部 10 个 mode/shape。† 表示该轮触发上述任一规则；δ 的单位为百分比，正值表示 hit+miss 多于 total。
每个命中率链接到那一轮的原始 CSV。第二轮有 5 个既有异常仍被标记、3 个新标记；第一轮另外 2 个标记在第二轮未再触发。

| Workload / Sq | B | Sk | 模式 | 首轮 L2 hit rate | 首轮 δ | 重测 L2 hit rate | 重测 δ |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| decode / 4 | 64 | 1048576 | standard | [60.37%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/decode/sq4_b64_sk1048576/attempt_001/standard.csv) | +15.29% | [60.00%](memory/profiles/decode/sq4_b64_sk1048576/attempt_001/standard.csv) | +1.35% |
| prefill / 128 | 2 | 1008576 | localized | [107.74%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b2_sk1008576/attempt_001/localized.csv) | +10.72% | [107.26%†](memory/profiles/prefill/sq128_b2_sk1008576/attempt_001/localized.csv) | +10.11% |
| prefill / 128 | 2 | 1008576 | standard | [114.64%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b2_sk1008576/attempt_001/standard.csv) | +20.30% | [82.07%†](memory/profiles/prefill/sq128_b2_sk1008576/attempt_001/standard.csv) | -13.04% |
| prefill / 128 | 16 | 1008576 | localized | [78.86%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b16_sk1008576/attempt_001/localized.csv) | -16.28% | [92.12%](memory/profiles/prefill/sq128_b16_sk1008576/attempt_001/localized.csv) | -2.03% |
| prefill / 128 | 16 | 1008576 | standard | [82.61%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b16_sk1008576/attempt_001/standard.csv) | -11.25% | [102.53%†](memory/profiles/prefill/sq128_b16_sk1008576/attempt_001/standard.csv) | +11.71% |
| prefill / 128 | 64 | 65536 | localized | [102.20%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b64_sk65536/attempt_002/localized.csv) | +5.54% | [108.47%†](memory/profiles/prefill/sq128_b64_sk65536/attempt_001/localized.csv) | +12.09% |
| prefill / 128 | 64 | 65536 | standard | [98.82%](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b64_sk65536/attempt_002/standard.csv) | +4.28% | [81.69%†](memory/profiles/prefill/sq128_b64_sk65536/attempt_001/standard.csv) | -13.21% |
| prefill / 128 | 64 | 524288 | localized | [106.56%†](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b64_sk524288/attempt_001/localized.csv) | +14.83% | [86.05%†](memory/profiles/prefill/sq128_b64_sk524288/attempt_001/localized.csv) | -7.02% |
| prefill / 128 | 64 | 524288 | standard | [77.91%](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b64_sk524288/attempt_001/standard.csv) | -0.21% | [74.07%†](memory/profiles/prefill/sq128_b64_sk524288/attempt_001/standard.csv) | -8.46% |
| prefill / 128 | 64 | 1008576 | standard | [76.26%](../localized_mla_b200_74_74_20260905/ncu_boundary/profiles/prefill/sq128_b64_sk1008576/attempt_001/standard.csv) | +1.47% | [85.67%†](memory/profiles/prefill/sq128_b64_sk1008576/attempt_001/standard.csv) | +8.79% |

首轮 B=64、Sq=128、Sk=65536 localized 在表中为主矩阵最终采用的 **102.20%（attempt_002）**。
其第一次采集是 **105.68%（attempt_001）**，当时严格校验中止；首次报告仍保留。
随后允许保留原值并标记异常，以完成其他配置的采集；没有把重试后的值视为修复成功。

具体例子：重测 B=64、Sq=128、Sk=65536 localized 的 total=25,233,132,772、hit=27,369,891,487、miss=912,755,217 sectors。
NCU 原始 hit rate 为 108.47%，hit 已经大于 total，且 δ=+12.09%。
同配置 standard 为 81.69%，虽然未越界，但 δ=−13.21%，因此该点不能用两者的差值推导 localized 对真实命中率的改善。

## 已保存的早期诊断复测

以下均为 localized prefill，B=64、Sq=128、Sk=65536。

| 诊断 | 实际请求 | 原始 hit rate | Replay pass |
| --- | --- | ---: | ---: |
| 首次联合采集 | 原 11 项指标 | 105.68% | 4 |
| repeat_combined | 原 11 项指标再测 | 113.08% | 4 |
| l2_only | duration、hit rate、total、hit、miss，共 5 项 | 102.59% | 4 |
| l2_hit_miss | 仅 hit 与 miss，共 2 项 | 未请求 hit rate | 3 |
| application_replay | 原 11 项，改用 application replay | 103.56% | 4 |

**`l2_only` 这个历史文件名不代表“只请求 hit rate 一个 metric”。**
这些复核没有消除越界；application replay 的一次复核也未消除异常。
原始报告、命令和细节见[首轮诊断目录](../localized_mla_b200_74_74_20260905/ncu_boundary/diagnostics/README.md)。

## 后续真正只请求 hit rate 的测量

所有后续测量的 `--metrics` 均只有 `lts__t_sector_hit_rate.pct`，沿用 random BF16、seed=42、每次 attention 前清 L2，
以及 `--cache-control all --clock-control boost --replay-mode kernel`。
**只请求一个 metric，实际仍需 3 个 replay pass 生成一个完整样本。** pass 不是独立测量样本；
NCU 自动导出的依赖列也不代表手动请求了额外指标。

### Baseline：原始证据保存在仓库外

| 模式 | B / Sq / Sk | 5 次原始 hit rate | >100% | 每个样本 pass |
| --- | --- | --- | ---: | ---: |
| standard | 16 / 128 / 1008576 | 85.20%、92.71%、86.93%、89.31%、90.04% | 0/5 | 3 |
| standard | 64 / 128 / 1008576 | 75.94%、81.39%、74.11%、83.73%、85.67% | 0/5 | 3 |

每个配置启动 5 个独立进程，每进程 3 次预热与 1 个完整样本；每组共 15 次被测 attention 执行加 15 次预热，
不含初始化和清缓存 kernel。B=16 的此前十指标读数为 102.53%；本次单指标 5 次未复现越界。
B=64 最大配置单指标测量也未越界，但读数跨度为 11.56 个百分点，不能认定已稳定。

- B=16 原始数据、脚本与复现：[/tmp/mla_baseline_l2_hit_rate_20260905_n3dz6bru/README.md](/tmp/mla_baseline_l2_hit_rate_20260905_n3dz6bru/README.md)。
- B=64 原始数据、脚本与复现：[/tmp/mla_baseline_max_l2_hit_rate_20260905_qavbj3i_/README.md](/tmp/mla_baseline_max_l2_hit_rate_20260905_qavbj3i_/README.md)。
- 编排脚本均为各目录 `run.py`；实际 target 为 `benchmarks/profile_cute_dsl_localized_mla_ltc_target.py`。
  每次精确 argv/env 在 `trial_N_command.json`，并提供等价 `.sh`。
- 整理时重新核对了 10 份原始 CSV 与命令。本仓库只记录结论和路径，未搬入这些补测原始文件。
  `/tmp` 路径是本机临时存储；本目录 `l2_hit_rate_review.json` 保留结果值和外部 summary 的 SHA-256，便于追溯。

### Localized：只保留已报告的历史结论

此前只测 hit rate 的重复实验：B=64、Sq=128、Sk=65536；预热 3/20/100 次，
每档两个进程、每进程 100 个完整样本。共 600 个样本，160 个超过 100%；
三档分别为 52/200、53/200、55/200。增加预热和连续采样未消除单次越界。

**该次原始数据、脚本和重复运行参数已按用户要求删除，本次没有恢复。**
上面数字来自当时已报告的结果，整理时无法再次用原始文件独立核验，证据级别与现存报告不同。
它与上面的 baseline 补测不是同一个 Sk，也不是相同样本量，不能据此比较两种模式的越界概率。

## 可以判断什么，尚不能判断什么

已确认：多指标采集下 baseline 与 localized 都可能越界；单指标请求也不保证单 pass；
localized 的历史单指标实验仍曾越界，baseline 当前两个配置各 5 次没有观察到越界。
这些结果不足以确定是 kernel、localized 布局、74/74 拓扑、NCU 或驱动中的哪一环造成。

NVIDIA 文档说明，多 pass 间工作分布变化可能使分别采集的 hit 与 query 合成出越界比例；
其他 GPU 引擎访问 L2/DRAM 等共享资源也可能影响测量。
[官方 Range and Precision](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision)

跨 pass 的计数不一致是当前优先怀疑的解释，但我们未确认 hit rate 的底层分子、分母各自在哪个 pass，
因此不能把“3 或 4 pass”直接当作根因证明。相同输入不保证完全相同的调度与缓存访问过程，这也尚未被本实验单独验证。
那个 localized 异常点的 NCU duration 约 93.4 ms，不符合典型“kernel 仅运行几微秒”的情形。
文档建议增加的是每次被测 launch 的工作量；增加预热次数或多采集若干单独 launch 并不等价。

已采用冷 L2 和 `--cache-control all`，有助于控制各 pass 的初始缓存状态；
它不保证执行中的访问次序完全一致。[官方 Cache Control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)
GPU 串行运行及空闲检查也不能单独排除所有共享资源干扰。

后续若验证根因，应优先检查底层计数的 pass 分配与重复性。本次整理没有新跑 GPU 实验。
独立性能结果与 L2 计数质量是两类证据；L2 告警不会自动否定独立计时，也不能反过来以计时稳定证明 L2 准确。
