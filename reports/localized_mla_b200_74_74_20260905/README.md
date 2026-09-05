# B200 74/74 SM MLA 补测（2026-09-05 UTC）

整理入口：[L2 hit rate 完整记录](../localized_mla_b200_74_74_separate_20260905/L2_HIT_RATE_NOTES.md) / [清理清单](cleanup_manifest.json) / [整理后核验](cleanup_verification.json) / [资源摘要](resource_summary.json)。
元数据以各组 `matrix_summary.json` 中的 `configs[].profiles[mode].target_metadata` 为准；原始日志也保留 `TARGET_METADATA`。
已删除逐点重复 metadata 文件、冗长资源采样和可再生缓存；正式 NCU 报告、原始 CSV、异常复测证据、命令及性能数据均保留。
本目录提供今天的 decode 性能结果和首轮联合指标历史；新 prefill 性能及分开采集的指标请见[重测目录](../localized_mla_b200_74_74_separate_20260905/README.md)。

当前状态：**全部采集完成，存在 L2 指标质量警告**。

- [结果表](results.md) / [机器可读汇总](summary.json)
- 新增 decode Sq=1/4 性能各 72 点，共 144 点；全部实测为 74/74 SM。
- 18 对边界指标配置全部完成，36 份主 NCU 报告，每份 11 项指标、4 个 replay pass。
- 7 个 mode/shape 的 L2 百分比触发质量标记（其中 4 个越界，另 3 个计数不一致）；
  保留原值，在表格中加 †、在 CSV 中给出质量字段、在 hit-rate 图中显示 N/A。
  这些值不可直接解释为有效命中率，详见 [异常复核](ncu_boundary/diagnostics/README.md)。
- Prefill Sq=128 dense 的 72 点性能沿用此前同拓扑结果，没有重测。
- [产物核对](verification.json)通过：36 份主报告各含一个 kernel row，JSON 的全部
  11 项数值与原始 CSV 相符；来源文件 SHA-256 未变化。注意该核对不代表异常 L2 百分比有效。

Sq=1 性能在 11:15:40–11:29:11 UTC 完成；Sq=4 在 11:30:13–11:44:34 UTC 完成。
NCU 主流程为 11:44:52–11:54:51 UTC，包含异常检查与暂停。
两组 decode 的几何平均加速分别为 **1.0382× / 1.0285×**，
最大 block 间加速比波动为 **3.13% / 5.10%**。

## 范围与测量点

- Decode：补测 Sq=1、Sq=4 的随机 BF16 性能矩阵，各 72 点。
  B=2,4,8,16,32,64；Sk=512 到 1,048,576，按 2 倍递增。
- Prefill：只测 Sq=128 的硬件指标；已有的 74/74 dense 性能结果直接复用，
  不重跑性能，也不将 70/78 的 iter_001 当作本轮性能数据。
- 每个 workload 的硬件指标选择 6 点：B=64 下 Sk=512,65,536,524,288,Sk_max；
  以及 Sk=Sk_max 下 B=2,16。共 18 对配置、36 次主结果 NCU 调用。
  另保留首次异常报告和 4 份诊断复测报告，合计 41 份 `.ncu-rep`。
  Decode 的 Sk_max=1,048,576，prefill 的 Sk_max=1,008,576。
- standard 和 localized 使用同一个 shape、相同随机种子和逻辑输入。

## 硬件与资源限制

实际探测：NVIDIA B200，148 SM，两个分区各 74 SM（37 个 2-SM cluster）。
GPU UUID：`GPU-831fae2c-f9be-da45-6749-cf96b741d59b`。
完整环境见 `environment.json`，资源采样端点与极值见 `resource_summary.json`。

开始时有 20 个逻辑 CPU，约 217 GiB 主机可用内存，50 GiB 空闲磁盘，
GPU 空闲。限制 `MAX_JOBS=5`、`FLASHINFER_NVCC_THREADS=1`、
`OMP_NUM_THREADS=5`。性能测量、正确性验证和 NCU 按顺序使用 GPU。
随机长序列负载期间观察到软件功耗限制，power limit 为 1000 W；未更改外部
application clocks。最大容量性能点的最低可用显存为 Sq=1 的 10.84 GiB、
Sq=4 的 10.01 GiB。实验结束后 GPU 显存占用和利用率均回到 0。

## 协议

性能实验沿用 70/78 参考矩阵的协议：随机 BF16（seed=42），固定连续页表，
H=128，latent/RoPE=512/64，page size=64，PDL 关闭。仅测 kernel，
分配、初始化与 scatter 均不计时。20 次 paired warmup，四个平衡 AB/BA/BA/AB
block，每个模式 500 ms warmup、1,000 ms repeat，至少 20 samples，
以四个 block median 的中位数作为最终时间；每个 sample 清 L2。

硬件指标使用 Nsight Compute，独立进程分别测 standard/localized，
3 次 warmup（每次之前清 L2），最终清 L2 并同步后启用 profiler，
仅采集一次 attention launch。NCU 选项为 `--cache-control all`、
`--clock-control boost`、`--replay-mode kernel`。
按配置交替 standard/localized 的进程顺序。所有主结果均为 4 个 replay pass。
每个 localized 进程必须通过
74/74 检查，否则直接失败。

首轮实际联合采集以下 11 个指标（此后已按要求另做分开采集，见重测目录）：

```
gpu__time_duration.avg
lts__t_requests_srcunit_ltcfabric.sum
lts__t_sector_hit_rate.pct
lts__t_sectors.sum
lts__t_sectors_lookup_hit.sum
lts__t_sectors_lookup_miss.sum
dram__bytes_read.sum
dram__bytes_write.sum
dram__bytes_read.sum.per_second
dram__bytes_write.sum.per_second
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

L2 质量检查：raw hit rate 超出 [0,100]%，或
`abs((hits+misses)/total-1)>5%` 时标记异常；5% 是本实验的质量阈值，
不是硬件精度保证。异常 L2 数据保留，hit/(hit+miss) 不作为修复值。
脚本默认继续保存带质量警告的采集结果，避免重跑其它指标；旧 memory 脚本的
默认严格越界校验保持不变。

NCU duration 只作为诊断数据，不作为独立性能实验结论。HBM 带宽是速率，
不能把带宽下降直接解读为性能下降；需同时查看 HBM 总字节数和独立计时结果。
边界点覆盖也不代表完整指标矩阵。

## 实验脚本位置

以下均为仓库根目录的相对路径：

| 脚本 | 用途 |
| --- | --- |
| `benchmarks/bench_cute_dsl_localized_mla.py` | Decode 性能矩阵；新增可选 SM 分区检查 |
| `benchmarks/localized_mla_benchmark.py` | Decode 成对数据与 kernel 准备 |
| `benchmarks/localized_mla_prefill_benchmark.py` | 原 prefill dense 性能实验的数据与 kernel 准备 |
| `benchmarks/profile_cute_dsl_localized_mla_boundary.py` | 18 个边界点 NCU 编排、指标校验、CSV/图表与断点续测 |
| `benchmarks/profile_cute_dsl_localized_mla_ltc_target.py` | 独立进程中的 decode/prefill target，仅圈定一次 attention launch |
| `benchmarks/profile_cute_dsl_localized_mla_memory.py` | 复用的 NCU 解析、L2/HBM 派生量与配置校验 |
| `benchmarks/validate_cute_dsl_localized_mla_profile_target.py` | 验证独立 target 的 output/LSE 与原 benchmark setup 逐位一致 |
| `benchmarks/plot_cute_dsl_localized_mla.py` | Decode 性能图 |
| `reports/localized_mla_b200_74_74_20260905/summarize.py` | 从完成的 JSON 离线重建结果表；支持 `--output-root` |
| `reports/localized_mla_b200_74_74_20260905/reproduce.sh` | 使用新输出目录完整复现；不重测 prefill 性能 |

## 复现方法

在具备仓库当前 `.venv`、CUDA/NCU 和 RM localized allocation 支持的机器上，
从仓库根目录运行。脚本会先输出 CPU/内存/负载/磁盘/GPU 状态。
当前 GPU 必须空闲且实际分区为 74/74。

```bash
bash reports/localized_mla_b200_74_74_20260905/reproduce.sh \
  reports/localized_mla_b200_74_74_reproduction
```

如果只需要指标测量，不重测 decode 性能：

```bash
export MAX_JOBS=5 FLASHINFER_NVCC_THREADS=1 OMP_NUM_THREADS=5
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python benchmarks/profile_cute_dsl_localized_mla_boundary.py \
  --device 0 --expected-partition-sm-counts 74 74 \
  --output-root reports/localized_mla_b200_74_74_metrics_reproduction
```

默认输入文件仅用于选择 shape 和 split_kv：原 decode 随机矩阵及
`reports/localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json`。
其中 decode 默认来源为旧 78/70 性能矩阵，**不会读取或复用其性能值作为
本轮测量结果**。完整复现脚本则显式使用本轮新生成的 decode shape manifest。

中断后使用相同参数及 `--resume` 继续，已完成配置跳过，失败配置保存旧 attempt
并创建新 attempt。只重建 CSV 和图表可加 `--summarize-only`，不运行 GPU kernel。

每个 profile 的精确 argv、环境变量、NCU 原始 CSV/二进制及日志
保存在 `ncu_boundary/profiles/<workload>/<shape>/attempt_NNN/`。
`matrix_summary.json` 记录来源文件 SHA-256、脚本 SHA-256、NCU 版本、
开始/结束时间、每项指标的原始值和单位，以及实际 replay pass 数。
二进制 `.ncu-rep` 按仓库惯例保留在本地并被 Git 忽略。

## 已有 prefill 性能结果

沿用 `../localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json`，
共 72 点，全部 `resident_partition_clusters=[37,37]`，2026-09-04 完成。
该文件没有被本轮修改。

## 结果文件位置

- `decode/sq1/post_flops.json`、`decode/sq4/post_flops.json`：新增性能数据。
- `decode/sq*/figures/performance_comparison.png`：性能热力图。
- `ncu_boundary/matrix_summary.json`：完整指标、质量标记、命令和来源记录。
- `ncu_boundary/boundary_metrics.csv`：18 点指标表，含原始 L2 百分比和可用性标记。
- `ncu_boundary/figures/`：decode Sq=1/4、prefill Sq=128 的指标图。
- `ncu_boundary/profiles/`：逐模式原始 NCU、CSV、command 与日志；metadata 已合并在 matrix_summary.json 中。
- `ncu_boundary/diagnostics/`：越界 L2 的重复采集、缩减指标与 application replay 复核。
- `validation/correctness.json`：三个 workload、两种 mode 的 output/LSE 逐位一致性验证。
- `execution_commands.json`：本轮执行命令，包含恢复命令。
- `environment.json`、`resource_summary.json`：环境与资源采样摘要。

检查了所有修改脚本的 Ruff lint/format、Python 语法及复现 shell 语法。
Python 依赖版本、NVCC 并发参数和 GPU UUID 均在 `environment.json` 中。

## 整理后的文件保留规则

- 原始 `.ncu-rep` 与 CSV 保留，包括所有越界值、首次失败 attempt 和诊断复测；二进制仍按仓库规则被 Git 忽略。
- `*_command.json`、NCU 日志、正式 `matrix_summary.json`、性能 JSON、结果表和图保留。
- 主矩阵中逐点 `*_metadata.json` 在确认与 `target_metadata` 完全相同后删除；完整元数据仍可从汇总和原始日志读取。
- `resource_samples.jsonl` 已提炼为 `resource_summary.json`，保留采样次数、时间端点、极值及特殊事件；完整资源时间线不再保留。
- 删除路径、原文件 SHA-256、字节数与理由见 `cleanup_manifest.json`。复现脚本运行在新目录时仍会生成原有的完整产物。
- 整理不会改写测量数值；离线重建结果时不会重新运行 GPU kernel。
