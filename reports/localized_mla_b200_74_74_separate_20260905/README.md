# B200 74/74 SM：Prefill 性能重测与独立 LTC、L2/memory 实验

整理入口：[L2 hit rate 完整记录](L2_HIT_RATE_NOTES.md) / [清理清单](cleanup_manifest.json) / [整理后核验](cleanup_verification.json) / [资源摘要](resource_summary.json)。
元数据以各组 `matrix_summary.json` 中的 `configs[].profiles[mode].target_metadata` 为准；原始日志也保留 `TARGET_METADATA`。
已删除逐点重复 metadata 文件、冗长资源采样和可再生缓存；正式 NCU 报告、原始 CSV、异常复测证据、命令及性能数据均保留。
本目录提供今天重新测量的 prefill 性能与分开采集的 LTC、L2/memory；decode 性能见[首轮目录](../localized_mla_b200_74_74_20260905/README.md)。

本轮是用户要求的重新采集，输出独立保存，不覆盖上轮联合 11 指标报告。
**本轮全部采集完成；L2 有质量警告，不能将全部采集值视为有效命中率。**

- [完整结果表](results.md) / [汇总 JSON](summary.json) / [原始数据核验](verification.json)。
- Prefill Sq=128 dense：72 点重新计时，2026-09-05 12:08:09–13:01:18 UTC；
  几何平均加速 0.9925×，localized 更快 12/72 点，最大 block 加速比波动 2.35%。
  72 点输出/LSE 逐位一致，全部 74/74 SM，每个 block 每模式至少 20 个样本。
- LTC：18 对配置、36 份独立单指标报告，全部 1 个 replay pass。
- L2/memory：18 对配置、36 份独立十指标报告，全部 4 个 replay pass。
  **8/36 条 L2 记录被标记，全部属于 prefill**；其中 3 条命中率超过 100%，
  另 5 条命中率在范围内但 hit+miss 与 total 的偏差超过 5%。原值保留并标注。
- 已核对 72 份 NCU 原始 CSV：每份一个 kernel row，全部请求指标与 JSON 一致，
  LTC 和 L2/memory 原始报告确实分开。脚本及 kernel 源码哈希未变化。

## 范围和协议

- **Prefill 性能**：重新执行 Sq=128 dense 全部 72 点，不复用旧性能值。
  B=2,4,8,16,32,64；Sk=512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1008576。
  使用原 `bench_cute_dsl_localized_mla_prefill.py` / `PreparedMLAPrefillCase`。
  BF16 随机输入 seed=42、H=128、latent/RoPE=512/64、page=64、split_kv=1、PDL 关闭。
  原 `paired-cold-l2-v2`：20 次 paired warmup；4 个 AB/BA/BA/AB block；
  每模式 warmup=500 ms、repeat=1000 ms、至少 20 samples；每 sample 清 L2。
  初始化、分配与 scatter 不计入 kernel 性能。每点核对 SM 分区为 74/74。
- **独立 LTC 实验**：18 对边界配置，36 份主报告，`--metrics` **只请求**
  `lts__t_requests_srcunit_ltcfabric.sum`，不附加 duration 或 L2/memory 指标。
  NCU 原始导出会自动附带该计数器的 avg/max/min 和 `gpu__time_duration.sum`；
  这些不是额外请求的指标，也不用于本轮独立性能结论。
- **独立 L2/memory 实验**：同样 18 对配置、36 份主报告；请求原来的 10 项指标，
  **不包含 LTC**：duration、L2 hit rate/total/hit/miss、HBM read/write bytes、
  read/write bytes-per-second、DRAM peak-sustained throughput percentage。
- 两组指标均覆盖 decode Sq=1/4 和 prefill Sq=128 dense。
  每个 workload 六点：B=64 时 Sk=512,65536,524288,Sk_max，以及 Sk_max 时 B=2,16。
  Decode Sk_max=1048576；prefill Sk_max=1008576。
  Decode 性能不重测，其上轮 JSON 仅为 shape/split_kv manifest。
- 两组指标按顺序运行，每个模式/配置均启动独立进程，只圈定一次 attention launch。
  使用同一 target、相同逻辑随机输入 seed=42、3 次预热，每次 attention 前清 L2。
  NCU 显式设置 `--cache-control all --clock-control boost --replay-mode kernel`。
  模式顺序按配置交替；每个 localized 进程核对 74/74。
  每组实际 replay pass 数由 NCU 原始报告读取，不预设相等。

本轮的分开采集恢复了 LTC 单指标与 L2/memory 十指标两次调用。
L2/memory 的指标集合和采集参数与旧 70/78 矩阵一致。
LTC 使用相同的随机输入、冷 L2 和显式 NCU 参数，以保证本轮两组实验的输入条件一致；
这不是最早 B200 LTC 单点命令的逐字重放（最早命令没有显式指定随机初始化、cold-L2、clock-control）。

## 数据质量

原始 `.ncu-rep`、CSV、日志与命令完整保留。采集完成不等于全部指标通过有效性检查。
L2 报告值超出 [0,100]%，或 `abs((hits+misses)/total-1)>5%` 会被标记；
5% 为实验质量检查阈值，不是 NVIDIA 精度规范。不截断百分比，不用 hit/(hit+miss) 替代修复。
标记值不用于有效 L2 命中率差异结论。NCU duration 仅供诊断，性能结论来自独立计时。

## 脚本位置与复现

从 `/workspace/flashinfer-mla` 运行，使用仓库现有 `.venv`、CUDA 13.2、NCU 2026.1.1。
GPU 必须空闲且具备 RM localized allocation 支持。没有使用 G-Watch。
运行前检查 CPU、可用内存、负载、磁盘和 GPU；限制 MAX_JOBS=5、NVCC_THREADS=1、OMP_NUM_THREADS=5。
当前机器 20 个逻辑 CPU；采集期间 GPU 工作负载串行运行，每 30 秒记录资源状态。

```bash
nproc
free -h
uptime
df -h /tmp /workspace
nvidia-smi
.venv/bin/python reports/localized_mla_b200_74_74_separate_20260905/run_experiments.py \
  --output-root reports/localized_mla_b200_74_74_separate_reproduction
.venv/bin/python reports/localized_mla_b200_74_74_separate_20260905/summarize.py \
  --output-root reports/localized_mla_b200_74_74_separate_reproduction
```

`run_experiments.py` 会先运行完整 prefill 性能，再分别运行 LTC 和 memory，最后画 prefill 图。
本轮初次 prefill 已先启动，编排器用 `--existing-prefill-pid` 接续等待该进程；
正常复现**不需要**这个参数。`execution_commands.json` 保存全部阶段的等价完整 argv，
逐 NCU 进程的精确 argv 和环境另存于每个 `*_command.json`。

只采集一组指标（以下 `ltc` 可替换为 `memory`）：

```bash
export MAX_JOBS=5 FLASHINFER_NVCC_THREADS=1 OMP_NUM_THREADS=5
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python benchmarks/profile_cute_dsl_localized_mla_separate.py \
  --experiment ltc --device 0 --expected-partition-sm-counts 74 74 \
  --timing-sources \
    reports/localized_mla_b200_74_74_20260905/decode/sq1/post_flops.json \
    reports/localized_mla_b200_74_74_20260905/decode/sq4/post_flops.json \
    reports/localized_mla_b200_74_74_separate_20260905/prefill/timing.json \
  --output-root reports/localized_mla_b200_74_74_ltc_reproduction
```

中断恢复可对该命令附加 `--resume`；已完成配置跳过，失败配置创建新 attempt，旧报告保留。

| 脚本 | 用途 |
| --- | --- |
| `benchmarks/bench_cute_dsl_localized_mla_prefill.py` | 原 Sq=128 dense 性能脚本；本轮仅新增可选 SM 检查 |
| `benchmarks/localized_mla_prefill_benchmark.py` | 原 prefill 数据准备、kernel 和布局 |
| `benchmarks/profile_cute_dsl_localized_mla_separate.py` | 新增独立 LTC / memory 编排；两组绝不合并 |
| `benchmarks/profile_cute_dsl_localized_mla_ltc_target.py` | 上轮已验证的独立进程 target；本轮未修改 |
| `benchmarks/profile_cute_dsl_localized_mla_memory.py` | 原 10 指标集合、NCU 命令构造与解析 |
| `benchmarks/profile_cute_dsl_localized_mla_boundary.py` | 只复用边界选点，不调用联合采集流程 |
| `benchmarks/plot_cute_dsl_localized_mla_prefill.py` | 原 prefill 性能绘图 |
| 本目录 `run_experiments.py` | 顺序编排、资源采样、环境与命令留档 |
| 本目录 `summarize.py` | 离线验证原始报告、输出表格和指标图 |

## 产物位置

- `prefill/timing.json`、`timing.csv`、`run.log`、`figures/`：新性能数据。
- `ltc/` 和 `memory/`：完全独立的 `matrix_summary.json`、`metrics.csv`、`run.log`、`figures/`。
- 每组 `profiles/<workload>/<shape>/attempt_NNN/`：逐模式原始 `.ncu-rep`、`.csv`、`.log`、command；metadata 已合并在 matrix_summary.json 中。
- `environment.json`、`resource_summary.json`、`execution_commands.json`：环境、脚本 hash、资源摘要和执行命令。
- `software_and_kernel_sources.json`：本轮软件版本、GPU UUID/驱动/功耗限制、63 个相关源码文件的 SHA-256。
- `results.md`、`summary.json`、`verification.json`：最终结果和原始数据核验。

## 整理后的文件保留规则

- 原始 `.ncu-rep` 与 CSV 保留，包括所有越界值、首次失败 attempt 和诊断复测；二进制仍按仓库规则被 Git 忽略。
- `*_command.json`、NCU 日志、正式 `matrix_summary.json`、性能 JSON、结果表和图保留。
- 主矩阵中逐点 `*_metadata.json` 在确认与 `target_metadata` 完全相同后删除；完整元数据仍可从汇总和原始日志读取。
- `resource_samples.jsonl` 已提炼为 `resource_summary.json`，保留采样次数、时间端点、极值及特殊事件；完整资源时间线不再保留。
- 删除路径、原文件 SHA-256、字节数与理由见 `cleanup_manifest.json`。复现脚本运行在新目录时仍会生成原有的完整产物。
- 整理不会改写测量数值；离线重建结果时不会重新运行 GPU kernel。
