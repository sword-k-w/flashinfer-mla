# B200 MLA：B=64、Sk=32768 NCU 补测

结果入口：[实测表和 sector 原始计数](data/results.md)、[汇总 JSON](data/summary.json)、[逐次指标 CSV](data/metrics.csv)、[核验与文件哈希](data/verification.json)。

**采集完成**：2026-09-07 08:39:35–08:44:06 UTC，36 份报告均已核对；LTC 每份 1 pass，memory 每份 4 pass。采集期间相关源码哈希未变化。

| 配置 | baseline LTC requests 中位数 | localized LTC requests 中位数 | LTC 减少 | baseline → localized L2 sector hit rate 中位数 |
| --- | ---: | ---: | ---: | --- |
| Decode Sq=1 | 18,925,190 | 353,313 | 98.13% | 37.07% → 46.97% |
| Decode Sq=4 | 44,183,145 | 1,366,063 | 96.91% | 64.31% → 84.11% |
| Prefill Sq=128 dense | 128,623,443 | 43,030,716 | 66.55% | 异常，不能给出可信比较 |

Decode 的 12 份 L2 报告均通过现有 QA 检查。Prefill 的原始 sector hit rate：baseline **84.28%、118.02%、107.11%**；localized **97.62%、100.82%、100.02%**。其中 **5/6 份**触发 QA：4 份超过 100%，另 baseline 84.28% 的 hit+miss 比 total 少 12.96%。完整原值保留，不能据此判断 prefill 命中率提升。

资源采样显示可用主存最低 216.54 GiB、临时盘余量最低 32.88 GiB；采样时 CPU 利用率最高约 7.71%。没有资源压力。

## 配置和测量口径

- Baseline：`standard`，常规 KV 分配及原调度；实验组：`localized`，分区本地 KV 分配及 partition-aware 调度。复用既有 target 和 kernel，没有修改实现。
- B=64、Sk=32768；decode Sq=1、Sq=4；prefill Sq=128 dense。H=128、latent/RoPE=512/64、page=64、BF16、split_kv=1、PDL 关闭。
- 相同逻辑随机输入，seed=42。每个 target 独立进程，3 次预热；每次 attention 前用 Triton benchmark cache 清 L2。只在 `cudaProfilerStart/Stop` 之间采集一个 attention launch，不含分配、随机初始化、scatter、清缓存。
- 当前 GPU 的 SM 分区为 **74/74**；逐个 localized target 强制核验。2026-09-07 更早的 Sk=65536 异常复测曾为 70/78，不混入本次数据。
- LTC 和 L2/memory 是两个独立 NCU 进程、两份报告。每种配置、指标组、模式做 **3 次**独立采集，共 **36 份**报告；顺序为 standard/localized、localized/standard、standard/localized。所有 GPU 工作串行运行。
- NCU：`--profile-from-start off --launch-count 1 --cache-control all --clock-control boost --replay-mode kernel`。
- LTC 只请求 `lts__t_requests_srcunit_ltcfabric.sum`，与此前实验一致，单位 **requests**；不将请求数乘以 sector 大小冒充字节流量。NCU 自动导出的派生项不算额外请求。
- L2 沿用此前独立 memory 实验的十指标集合，命中率使用 **sector** 指标 `lts__t_sector_hit_rate.pct`，同时保留 `lts__t_sectors.sum`、`lts__t_sectors_lookup_hit.sum`、`lts__t_sectors_lookup_miss.sum`。其余六项为 duration、HBM read/write bytes、read/write bandwidth、DRAM peak-sustained throughput；精确列表见 `data/results.json` 的 `settings.groups.memory`。
- L2 保留原始百分比，不截断、不用 hit/(hit+miss) 修复。越界或 `abs((hit+miss)/total-1)>5%` 标记为 QA 异常；5% 是已有项目阈值，不是 NVIDIA 精度保证。重复值全列出，异常值不用于有效命中率提升结论。
- LTC 减少率 = `100 × (1 − median(localized requests)/median(standard requests))`。NCU replay pass 不是独立样本；duration 不作为正常运行性能结论。

## 复现

从 `/workspace/flashinfer-mla` 运行。使用现有 `.venv`；需空闲 B200、NCU 计数器权限和 RM localized allocation 支持。输出目录必须不存在，避免覆盖原始实验。

```bash
nproc
free -h
uptime
df -h /tmp /workspace
nvidia-smi
.venv/bin/python reports/mla_b64_sk32768_ncu_20260907/run.py \
  --output-root reports/mla_b64_sk32768_ncu_reproduction \
  --trials 3 --expected-partition-sm-counts 74 74
.venv/bin/python reports/mla_b64_sk32768_ncu_20260907/summarize.py \
  --output-root reports/mla_b64_sk32768_ncu_reproduction
```

仅重建本次结果和核验文件（无 GPU 工作）：

```bash
.venv/bin/python reports/mla_b64_sk32768_ncu_20260907/summarize.py \
  --output-root reports/mla_b64_sk32768_ncu_20260907/data
```

每个 `data/<workload>_sq<Sq>/<ltc|memory>/<mode>_<trial>.command.json` 保存精确 argv、cwd、环境覆盖。单次重放应将 argv 中 `--export` 后的路径改为新路径，以保留旧报告。原始报告可离线导出：

```bash
ncu --import reports/mla_b64_sk32768_ncu_20260907/data/decode_sq1/ltc/standard_01.ncu-rep \
  --page raw --csv --print-units base
```

## 环境与脚本

本轮 commit `954d72f50d06611f9c201222dc7fd285a69e3b48`；驱动 595.45.04；NCU 2026.1.1；Python 3.12.3；Torch 2.12.1+cu132；Triton 3.7.1；CuTe DSL 4.7.1。实测环境、GPU UUID、起止时间及源码 SHA-256 以 `data/results.json` 为准。与此前 74/74 实验共有的 63 个源码文件哈希一致。

- `run.py`：串行采集、命令及环境记录、元数据校验、原始异常值保留、资源采样。
- `summarize.py`：离线核对每份原始 CSV、requested metrics、replay pass、日志元数据、报告存在性及源码哈希；生成表格和机器可读汇总。
- 复用 `benchmarks/profile_cute_dsl_localized_mla_ltc_target.py` 和 `profile_cute_dsl_localized_mla_memory.py` 的 target、命令构造及解析；复用旧 `run_experiments.py` 的资源采样函数。
- 20 个逻辑 CPU；运行前可用内存约 217 GiB、临时盘余量约 33 GiB。设置 `MAX_JOBS=5`、`FLASHINFER_NVCC_THREADS=1`、`OMP_NUM_THREADS=5`。每次子进程结束及运行超过 15 秒时记录资源；采样极值不代表连续监控的绝对峰值。
- 保存全部 `.ncu-rep`、原始 `.csv`、`.log`、`.command.json`、`results.json` 和 `resource_samples.jsonl`。二进制报告及日志遵循仓库既有 Git ignore 规则，文件仍保留在本地。

全程使用 NCU，未使用 G-Watch。
