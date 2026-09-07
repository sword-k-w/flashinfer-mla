# H200 MLA：B=64、Sk=32768 NCU 补测

结果入口：[逐次结果与 sector 原始计数](data/results.md)、[汇总 JSON](data/summary.json)、
[逐次指标 CSV](data/metrics.csv)、[原始记录](data/results.json)、[核验及文件哈希](data/verification.json)。

主采集完成于 **2026-09-07 13:35:55–13:40:21 UTC**，36 份报告均已核验，
LTC 每份 1 pass，memory 每份 4 passes。被测源码与上一轮 H200 实验一致。

| 配置 | baseline LTC requests 中位数 | compact LTC requests 中位数 | LTC 减少 | baseline → compact L2 sector hit rate 原始中位数 |
| --- | ---: | ---: | ---: | --- |
| Decode Sq=1 | 19,035,409 | 255,981 | **98.66%** | **1.14% → 1.12%**，基本不变 |
| Decode Sq=4 | 36,292,519 | 551,569 | **98.48%** | 65.56% → 74.77% **†** |
| Prefill Sq=128 dense | 210,968,563 | 17,956,238 | **91.49%** | 异常，不能可信比较 |

**† Sq=4 的中位数只描述原始读数，不作为稳定的命中率提升估计。**
其 baseline 三次为 64.58%、65.56%、65.81%；compact 为 77.47%、74.77%、70.74%。
77.47% 那份报告的 hit+miss 比 total 高 5.33%，触发 QA；compact 的波动达 6.73 pp。

Prefill baseline 三次为 **82.76%、112.97%、89.08%**；compact 为
**92.71%、88.86%、91.15%**。baseline 3/3、compact 2/3 触发 QA，包含一次超过
100% 的读数。所有原值与计数保留，不用其估算有效提升。
全部 18 份主 memory 报告共有 6 份触发 QA；Sq=1 的 6 份均通过。

额外对 Sq=4/128 以四项 sector-only 指标做复测，见
[复测原始表](sector_only/results.md)、[记录](sector_only/results.json)、
[核验](sector_only/verification.json)。缩小指标集合后仍然是 4 passes，
因此不能假设移除 DRAM 指标就能消除 replay 不一致。该组单独报告，不与主采集
混合取中位数，也不选择性丢弃异常样本。

**额外 12 份报告已完成并核验，5 份触发 QA。** Sq=4 baseline 三次为
64.05%、65.57%、65.90%，compact 为 78.73%（QA 异常）、77.58%、71.77%。
Prefill baseline 三次为 **103.08%、103.19%、105.92%**，均越界；compact 为
92.67%、85.40%（QA 异常）、93.93%。因此复测没有恢复可信的 prefill 命中率比较。
全轮共 48 份报告；所有 target 的输出/LSE 正确性检查通过，被测源码哈希未改变。

主采集资源采样：可用主存最低 191.61 GiB、聚合 RSS 最高 2.94 GiB、
临时盘剩余最低 51.97 GiB、区间 CPU busy 最高 13.78%。无资源压力。
采样极值不代表连续监控的绝对峰值。

## 实验对象与边界

- 被测实现来自 `88c39df4c75ced166797d681e0ae1853a32a4152`。
  运行前核对上一轮 H200 `source_manifest.json` 的全部 24 个文件，运行后再次检查
  本次 manifest 中的源文件哈希；没有修改 kernel 或原 benchmark。
- Baseline：原调度 + 普通 KV 的 SM90 attention-only specialization。
  实验组 `compact`：owner-local 静态调度 + cudaMalloc compact KV，关闭审计。
  对应上一轮矩阵的 A/C。`schedule_only` 仅在初始化时创建并校验，以保留上一轮
  分配顺序，不属于本次 NCU 采集组。
- 仅 B=64、Sk=32768；decode Sq=1、4；prefill Sq=128 dense/noncausal。
  BF16、H=128、CKV512/KPE64、page=64，正态随机输入、seed=42。
  decode scale=0.1352337788608801；prefill scale=1/sqrt(512)。
- 使用原 SM90 planner，相同 shape 的 baseline/compact split plan 一致，
  不强制 split_kv=1。完整 plan summary、owner/task 审计、SM partition/rank、
  arena hash base/mask 均逐次保留。每个 target 实测并要求 66/66 SM。
- 两组使用同一初始化代码和逻辑随机输入。每个 target 都先校验 baseline、
  schedule-only、compact 的完整输出/LSE 与原 fused kernel 逐位一致，
  审计及无审计版本均检查。NCU 结束后单独执行 merge，再检查被测输出/LSE。
- 3 次预热，每次 attention 前清理 256 MiB benchmark cache。
  `cudaProfilerStart/Stop` 之间仅一次 attention launch；不含 merge、GPU 审计
  清零、初始化、runtime 探测、scatter、L2 eviction。内部 partial 写回属于被测 kernel。

## 计数器口径

- LTC 单独请求 `lts__t_requests_srcunit_ltcfabric.sum`，单位 **requests**，与此前
  B200 实验一致。请求数不换算为 bytes。
- L2/memory 独立进程采集，复用 B200 的十指标集合。L2 hit rate 使用
  **sector** 指标 `lts__t_sector_hit_rate.pct`，同时保存 `lts__t_sectors.sum`、
  `lts__t_sectors_lookup_hit.sum`、`lts__t_sectors_lookup_miss.sum`。
  其余六项为 duration、DRAM read/write bytes、read/write bandwidth、DRAM
  peak-sustained throughput，精确列表见 `data/results.json`。
- 每个 Sq、指标组和模式各 3 次独立采集，共 36 份 NCU 报告。
  顺序 baseline/compact、compact/baseline、baseline/compact，GPU 工作全串行。
- NCU 2026.1.1：`--profile-from-start off --launch-count 1 --cache-control all
  --clock-control boost --replay-mode kernel`。
  [NVIDIA CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
  说明 `cache-control all` 在每次 replay 前清空 GPU cache。本次为冷 L2 口径。
- 保留原始 sector 百分比，不截断、不用 hit/(hit+miss) 修复。
  超出 [0,100]% 或 `abs((hit+miss)/total-1)>5%` 标记 QA 异常；
  5% 沿用仓库已有阈值，不代表 NVIDIA 精度规范。
- LTC 减少率为 `100*(1-median(compact)/median(baseline))`。
  replay pass 不是独立样本；NCU duration 不作为无 profiler 的性能测量。
  这些是 attention kernel 指标，不是端到端 MLA（含 merge）或服务指标。

## 复现

从仓库根目录运行，先检查机器资源及 GPU 是否空闲。输出目录必须不存在。
使用本机已有 `.venv` 和 NCU，不使用 G-Watch。

```bash
nproc
free -h
uptime
df -h /tmp /workspace
nvidia-smi
.venv/bin/python -u reports/h200_mla_b64_sk32768_ncu_20260907/run.py \
  --output-root reports/h200_mla_b64_sk32768_ncu_reproduction --trials 3
.venv/bin/python reports/h200_mla_b64_sk32768_ncu_20260907/summarize.py \
  --output-root reports/h200_mla_b64_sk32768_ncu_reproduction
```

runner 固定 `MAX_JOBS=4`、`FLASHINFER_NVCC_THREADS=1`、`OMP_NUM_THREADS=4`、
`FLASHINFER_CUDA_ARCH_LIST=9.0a`。本机 16 逻辑 CPU，运行前主存约 191 GiB 可用、
临时盘约 53 GiB 可用。每份报告结束及子进程运行超过 15 秒时记录 CPU jiffies、
聚合 RSS、可用内存、负载、临时盘和 GPU 信息。

仅重新汇总本次结果（不运行 GPU）：

```bash
.venv/bin/python reports/h200_mla_b64_sk32768_ncu_20260907/summarize.py \
  --output-root reports/h200_mla_b64_sk32768_ncu_20260907/data
```

复现额外 sector-only 复测及其核验：

```bash
.venv/bin/python -u reports/h200_mla_b64_sk32768_ncu_20260907/refine_l2.py \
  --source reports/h200_mla_b64_sk32768_ncu_20260907/data \
  --output-root reports/h200_mla_b64_sk32768_ncu_sector_reproduction \
  --sq 4 128 --trials 3
.venv/bin/python reports/h200_mla_b64_sk32768_ncu_20260907/summarize.py \
  --output-root reports/h200_mla_b64_sk32768_ncu_20260907/data \
  --sector-only-root reports/h200_mla_b64_sk32768_ncu_sector_reproduction
```

每次采集的 `*.command.json` 记录 argv、cwd 和环境覆盖；`.ncu-rep`、原始 CSV、
日志均保留在本地。二进制报告和日志遵循仓库已有 Git ignore 规则。
离线导出示例：

```bash
ncu --import reports/h200_mla_b64_sk32768_ncu_20260907/data/decode_sq1/ltc/baseline_01.ncu-rep \
  --page raw --csv --print-units base
```

`summarize.py` 核对 36 份报告的唯一组合、单 kernel CSV、requested metrics、
replay passes、命令、target 元数据、前后正确性记录、相同 shape 的计划一致性、
L2 QA 以及源码哈希，并保存原始文件 SHA-256。
