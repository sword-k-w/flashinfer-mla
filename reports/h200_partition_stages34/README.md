# H200 partition-aware MLA：阶段 3 / 4

本轮把本仓库独立 cudaMalloc runtime 的 owner-split compact KV 接入 SM90
attention，并运行与先前 B200 尽量对齐的完整矩阵。正式结果见
[RESULTS.md](RESULTS.md)，原始记录见 [matrix.json](matrix.json)。
`matrix.json` 的 `status` 为 `passed` 且包含 216 个点时，才代表正式矩阵完成。

## 实现

- [地址预取](../../include/flashinfer/attention/mla_hopper.cuh) 根据 KV 页的
  owner-local slot 和真实 SM partition，将 CKV/KPE 逻辑地址映射到 cudaMalloc
  arena 中的物理地址。仍使用原来的 `cp.async`、WGMMA 和流水线。
- [共享地址映射](../../include/flashinfer/partition/address.cuh) 同时供 runtime
  scatter/gather 与 attention 使用。来源和 BSD-3-Clause 许可在
  [本地 runtime 目录](../../csrc/partition_runtime/README.md)；运行和编译不依赖
  `/workspace/vllm-fa` 或外部 `partition_kv` 包。
- CKV 的 BF16 行为 1 KiB，KPE 为 128 B，均不会跨越 remap 的 4 KiB 页。
  只改行起始地址计算，行内 `cp.async` 递增保持有效。
- [实验 runner](../../benchmarks/sm90_mla_partition.py) 通过 `compact=True`
  选择 compact 加载，通过 `set_audit(False)` 切换无审计编译版本。
  后者同时移除设备端审计原子操作和 FFI 中三个诊断缓冲区清零操作。
- 原 planner 的 split 边界、partial 输出布局及 merge 元数据继续使用。
  任务按实际 `%smid` 对应 partition 内的逻辑 rank 分配；H200 探测为 66/66 SM。
  rank 为 r 的 SM 依次处理本 partition 的 r、r+66、r+132 等任务。
- 默认公开 FA3 调用仍走原 fused 路径；实验使用独立 JIT specialization。

## 正确性与计时边界

[pytest 日志](tests.log)：**35 passed**，包括先前 separate-merge 和 owner schedule
回归，以及新增 compact attention 检查。3 条 warnings 来自既有依赖弃用提示。

新增测试将普通 CKV/KPE 在 scatter 后写成 NaN，确认 compact attention 仍与保存的
原 fused 输出及 LSE 逐位一致，覆盖 split/direct、audit/no-audit 及 graph replay。
无审计版本另检查诊断缓冲区哨兵未被改写。

[最大边界验证](capacity_validation.json) 已通过
`B=64, Sq=128, Sk=932032`。正式矩阵中，每个配置都先验证 A/B/C 完整输出，
检查两个 partition 的 SM/task 覆盖，再关闭审计进行计时；计时结束后单独执行 merge
并再次比对输出。所有变体使用同一份随机 Q/KV 和相同 split plan。

[调用 trace](attention_traces/summary.json) 对 A/B/C 分别确认一次调用只启动一个
attention kernel，无 merge kernel、GPU memset 或 memcpy。
计时图的每个样本顺序为：

```text
256 MiB L2 eviction → start event → attention kernel → end event
```

事件为 CUDA Graph 中的 external events。分配、hash/SM 探测、owner 规划、scatter、
Python/FFI 提交与检查、L2 eviction、merge、审计操作均不在测量区间。
attention 内部 split partial 写回属于被测 kernel；最终 split reduction 不属于它。
因此这里的时间与加速比是 attention kernel 指标，不能作为端到端推理加速比。

## 固定的矩阵

| 轴/设置 | 值 |
| --- | --- |
| Batch | 2, 4, 8, 16, 32, 64 |
| Sq | 1, 4, 128 |
| Sk（所有行一致） | 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, **932032** |
| 总点数 | **3 × 6 × 12 = 216** |
| dtype / heads / 维度 / 页大小 | BF16 / 128 / CKV512 + KPE64 / 64 |
| 数据 | 正态随机，seed=42，三条路径 KV 完全相同，noncausal |
| Scale | Sq1/4 使用先前 DeepSeek effective scale；Sq128 为 1/sqrt(512) |
| 测量块顺序 | ABC / CBA / CBA / ABC |
| 预算 | paired warmups=20；每路径每块 warmup=500 ms、repeat=1000 ms、至少 20 samples |
| 汇总 | 每块样本中位数，再取 4 块中位数的中位数 |

A 为原始调度 + 普通 KV 的 SM90 attention-only baseline；B 为 owner-local 调度
+ 普通 KV；C 为同一调度 + cudaMalloc compact KV。A/C 是总体效果，A/B 表示
调度变化，B/C 比较固定调度下 compact 加载的净效果。B/C 同时包括地址计算代价和
存储访问变化，不能仅凭时间把其全部归因于内存 locality。

最大长度由整个矩阵的最大 B/Sq 点约束。当前实现需要 64 GiB arena，最大 B=64 时：

```text
floor_to_multiple_of_64(64 GiB / (64 × (512 + 64) × 2 bytes)) = 932032
```

均衡 owner 的 compact 物理 span 接近实际 KV 字节数。除此之外，测量同时保留普通
KV、Q、四份输出/工作区和缓存清理缓冲区，并预留 4 GiB。此次 HBM 估算上限为
951296 tokens，实际限制来自 arena。每行固定相同最大值；不会按 batch 缩短行，
也不会遇到失败就跳过。容量和实际物理 span 在 JSON 中保留。

## 与 B200 的对齐和差异

对齐依据是已保存的 B200 **实际报告**：

- [Sq1 decode](../localized_mla_b200_74_74_20260905/decode/sq1/post_flops.json)
  与 [Sq4 decode](../localized_mla_b200_74_74_20260905/decode/sq4/post_flops.json)。
- [Sq128 prefill](../localized_mla_b200_74_74_separate_20260905/prefill/timing.json)。

B/Sq 轴、dtype、heads、CKV/KPE 维度、页大小、随机数据、scale、采样预算、冷 L2
策略和交错测量保持对应。B200 decode 的末列为 1048576，prefill 为 1008576；
此次受 H200 arena/显存约束，三张矩阵统一使用 932032，其余 11 列相同。

硬件与 kernel 不同：B200 是 74/74 SM 的 Blackwell 路径，此次是 66/66 SM 的
SM90 FA3 MLA。H200 保留原 MLA planner 的 split 方案，没有强制 num_split=1；
最终合并拆到独立 kernel 并排除计时。三条 H200 路径的 split plan 一致。

另一个有意的区别是计时方式：先前 B200 使用 eager Triton `do_bench`；此次 FFI
检查较多，使用 captured CUDA events 排除 CPU 提交空隙，严格计一个 attention
kernel。warmup/repeat 参数采用相同的预算转迭代数思路，估计包括 L2 eviction，
故参数不是“纯 attention 必须累计达到该毫秒数”的含义。最低 20 samples 在大 kernel
上可超过名义预算。小 graph 批量重放，批尾同步，逐样本读取事件时间。

本报告比较同一台 H200 上 A/B/C；不把两个架构的绝对耗时直接解释成 partition-aware
收益。未采集硬件计数器，因此不作 L2/DRAM 流量下降比例或瓶颈成因的测量结论。

## 复现

在仓库根目录执行，先检查 CPU、可用内存、系统负载、临时磁盘和 GPU 余量。
本轮使用 4 jobs × 1 NVCC thread，避免编译占满机器。

```bash
export MAX_JOBS=4
export FLASHINFER_NVCC_THREADS=1
export FLASHINFER_CUDA_ARCH_LIST=9.0a
export OMP_NUM_THREADS=4

.venv/bin/python -m pytest tests/attention/test_sm90_mla_partition.py \
  tests/attention/test_sm90_mla_separate_merge.py -v
.venv/bin/python -u -m benchmarks.bench_sm90_mla_partition \
  --max-seqlen-k 932032 --output reports/h200_partition_stages34/matrix.json
# 绘图需要 matplotlib（只用于 CPU 上的后处理）。
.venv/bin/python -m pip install matplotlib
.venv/bin/python reports/h200_partition_stages34/analyze.py
```

中断后使用相同命令加 `--resume`。runner 检查 axes、timing、configuration 和源文件
哈希后才接受历史点。每次进程运行的 arena hash base、SM partition/rank 图分别保存。
图表脚本只在完整矩阵及哈希验证通过后生成 CSV、汇总 JSON、热图 PNG/PDF 与 RESULTS.md。

实验目前限定 BF16、noncausal、H128、page64、identity 页表、非空页对齐 split，
并需要独占 GPU 执行静态 SM 分工。不同 compact layout 共享 arena 起点，应依次使用；
arena、Q/KV 和元数据必须保留到 GPU 工作结束。这是实验入口，尚非通用生产分配器。

`timing_smoke_v2.json` 是低预算的 8 点 smoke，`capacity_validation.json` 只做最大点
正确性验证，都不是正式性能结果。`timing_smoke.json/log` 保留了初次使用非 external
captured event 导致 invalid argument 的失败记录；修正后才运行 v2 与正式矩阵。
