# H200 partition-aware：阶段 1 / 2 实现记录

2026-09-07。已完成本仓库独立 runtime，以及使用普通 KV 内存的分区内静态调度。

## 阶段 1：本地 runtime 与数据 owner

- [Python runtime](../../flashinfer/mla/experimental/partition_runtime.py) 与
  [CUDA/FFI 源码](../../csrc/partition_runtime/runtime.cu) 在本仓库独立构建。
  不导入外部 `partition_kv`，不读取外部 checkout。来源、原始文件 SHA256 和
  BSD-3-Clause 许可证保存在 `csrc/partition_runtime/`。
- 使用真正的 64 GiB cudaMalloc arena，重新恢复并验证本次分配的 hash base，
  探测 SM affinity。本轮得到 P0/P1=66/66，分区内逻辑 rank 均为 0…65。
- [owner planner](../../flashinfer/mla/experimental/partition_schedule.py) 将
  读取同一 KV 范围的所有 Q 子块归到同一 owner；按预计工作量/SM 数平衡两个分区。
  保留原 planner 的 split 边界、partial 索引和 merge 元数据。
- CKV/KPE 使用独立 compact span，物理页有 owner 和 owner-local slot。
  任意页 owner 的 compact scatter/gather 支持 BF16/FP16，均验证逐位一致。
  当前不同 layout 复用同一 arena 起点，应顺序使用。

## 阶段 2：按真实 SM 逻辑编号静态分工

- CTA 读取 `%smid`，查 partition 与 local rank；rank=r 的 SM 处理本分区
  任务列表中的 `r, r+66, r+132, ...`。没有动态抢任务队列。
- Q 子块编号作为任务元数据，计算和 partial 写回均不再将 `blockIdx.x` 当作
  Q 子块编号。persistent、split-KV、cp.async 和独立 merge 保留。
- attention 仍从普通线性 CKV/KPE tensor 加载。本轮 compact 搬运是独立验证，
  未接入 attention 的加载地址；那是阶段 3。
- launcher 检查该 specialization 的 occupancy 为 1 CTA/SM、总 CTA 数为 SM 数。
  运行时另记录 SM visits、task visits 和 task SMIDs，验证每个 SM 一次、每个任务一次、
  实际 owner/local rank 与计划完全一致。occupancy 检查本身不替代执行覆盖验证。
- 当前静态分工要求 GPU 独占执行。`run_checked()` 是带同步检查的完整调用；
  `attention()/merge()/run()` 可用于 graph capture，重放后调用 `validate()`。
  审计计数与重置当前保持开启，尚未作性能优化或报告加速比。

## 验证结果

[pytest 日志](tests.log)：**30 passed**，包含 19 项新增检查与原 separate-merge 的
11 项回归测试。既有依赖产生 3 条 deprecation warnings。

- BF16/FP16 compact roundtrip：均衡、不均衡、全部页归一个 owner；重复改写后回读。
- 新调度与原 fused 输出及 LSE 逐位一致，覆盖 split、无 split、混合 split/direct。
- 混合请求与独立 FP32 reference 比较通过；output 容差 rtol/atol=1e-2，
  LSE 容差 rtol/atol=1e-3。
- 400 次修改输入后的 CUDA Graph 重放，通过完整/显式两阶段、有/无 LSE 四种组合。
  每次验证任务恰好一次以及实际 SM rank，输出不依赖旧 workspace。
- 人为污染审计记录时，验证器能够报告任务遗漏或 SM 重复。

[综合运行记录](results.json) 使用同一个本地 arena，逐项完成调度、CKV/KPE 搬运和
20 次 graph replay。所有形状均为 0 重复、0 遗漏、0 跨 owner 任务：

| B | Sq | Sk | CTA 任务数 | P0/P1 任务数 | 输出/LSE、搬运、覆盖 |
| ---: | ---: | ---: | ---: | --- | --- |
| 2 | 1 | 1024 | 32 | 16/16 | 通过 |
| 64 | 1 | 32768 | 256 | 128/128 | 通过 |
| 64 | 4 | 32768 | 512 | 256/256 | 通过 |
| 2 | 128 | 1024 | 512 | 256/256 | 通过 |
| 2 | 1 | 1048576 | 132 | 66/66 | 通过 |

上述 CTA 任务数包含两个 Q 子块，因此可能是原 planner work item 数的两倍。
完整探测的 SM map/rank、两侧页数和预计负载均在 JSON 中。

## 复现

在仓库根目录执行：

```bash
export MAX_JOBS=4
export FLASHINFER_NVCC_THREADS=1
export FLASHINFER_CUDA_ARCH_LIST=9.0a

.venv/bin/python -m benchmarks.sm90_mla_partition --output stages12.json
.venv/bin/python -m pytest tests/attention/test_sm90_mla_partition.py \
  tests/attention/test_sm90_mla_separate_merge.py -v
```

最小调用：

```python
from flashinfer.mla.experimental.partition_runtime import H200PartitionRuntime
from benchmarks.sm90_mla_separate_merge import make_uniform_case
from benchmarks.sm90_mla_partition import SM90PartitionExperiment

with H200PartitionRuntime() as runtime:
    baseline = make_uniform_case(2, 1, 1024)
    experiment = SM90PartitionExperiment(baseline, runtime)
    out, lse = experiment.run_checked()  # 普通 KV + owner-local static scheduler
```

第一版调度限定 noncausal BF16、H=128、512/64 维度、page=64、identity 页表和
非空、页对齐的 KV 范围；不满足条件时明确拒绝。构建使用 4 jobs × 1 NVCC thread，
开始前检查 CPU/load、可用内存与临时磁盘，并在编译期间抽样检查进程资源。
测试结束后 arena 已释放。原 fused 与 separate-merge baseline 入口保持可用。
