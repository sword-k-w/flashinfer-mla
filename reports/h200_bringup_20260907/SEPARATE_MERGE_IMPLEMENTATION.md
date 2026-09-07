# H200 SM90 MLA：attention / merge 分离实现

2026-09-07，已在当前 NVIDIA H200 上完成实现、正确性测试、launch trace 和基础计时。

## 实现结果

- 原版 `BatchMLAPagedAttentionWrapper(backend="fa3")` 保持 fused cooperative launch。
- 新实验版本将 attention 尾部的 `grid.sync()` 和 merge 编译移除；主计算、
  persistent worklist、split-KV planner、partial workspace 格式及 BF16/FP16
  `cp.async` 加载路径保持原实现。
- merge 是单独的 kernel，直接复用 `DevicePersistentMergeStates`。两个 kernel
  在同一 stream 顺序启动，以 kernel 边界完成同步。沿用原 grid/线程布局。
- `attention()` 只启动主 kernel；`merge()` 只启动合并；`run()` 顺序启动两者。
  split 行在 merge 完成前只有 workspace 中的 partial 结果，最终输出尚不可读。
  非 split 行由 attention 直接写最终输出，merge 不覆盖这些行。
- 无 split 的计划仍启动一个空 merge，方便保持一致的实验入口；只测主计算时调用
  `attention()` 即可。
- 本轮完成 baseline 拆分。cudaMalloc owner-split compact 和 partition-aware
  scheduler 尚未移植到此 SM90 kernel。

代码位置：

- [CUDA kernel 与 launcher](../../include/flashinfer/attention/mla_hopper.cuh)
- [FFI dispatch](../../csrc/batch_mla_sm90_run.cu)、[binding](../../csrc/batch_mla_sm90_binding.cu)
- [独立 JIT URI 与编译选项](../../flashinfer/jit/attention/modules.py)
- [实验入口](../../benchmarks/sm90_mla_separate_merge.py)
- [GPU 回归测试](../../tests/attention/test_sm90_mla_separate_merge.py)
- [接口与数据生命周期说明](../../docs/design_docs/batch_mla_backend_architecture.md#experimental-sm90-separate-merge-baseline)

实验 JIT 使用 `separate_merge=True` 和独立 `_separate_merge` URI；其底层 `run`
额外接收 phase：0=完整、1=attention、2=merge。普通 JIT 的函数参数保持不变。
实验 Python runner 限定 SM90、BF16/FP16、contiguous split Q/KV、512/64 维度。
自定义 kernel 内 profiler 暂未启用；CUDA activity trace 正常可用。

## 验证

[pytest 日志](separate_merge_tests.log)：**11 passed**。

- BF16 / FP16，causal / noncausal，LSE base2 / base-e。
- 全 split、全直接输出、同一计划内 split 与直接输出混合。
- Sq=1/4/128，非整页 KV 尾部、随机重排页表、不同请求长度。
- 对独立 FP32 attention 参考检查 output / LSE；output 容差 `rtol=1e-2, atol=1e-2`，
  LSE 容差 `rtol=1e-3, atol=1e-3`。
- 拆分版与 fused 版 output / LSE 逐位一致。
- 用 NaN 污染 workspace / output，验证主计算写 partial、merge 补全最终输出，
  排除未写入与读旧数据造成的假通过。
- CUDA Graph 同时覆盖完整调用与分别调用两阶段，以及有/无 LSE；每次修改输入再重放。
- 额外 B=64、Sk=32768 与 B=2、Sk=1048576 的输出和 LSE 也与 fused 逐位一致。

[CUDA trace 摘要](separate_merge_launches.json) 验证实际 launch：

| 调用 | Kernel 数量 | CUDA launch API |
| --- | ---: | --- |
| 原版 fused | 1 | cudaLaunchCooperativeKernel |
| attention-only | 1 | cudaLaunchKernel |
| merge-only | 1 | cudaLaunchKernel |
| 完整拆分版 | 2 | cudaLaunchKernel × 2 |

## 基础计时

H200，BF16，H=128，CKV/KPE=512/64，page=64，noncausal。
CUDA Graph 每次包含 20 次调用，以 CUDA events 计时，10 次 replay 的中位数。
输入重复使用、未锁频；attention / merge / full 独立测量，缓存状态不同，不能要求
三者时间严格相加。这是 baseline 计时，不是 partition-aware 加速结果。

单位 µs：

| B | Sq | Sk | Split | 原版 fused | Attention only | Merge only | 完整拆分版 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2 | 1 | 1024 | 是 | 14.87 | 9.19 | 5.36 | 14.86 |
| 64 | 1 | 32768 | 是 | 1166.49 | 1164.15 | 12.41 | 1175.04 |
| 64 | 4 | 32768 | 否 | 4113.65 | 4129.10 | 1.25 | 4149.28 |
| 2 | 128 | 1024 | 否 | 148.80 | 144.71 | 1.17 | 146.14 |
| 2 | 1 | 1048576 | 是 | 1095.39 | 1090.04 | 9.78 | 1101.84 |

原始结果：[常规尺寸](separate_merge_benchmark.json)、[长上下文](separate_merge_long_context.json)。
小 workload 中移出 merge 可明显缩短 attention-only 计时；完整 attention 仍需计入
独立 merge。后续比较 partition-aware 时，应同时保留主 kernel 与完整调用的测量结果。

## 复现

在仓库根目录执行；使用当前已经准备好的 Python 3.12 环境：

```bash
export MAX_JOBS=4
export FLASHINFER_NVCC_THREADS=1
export FLASHINFER_CUDA_ARCH_LIST=9.0a

.venv/bin/python -m pytest tests/attention/test_sm90_mla_separate_merge.py -v
.venv/bin/python -m benchmarks.sm90_mla_separate_merge
.venv/bin/python -m benchmarks.sm90_mla_separate_merge --case 2,1,1048576
.venv/bin/python -m reports.h200_bringup_20260907.check_separate_merge_launches
```

独立调用示例（同一 stream、同一 plan / workspace）：

```python
from benchmarks.sm90_mla_separate_merge import make_uniform_case

case = make_uniform_case(2, 1, 1024)  # 构造、分配与 JIT 均在计时前
case.attention()                    # 后续 partition-aware 的主 kernel baseline
out, lse = case.merge()             # 正确性检查前补全最终输出
out, lse = case.run()               # 或一次调用执行完整 attention
```

构建资源限制为 MAX_JOBS=4、NVCC threads=1；机器有 16 logical CPUs。
启动前检查了 CPU/load、可用内存与临时磁盘，构建期间抽样检查进程内存和 CPU；
内存、磁盘余量充足。仅当前实验相关 specialization 被编译。
