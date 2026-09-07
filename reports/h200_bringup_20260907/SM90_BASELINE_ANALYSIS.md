# 用 FlashInfer SM90 MLA 建立 H200 partition-aware 实验

结论：可行。保留本仓库未修改的 `backend="fa3"` 作为 baseline，在同一个
SM90 MLA kernel 上增加可选 compact addressing / partition-aware scheduling，
能够形成有意义的对照。直接与 `/workspace/vllm-fa` 的 compact FA3 对比只能
得到两个实现的性能差异，不能归因于 partition awareness。

## 已确认的事实

- BF16、H=128、latent/RoPE=512/64、page=64 的 absorbed MLA 已通过此前
  Sq=1/4/128 的小规模 output/LSE reference 检查。
- `include/flashinfer/attention/mla_hopper.cuh` 使用 Hopper WGMMA。
  BF16 KV 路径经 `prefetch_offset()`、`load_kv()`、
  `smem_t::load_128b_async()` 发出 cp.async；不是 TMA descriptor 加载。
- 每个 CTA 256 threads、Q/KV tile=64/64、两级 pipeline。
- `MLAPlan()` 在 CPU 上建立分块任务，以 min-heap 分配到逻辑工作组；
  kernel 按 `work_indptr[blockIdx.y]` 遍历静态任务。没有 GPU 动态抢任务队列。
- H=128 的这些 shape 使用二维 grid `(2,66)`。这里的两个 CTA 是逻辑 Q
  分块，launcher 使用 `cudaLaunchCooperativeKernel`，没有指定硬件 thread
  block cluster。不能假定同一 `blockIdx.y` 的两块属于同一物理 partition。
- 所有 CTA 完成 attention 阶段后经过 `grid.sync()`，在同一 kernel 内执行
  `DevicePersistentMergeStates()`。即使某个 partition 没有 attention 工作，
  也不能让它的 CTA 在全局 barrier 前提前退出。

## 实际 planner 探测

本轮调用真实 `plan()` 并读取工作表，未启动 attention 或进行计时。
原始结果保存在 `sm90_plan_probe.json`。以下均为 H=128、page=64、noncausal。

| B | Sq | Sk | 工作项数 | 活跃逻辑组 / 66 | KV chunk 长度 | 使用 partial output |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | 1 | 512 | 16 | 16 | 64 | 是 |
| 2 | 1 | 1024 | 16 | 16 | 128 | 是 |
| 64 | 1 | 32768 | 128 | 66 | 32000、768 | 是 |
| 64 | 4 | 32768 | 256 | 66 | 32768 | 否 |
| 64 | 128 | 32768 | 8192 | 66 | 32768 | 否 |
| 2 | 1 | 1048576 | 66 | 66 | 32000、24576 | 是 |

这说明它不是固定 `num_splits=1` 的外部 FA3 路径。Split 数量和尾块长度随
shape 变化；B=64、Sq=1、Sk=32768 的尾块明显不均匀，划分 owner 时不能仅
按工作项数量平均切分。以上是调度元数据，不是实测 SM 利用率或瓶颈结论。

探测的 split 起点均按 64-token page 对齐。`MLAPlan()` 的通用代码也存在
32-token chunk 档位，因此不能把这个性质泛化到任意更短序列或其他形状。

## 推荐的集成边界

1. 复用外部 runtime 的 cudaMalloc arena、hash recovery/validation、SM
   partition/rank probe 和 4 KiB remap 算法。为本仓库编写适配层，不能把其
   compact tensor 直接作为普通线性 `ckv/kpe` 传入现有 kernel。
2. 在 BF16 `prefetch_offset/load_kv` 地址路径增加可选 remap。`ckv` 对应
   外部 MLA 的 512-dim V/latent，`kpe` 对应 64-dim K/RoPE。
   两者 stride、compact span、owner offset 和 scatter 必须分别一致。
   保持 QK/PV WGMMA、softmax、shared-memory swizzle 与 pipeline 不变。
3. 保留原 planner 的数学工作分块、partial-output 索引和 merge 元数据，
   对工作项增加 owner 并在 owner 内重新分配。起步可用 batch-owner；
   更接近原实验的方案是 `(batch, KV range)` owner，可细分 decode 的长 KV。
4. 使用真实 `%smid` 和 owner-local rank 选择逻辑任务槽及 Q 半块。
   修改 attention 阶段所有依赖 `blockIdx.x/y` 的任务读取和写回索引；
   merge 阶段可继续按原始全局 CTA 编号唯一分工，避免重复或漏算。
   必须验证驻留 CTA 数和 owner-local 槽位的完整覆盖。
5. Sq>1 时不同 Q tiles 会复用同一 KV range，owner 必须以数据范围为准，
   不能独立对每个 `(Q tile, KV split)` 随意指定不同 owner。
   首轮限定固定长度、连续页表、noncausal，便于保持无复制的一致 ownership。

基于这些代码结构，改动可以集中在 runtime/FFI 参数、CPU planner、任务索引
和 KV 地址路径，不需要重写 Hopper attention 数学主体。调度和归并的一致性
是主要正确性风险；地址计算开销、owner 负载不均衡和归并通信可能抵消 locality
收益，目前没有性能数据可以断言会加速。

## 对照与测量要求

建议保留三条可独立选择的路径：

- A：原 SM90 MLA、普通 KV allocation、原 planner。
- B：相同 SM90 MLA、owner 调度、普通 KV allocation。
- C：与 B 相同工作分块和调度、cudaMalloc compact placement/remap。

A→C 是总效果，A→B 展示调度及相关索引改动的净影响，B→C 展示 compact
放置与地址 remap 的净影响。不能将 B→C 称为完全不含额外开销的纯 allocator
收益。三者使用相同逻辑输入、scale、mask、输出/LSE 设置及 split 分块。

计时覆盖完整 cooperative kernel，包括 grid barrier 与 merge；plan、scatter、
初始化放在 kernel-only 计时之外，并单独记录其成本。再用 cold-L2 的成对计时
以及 LTC 请求、HBM 字节数等指标验证收益来源。当前小规模正确性通过不能证明
该 baseline 在全矩阵上性能良好，也不能代替长序列和大 batch 的验证。

外部 H200 runtime 当前固定使用 64 GiB arena。迁移后需要重新按两边 owner
占用、4 KiB 对齐和实际 mapped span 计算容量，不能照搬 B200/B300 的
128 GiB arena 容量轴。配对测试还需预留普通 baseline KV、workspace 和验证
缓冲的显存。

这应记录为“FlashInfer SM90 MLA 的 H200 partition-aware 实验”，与原
Blackwell CuTe DSL 结果保持区别。本轮只做源码分析和 planner 探测，没有
修改 kernel 或实现 compact 集成。
