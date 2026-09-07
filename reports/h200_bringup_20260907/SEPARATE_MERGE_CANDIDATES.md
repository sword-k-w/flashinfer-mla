# H200 MLA：主 kernel 不包含 split 归并的 baseline 候选

2026-09-07 完成官方源码、历史版本和本地实现核对。本轮没有构建或运行
FlashMLA；以下推荐是结构和适配难度判断，不是本机性能结论。

## 结论

最符合已有 kernel 要求的是 DeepSeek FlashMLA 的 SM90 dense decode：
原生支持 split-KV、persistent 工作循环，combine 是独立 kernel。

考虑后续 cudaMalloc owner-split compact，优先验证其旧 cp.async 实现
`b31bfe72a83ea205467b3271a5845440a03ed7cb`（2025-03-01）。它避免了新版本
TMA tensor map 与 4 KiB 非线性 remap 的适配问题。需要明确标注这是历史版本，
不能称为最新 FlashMLA baseline。若必须统一覆盖 Sq=128，该旧版不宜直接使用。

## 候选比较

| 候选 | split-KV | persistent | 独立 combine | compact 适配评价 |
| --- | --- | --- | --- | --- |
| FlashMLA b31bfe7 SM90 dense | 是 | 是 | 是 | cp.async，可在向量加载处 remap；decode 优先候选 |
| FlashMLA 15f13e5 SM90 dense | 是 | 是 | 是 | TMA，需要重新设计加载地址表达；当前实现的独立性能参考 |
| 本仓库 SM90 fa3 | 是 | 是 | 否 | 需先把现有 merge 拆成独立 launch，属于修改后的 baseline |
| 本仓库 `decode_mla_cute_sm80.cuh` | 是 | 当前没有跨工作项的 persistent 循环 | 是 | 同时需要迁移 scheduler；不是优先 Hopper 基线 |
| `/workspace/vllm-fa` 当前 compact FA3 | 强制 effective num_splits=1 | 是 | 当前没有 split 归并可测 | 不满足保留 split-KV 的要求 |
| 本仓库 Blackwell CuTe / CUTLASS MLA | 是 | 有对应路径 | 有对应路径 | 不支持本机 SM90 |

本仓库旧 SM80 decoder 中 `partition_kv` 命名表示 split-KV 中间输出路径，
不能把它当成物理分区感知 allocator/scheduler。

## 旧版 FlashMLA 的直接证据

源码固定链接：
https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h

- 第 91 行：`SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>`。SM80 在这里是
  copy 指令类型名称，attention compute 仍使用 Hopper WGMMA。
- 第 449–480 行：`flash_fwd_splitkv_mla_kernel` 从 scheduler metadata 读取
  begin/end request 和 KV block 范围，CTA 在 for-loop 中连续处理工作。
- 第 572–594 行：`run_flash_splitkv_fwd_mla()` 先 launch attention，再
  launch `flash_fwd_splitkv_mla_combine_kernel`。没有把跨 split 合并嵌入主 kernel，
  也没有新版的 PDL launch。
- 支持 BF16/FP16，Q/K=576、V=512、page=64，适合当前 absorbed-MLA 数据。
  KV 是 packed latent+RoPE，512-dim V 复用 latent 部分。

旧 API 的 `num_sm_parts` 是逻辑调度槽，不是物理 GPU partition：

```text
num_sm_parts = SM_count / Hkv / ceil((Sq * Hq / Hkv) / 64)
```

在 H200（132 SM）、Hq/Hkv=128/1 上，Sq=1/4 分别得到 66/16。
后者主 grid 有 8×16=128 CTA；partition-aware 适配不能简单假定每个 SM
都恰好承载一个 CTA，仍需处理实际驻留和 owner-local 任务完整覆盖。
Sq=128 得到 132/256=0，必须修改调度规划才能使用，不能直接照搬 prefill 矩阵。

该源码检出在 `/tmp/flashmla-cpasync-baseline-review`。依赖子模块尚未初始化，
未安装扩展，CUDA 13.2/PyTorch 2.12 的编译兼容性和当前数据正确性尚待验证。

## 新版 FlashMLA 的差别

本轮最新检出 revision：`15f13e5030374295491c5ce31b02d7e63a7772c6`。
源码检出在 `/tmp/flashmla-baseline-review`。

- `csrc/api/dense_decode.h` 分别调用
  `run_flash_splitkv_mla_kernel()` 和 `run_flash_mla_combine_kernel()`。
- `csrc/sm90/decode/dense/splitkv_mla.cuh` 使用 scheduler metadata 驱动
  persistent request loop，KV 经 TMA 加载。
- `num_sm_parts` 增加了 `max(..., 1)`，不会触发旧版 Sq=128 的零值问题；
  但本轮未验证其大 Sq 正确性、性能或 partition-aware 映射。
- 默认启用 PDL，代码要求主 kernel 后接 combine。要得到隔离的主 kernel
  实验，需要提供受控 launch/profiling 入口并处理 PDL，不能拿整个 Python
  `flash_mla_with_kvcache()` 的延迟当作纯 attention 时间。

官方说明：
https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/docs/20250422-new-kernel-deep-dive.md

对 compact 的适配难点是：现有 cudaMalloc 4 KiB remap 非线性；FlashMLA
TMA 描述的是固定 strides 的 page tensor。只改 descriptor 基地址或逻辑页号
不能普遍表达该 remap，不能把现有 owner-split tensor 原样接入。需要额外
布局/加载设计，这也是优先验证旧 cp.async 版本的原因。

## 实验边界

“主 kernel 不包含合并”是指分离跨 split 的 combine；主 kernel 仍然必须
进行自己的 online-softmax 累积、归一化和 partial-result 写回。统计指标与
主 kernel 计时可以单独圈定 attention launch；正确性验证仍运行 combine。
建议同时保留完整 attention+combine 延迟作为补充，避免把主 kernel 收益
误当成端到端收益。

如果后续要求 decode 和 Sq=128 使用同一套可方便 remap 的 kernel，另一条
路线是把本仓库 SM90 fa3 的 `DevicePersistentMergeStates()` 提取为独立
kernel，同时保持 split plan 和数学主体一致。这个版本需要明确标记为
“FlashInfer SM90 separate-merge variant”，并先验证与原版 output/LSE 一致。
