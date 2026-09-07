# H200 MLA baseline 与 cudaMalloc compact 可运行性检查

2026-09-07，在当前 H200 上实测完成。

后续已实现本仓库 SM90 attention/merge 分离，测试和复现入口见
[SM90 separate-merge 实现记录](SEPARATE_MERGE_IMPLEMENTATION.md)。下文记录初次可运行性检查。

| 路径 | 结果 |
| --- | --- |
| 原实验 `standard`：Blackwell modular CuTe DSL MLA | 无法在 H200 编译；实际报错为期望 SM100/SM110 等架构，收到 `sm_90a` |
| 本仓库 `BatchMLAPagedAttentionWrapper(backend="fa3")` | SM90 baseline 跑通；Sq=1、4、128 的小规模正确性检查通过 |
| 原实验 RM localized allocator | 源码明确只支持 B200/B300；本轮没有绕过检查或尝试 RM 分配 |
| `/workspace/vllm-fa` 普通 FA3 absorbed MLA | 补编 MLA specialization 后跑通 |
| `/workspace/vllm-fa` cudaMalloc owner-split compact MLA | allocator、scatter/gather、attention 端到端检查均通过 |

当前仓库没有与原 localized 实验对应的 cudaMalloc owner-split compact 实现。
现成实现位于 `/workspace/vllm-fa/partition_kv/` 和 `hopper/`。本轮直接复用
该实现验证，没有将它移植到 FlashInfer，也没有修改任何 attention kernel。

## 实测环境

- NVIDIA H200，SM90，132 SM；compact runtime 恢复出的分区为 66/66 SM。
- Driver 580.173.02，PyTorch 2.12.1+cu132，NVCC 13.2.78。
- 16 logical CPU；开始时约 194 GiB 可用主机内存、57 GiB 空闲磁盘，GPU 空闲。
- 外部 FA3 构建限制为 `MAX_JOBS=4 NVCC_THREADS=1`，约 4 分 30 秒完成。
  监测中主机可用内存始终超过 180 GiB，磁盘剩余超过 54 GiB。
  最后只剩一个 FA3 编译任务时，FlashInfer JIT 使用两个 worker；未超过总并发预算。
- 补建本仓库 Python 3.12 `.venv`、安装 MLA 所需依赖、初始化 pinned
  CUTLASS/CCCL/spdlog 子模块，并完成 editable 安装。
- 原外部 `hopper/flash_attn_3_cuda*.so` 未覆盖；本轮专用产物在 `fa3_lib/`。
- `source_manifest.json` 记录两个仓库的 revision 和实验入口 SHA-256；
  `flashinfer_environment.txt` 记录 Python 包版本，`resource_snapshot.json`
  保留一次编译期间的资源快照。资源监测并非完整连续采样。

## 正确性结果

共同参数：BF16，128 query heads，latent/RoPE=512/64，page size=64，
DeepSeek-V3 effective softmax scale≈0.135233779。

FlashInfer SM90 backend，B=2、Sk=1024、noncausal：

| Sq | Output 最大绝对误差（FP32 reference） | LSE 最大绝对误差（base2） | 结果 |
| ---: | ---: | ---: | --- |
| 1 | 0.01428080 | 0.00002289 | 通过 |
| 4 | 0.01329732 | 0.00001717 | 通过 |
| 128 | 0.01425648 | 0.00003242 | 通过 |

输出使用 `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`；LSE 使用
`rtol=1e-3, atol=1e-3`。参考直接计算
`softmax((q_latent @ c_latent.T + q_rope @ c_rope.T) * scale) @ c_latent`。
这是独立 SM90 backend 的功能检查，不是原 Blackwell prefill/decode 性能的复现。

外部 FA3 absorbed-MLA，Sq=1、num_splits=1、noncausal：

| B | Sk | Compact K/V roundtrip | 普通 FA3 vs compact | 两者各自 vs FP32 reference |
| ---: | ---: | --- | --- | --- |
| 2 | 1024 | 逐位一致 | 逐位一致，max diff=0 | 通过仓库原有误差界限，max diff=0.015625 |
| 16 | 4096 | 逐位一致 | 逐位一致，max diff=0 | 通过仓库原有误差界限，max diff=0.015625 |

两个 compact case 均恢复出 P0/P1=66/66 SM，使用固定 64 GiB cudaMalloc
arena、capacity-contiguous batch/KV-head ownership，以及
`PartitionStaticPersistentTileScheduler`。普通路径使用
`StaticPersistentTileScheduler`。

首次直接使用外部现有二进制时，两条 MLA 路径均报：
`This flash attention build does not support hdim != hdim_v when hdim <= 64`。
按外部仓库 `mla_baseline/README.md` 补编 BF16 64/256、64/512 specialization
后通过；原始失败日志与成功复测分开保留。普通 FA3 runner 附带的约 32.18 µs
只是 20 次 unlocked-clock smoke timing，不用于速度比较。

## 对后续实验的影响

原实验 kernel 使用 `tcgen05`、TMEM 和 Blackwell 2-CTA MMA。实际失败位置是
`flashinfer/cute_dsl/attention/collective_builder.py:441`。更换 allocator
不足以将该 kernel 移植到 Hopper。

外部 compact 实现也不是单纯把两个 localized pools 换成普通 Tensor：它包含
cudaMalloc arena 的 hash 恢复、SM 分区探测、compact 地址映射、scatter/gather
和配套的 partition-local scheduler。目前本轮验证范围是固定连续页表、
owner 按 `(batch, KV head)` 切分、单 split、noncausal。原实验按
`(batch, split_kv)` 分配工作，两者调度粒度不同。

因此，后续可以直接使用外部仓库普通 FA3 / compact FA3 这一对做 H200 实验。
若要继续使用 FlashInfer 的 SM90 baseline，需要另行实现对应的 compact
地址映射与调度集成。性能比较必须在同一个 kernel family 内配对；本轮没有
完整矩阵、LTC/L2/HBM 指标或加速比结论，也没有验证最大容量点。

## 复现

从 `/workspace/flashinfer-mla` 执行，确保 GPU 空闲。

本仓库原实验的预期架构失败：

```bash
MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python benchmarks/profile_cute_dsl_localized_mla_ltc_target.py \
  --mode standard --batch 2 --seqlen-k 1024 --data-initialization random
```

本仓库 SM90 baseline 功能检查：

```bash
MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1 OMP_NUM_THREADS=2 \
  FLASHINFER_CUDA_ARCH_LIST=9.0a \
  .venv/bin/python reports/h200_bringup_20260907/check_flashinfer_hopper.py
```

外部 FA3 的独立构建命令和全部环境开关见 `fa3_build_command.json`，可重放：

```bash
python - <<'PY'
import json, os, subprocess
from pathlib import Path
p = Path('reports/h200_bringup_20260907/fa3_build_command.json')
c = json.loads(p.read_text())
subprocess.run(c['argv'], cwd=c['cwd'], env={**os.environ, **c['env']}, check=True)
PY
```

独立构建后，运行三个外部检查并保存到新目录：

```bash
python reports/h200_bringup_20260907/run_fa3_checks.py \
  --output-dir reports/h200_bringup_recheck
```

runner 会先加载 `fa3_lib/` 中的扩展，并把实际加载路径写入日志，避免
外部脚本 prepend `hopper/` 时误用旧二进制。

主要记录：`flashinfer_standard.log`、`flashinfer_hopper.json`、
`compact_layout.json`、`verified/*.json` 与 `verified/commands.json`。
两个新增检查脚本通过 Ruff lint/format 和 Python 语法检查。
