# FlashInfer CuTe DSL MLA localized partition-aware experiment

## Result

The split-granular localized implementation passes correctness testing and
removes nearly all sequence-length-dependent LTC fabric traffic. On the full
78-cell decode matrix, localized MLA has a 1.0183x geometric-mean speedup and
wins 66 of 78 cells. The benefit becomes consistent once either the batch or
sequence is large enough to amortize the experimental-path setup:

- `Sk >= 16,384`: 48/48 wins, 1.0362x geometric mean.
- `Sk >= 262,144`: 1.0423x geometric mean.
- `B >= 16`: 39/39 wins, 1.0367x geometric mean.
- Full matrix: 66/78 wins, 1.0183x geometric mean.

This is a kernel experiment, not an end-to-end inference result. It measures a
fixed page table and excludes cache construction, localized allocation, and KV
population from timed regions.

Per request, the experiment has no third partition-aware cudaMalloc control.
The measured timing and NCU comparisons are localized versus the unmodified
modular MLA path, so they do not numerically separate scheduler effects from
physical-placement effects.

## Workload and environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA B300 SXM6 AC, 148 SMs, 275,040 MiB HBM |
| Software | Python 3.12.3, PyTorch 2.13.0+cu130, CUDA runtime 13.0 |
| Decode shape | `Sq=1`, `H=128`, BF16, latent/rope dimensions `512/64` |
| Paged KV | page size 64, fixed identity-contiguous logical page table |
| Batch axis | 2, 4, 8, 16, 32, 64 |
| `seqlen_k` axis | 512 through 1,754,432 (13 values) |
| Capacity endpoint | one layout 120.47 GiB; paired layouts 240.93 GiB at B=64 |
| Standard | original modular MLA, static persistent scheduler, cudaMalloc KV |
| Localized | split-granular partition-aware static scheduler, two RM-localized KV pools |

The repository is installed in the independent
`/workspace/flashinfer-mla/.venv`. Its editable-build hook deliberately
upgrades `nvidia-nccl-cu13` to 2.31.2 with `--no-deps` for the B200 EP runtime,
while this PyTorch wheel declares an exact 2.29.7 dependency. Consequently
`pip check` reports that known metadata mismatch. It does not affect these
single-GPU MLA kernels, and no package in `/workspace/vllm-fa/.venv` was
changed.

Timing uses four balanced paired blocks in the order AB, BA, BA, AB. Each
block uses `triton.testing.do_bench`, which clears L2 before every sample. The
reported latency is the median of per-mode block medians; the full per-block
samples and metadata are retained in `iter_000_evaluation/post_flops.json`.

## Matrix summary

Geometric-mean speedup by batch:

| Batch | Geomean | Range | Wins |
| ---: | ---: | ---: | ---: |
| 2 | 0.9892x | 0.8972x-1.0640x | 8/13 |
| 4 | 0.9931x | 0.9002x-1.0577x | 9/13 |
| 8 | 1.0185x | 0.9855x-1.0555x | 10/13 |
| 16 | 1.0304x | 1.0019x-1.0587x | 13/13 |
| 32 | 1.0379x | 1.0019x-1.0611x | 13/13 |
| 64 | 1.0418x | 1.0068x-1.0848x | 13/13 |

Geometric-mean speedup by sequence length:

| `seqlen_k` | Geomean | Wins |
| ---: | ---: | ---: |
| 512 | 0.9817x | 3/6 |
| 1,024 | 0.9757x | 3/6 |
| 2,048 | 0.9869x | 3/6 |
| 4,096 | 0.9966x | 4/6 |
| 8,192 | 1.0108x | 5/6 |
| 16,384 | 1.0281x | 6/6 |
| 32,768 | 1.0341x | 6/6 |
| 65,536 | 1.0267x | 6/6 |
| 131,072 | 1.0314x | 6/6 |
| 262,144 | 1.0369x | 6/6 |
| 524,288 | 1.0427x | 6/6 |
| 1,048,576 | 1.0448x | 6/6 |
| 1,754,432 | 1.0450x | 6/6 |

The worst point is B=2, Sk=512 at 0.8972x. At that scale, both kernels are
only about 16-18 microseconds, so the extra SM-topology loads, partition-aware
scheduler arithmetic, and localized descriptor selection dominate. The
largest observed speedup is B=64, Sk=512 at 1.0848x, but short-kernel points
are more susceptible to small absolute timing shifts. The long-sequence trend
is the more useful signal: it settles around 1.03x-1.05x.

The capacity endpoint B=64, Sk=1,754,432 completes successfully:

| Mode | Time |
| --- | ---: |
| Standard | 21.195296 ms |
| Localized | 20.169456 ms |
| Speedup | 1.05086x |

## NCU LTC fabric traffic

Nsight Compute 2026.1.1 collected
`lts__t_requests_srcunit_ltcfabric.sum` for one isolated, profiled decode
launch per mode and shape at B=64. G-Watch was not used.

| `seqlen_k` | Standard requests | Localized requests | Reduction | Standard/localized |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 641,973 | 310,874 | 51.58% | 2.07x |
| 65,536 | 37,808,727 | 354,298 | 99.06% | 106.71x |
| 524,288 | 302,418,177 | 369,120 | 99.88% | 819.30x |
| 1,754,432 | 1,011,324,942 | 408,449 | 99.9596% | 2,476.01x |

Localized request counts stay close to a fixed 0.31-0.41 million while the
standard path grows approximately with KV length. This is direct evidence that
the owned KV pages are being served from their matching physical partition.
Latency does not improve by the same ratio because LTC fabric requests are
only one part of the kernel: local HBM/L2 traffic, TMA setup, softmax, MMA,
split-KV reduction, and scheduler overhead remain.

## Figures and raw artifacts

- `iter_000_evaluation/figures/performance_comparison.png`: complete latency
  heatmap, with standard/localized milliseconds in every cell.
- `iter_000_evaluation/figures/ltc_fabric_requests.png`: absolute NCU request
  counts.
- `iter_000_evaluation/figures/ltc_fabric_reduction.png`: reduction versus
  sequence length.
- `iter_000_evaluation/post_flops.json`: all timing samples and allocation /
  scheduler metadata.
- `iter_000_evaluation/post_ncu_ltc.json`: parsed NCU results, commands, and
  paths to all `.ncu-rep`, raw CSV, and log files.

## Correctness and regression checks

- `tests/attention/test_cute_dsl_mla_localized.py`: 14/14 passed for B in
  `[2, 3, 4, 8, 16, 32, 64]` and Sk in `[128, 512]`, comparing localized output
  against the original modular MLA.
- An additional B=4, Sk=65,536 split-granular check (`split_kv=18`, owner work
  37/35) produced bit-exact output and LSE versus the original modular kernel.
- One original modular BF16 case and one original modular FP8 case passed after
  the shared call signature was extended.
- Ruff formatting and lint checks passed for all changed Python files.

## Reproduction

```bash
MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2 \
  .venv/bin/python benchmarks/bench_cute_dsl_localized_mla.py

MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2 \
  .venv/bin/python benchmarks/profile_cute_dsl_localized_mla_ltc.py \
  --python /workspace/flashinfer-mla/.venv/bin/python

.venv/bin/python benchmarks/plot_cute_dsl_localized_mla.py
```
