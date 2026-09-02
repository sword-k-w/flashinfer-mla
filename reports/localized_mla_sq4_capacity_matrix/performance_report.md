# FlashInfer localized partition-aware MLA: Sq=4 results

## Result

The localized experimental path now supports non-causal `Sq=1..4`. The new
`Sq=4` path passed all correctness checks and improved the original modular MLA
kernel in 69 of the 78 measured cells. Across the full matrix, the geometric
mean is **1.0323x** and the median is **1.0421x**.

- Full matrix: 69/78 wins, 1.0323x geometric mean.
- `Sk >= 16,384`: 48/48 wins, 1.0452x geometric mean.
- `Sk >= 65,536`: 36/36 wins, 1.0441x geometric mean.
- `Sk >= 524,288`: 18/18 wins, 1.0469x geometric mean.
- Capacity endpoint, B=64 and Sk=1,754,432: 67.7324 ms to 63.4794 ms,
  or 1.0670x.

The earlier `Sq=1` matrix had a 1.0183x geometric mean and 66/78 wins. This is
not a like-for-like latency comparison because `Sq` changes the workload, but
the direction is consistent with the design: four query tiles amortize the
fixed topology/descriptor overhead and provide enough work to fill more of the
resident clusters.

This remains a kernel-only experiment. It uses a fixed page table and excludes
cache construction, localized allocation, and KV population from the timed
region. As requested, there is no partition-aware cudaMalloc control.

## Workload and measurement

| Item | Value |
| --- | --- |
| GPU | NVIDIA B300 SXM6 AC, 148 SMs, 74 physical 2-SM clusters |
| Partition topology | 38 / 36 physical clusters (76 / 72 SMs) |
| Software | Python 3.12.3, PyTorch 2.13.0+cu130, CUDA runtime 13.0 |
| Shape | non-causal `Sq=4`, `H=128`, BF16, latent/rope `512/64` |
| Paged KV | page size 64, fixed identity-contiguous logical page table |
| Batch axis | 2, 4, 8, 16, 32, 64 |
| `seqlen_k` axis | 512 through 1,754,432 (13 values) |
| Capacity endpoint | one KV layout 120.47 GiB; paired layouts 240.93 GiB |
| Standard | original modular MLA static persistent scheduler, cudaMalloc KV |
| Localized | owner-local static persistent scheduler, two RM-localized KV pools |

Timing uses four balanced paired blocks in AB, BA, BA, AB order. Each block
uses `triton.testing.do_bench`, which clears L2 before every sample. The
reported latency is the median of per-mode block medians; every block and
sample count is retained in `iter_000_evaluation/post_flops.json`.

The first long-running matrix process encountered transient RM/HBM
fragmentation while allocating the paired 240.93-GiB endpoint after completing
77 cells. The benchmark's incremental result file preserved those cells, and
the final cell completed in a fresh process via `--resume`. Only kernel calls
are timed, so this allocation event is outside the reported latency.

## Scheduler geometry

Every `(batch, splitKV)` owner work unit expands to four independently
scheduled query tiles. All four tiles remain with the same owner because they
reuse the same KV interval. On this B300, the measured geometry was:

| B | splitKV | P0/P1 work units | P0/P1 query tiles | Standard active clusters | Localized active clusters |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 9 | 9 / 9 | 36 / 36 | 72 | 72 |
| 4 | 4 | 8 / 8 | 32 / 32 | 64 | 64 |
| 8 | 2 | 8 / 8 | 32 / 32 | 64 | 64 |
| 16 | 1 | 8 / 8 | 32 / 32 | 64 | 64 |
| 32 | 1 | 16 / 16 | 64 / 64 | 74 | 74 |
| 64 | 1 | 33 / 31 | 132 / 124 | 74 | 74 |

Thus, for all six batches the localized path activates exactly as many
clusters as the standard path can activate. There is no localized-specific
first-wave underfill in this matrix.

## Timing matrix summary

Geometric-mean speedup by batch:

| Batch | Geomean | Range | Wins |
| ---: | ---: | ---: | ---: |
| 2 | 1.0043x | 0.9195x-1.0740x | 8/13 |
| 4 | 1.0275x | 0.9330x-1.0920x | 11/13 |
| 8 | 1.0414x | 1.0000x-1.0877x | 12/13 |
| 16 | 1.0403x | 0.9660x-1.0802x | 12/13 |
| 32 | 1.0373x | 1.0124x-1.0654x | 13/13 |
| 64 | 1.0437x | 1.0298x-1.0670x | 13/13 |

Geometric-mean speedup by sequence length:

| `seqlen_k` | Geomean | Wins |
| ---: | ---: | ---: |
| 512 | 0.9838x | 3/6 |
| 1,024 | 1.0023x | 5/6 |
| 2,048 | 1.0143x | 4/6 |
| 4,096 | 1.0294x | 4/6 |
| 8,192 | 1.0311x | 5/6 |
| 16,384 | 1.0508x | 6/6 |
| 32,768 | 1.0463x | 6/6 |
| 65,536 | 1.0391x | 6/6 |
| 131,072 | 1.0405x | 6/6 |
| 262,144 | 1.0444x | 6/6 |
| 524,288 | 1.0445x | 6/6 |
| 1,048,576 | 1.0476x | 6/6 |
| 1,754,432 | 1.0486x | 6/6 |

The worst cell is B=2, Sk=2,048 at 0.9195x (20.656 versus 22.464 us).
At this scale, roughly 1.8 us of fixed partition-aware setup dominates. The
largest observed speedup is B=4, Sk=16,384 at 1.0920x. The more robust signal
is the stable 1.04x-1.05x improvement throughout the long-sequence region.

## NCU LTC fabric traffic

Nsight Compute 2026.1.1 measured
`lts__t_requests_srcunit_ltcfabric.sum` for one isolated decode launch per mode
and shape at B=64. G-Watch was not used.

| `seqlen_k` | Standard requests | Localized requests | Reduction | Standard/localized |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,980,198 | 1,269,780 | 35.876% | 1.559x |
| 65,536 | 63,591,071 | 1,366,860 | 97.851% | 46.523x |
| 524,288 | 540,270,065 | 1,381,776 | 99.744% | 390.997x |
| 1,754,432 | 1,929,401,421 | 1,421,692 | 99.9263% | 1,357.116x |

The standard path's remote requests grow with KV length. The localized path
stays near 1.3-1.4 million, so the sequence-dependent remote KV component is
almost eliminated. The residual fixed traffic is more visible at Sk=1,024,
which explains why the percentage reduction there is much smaller.

## Correctness and artifacts

- 56/56 parameterized tests passed for `Sq=1..4`, B in
  `[2, 3, 4, 8, 16, 32, 64]`, and Sk in `[128, 512]` against the original
  modular MLA path.
- An additional `Sq=4`, Sk=512 sweep over all six benchmark batches produced
  bit-exact output and LSE for every splitKV geometry.
- Ruff formatting and lint checks passed for all changed Python files.

Artifacts:

- `iter_000_evaluation/figures/performance_comparison.png`
- `iter_000_evaluation/figures/ltc_fabric_requests.png`
- `iter_000_evaluation/figures/ltc_fabric_reduction.png`
- `iter_000_evaluation/post_flops.json`
- `iter_000_evaluation/post_ncu_ltc.json`
- `iter_000_evaluation/ncu_ltc/`: `.ncu-rep`, raw CSV, and logs

## Reproduction

```bash
MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2 \
  .venv/bin/python benchmarks/bench_cute_dsl_localized_mla.py --seqlen-q 4

MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2 \
  .venv/bin/python benchmarks/profile_cute_dsl_localized_mla_ltc.py \
  --seqlen-q 4 --python /workspace/flashinfer-mla/.venv/bin/python

.venv/bin/python benchmarks/plot_cute_dsl_localized_mla.py \
  --timing reports/localized_mla_sq4_capacity_matrix/iter_000_evaluation/post_flops.json \
  --ncu reports/localized_mla_sq4_capacity_matrix/iter_000_evaluation/post_ncu_ltc.json \
  --output-dir reports/localized_mla_sq4_capacity_matrix/iter_000_evaluation/figures
```
