# FlashInfer localized MLA on B300: B200-aligned random-data matrix

## Result

The B300 rerun uses the same workload axes, random-data initialization, and
timing configuration as the retained B200 matrix. Both accepted matrices
completed all 72 requested shapes. Query and KV tensors contain deterministic
random BF16 values (`torch.manual_seed(42)` followed by `torch.randn`), and the
standard KV values are scattered into the localized pools so that both layouts
read identical data.

The retained results are:

| Workload region | Wins | Geomean | Median | Range |
| --- | ---: | ---: | ---: | ---: |
| `Sq=1`, all cells | 62/72 | 1.0347x | 1.0391x | 0.9273x-1.0876x |
| `Sq=1`, `Sk >= 16,384` | 40/42 | 1.0567x | 1.0668x | 0.9732x-1.0876x |
| `Sq=1`, `Sk >= 65,536` | 30/30 | 1.0660x | 1.0710x | 1.0027x-1.0876x |
| `Sq=1`, `Sk >= 524,288` | 12/12 | 1.0760x | 1.0775x | 1.0604x-1.0876x |
| `Sq=4`, all cells | 56/72 | 0.9970x | 1.0366x | 0.7233x-1.0968x |
| `Sq=4`, `B >= 4` | 55/60 | 1.0350x | 1.0423x | 0.9382x-1.0968x |
| `Sq=4`, `B >= 4, Sk >= 16,384` | 35/35 | 1.0487x | 1.0462x | 1.0260x-1.0968x |
| `Sq=4`, `B = 2` | 1/12 | 0.8268x | 0.7768x | 0.7233x-1.0072x |

The capacity endpoints are:

- `Sq=1, B=64, Sk=1,048,576`: 17.6901 ms standard versus 16.3547 ms
  localized, or 1.0817x.
- `Sq=4, B=64, Sk=1,048,576`: 67.5595 ms standard versus 61.6893 ms
  localized, or 1.0952x.

For `Sq=4, B=2`, both owners have 36 query tiles but the measured B300
partitions have 35 and 39 resident clusters. One owner therefore takes two
waves while the other takes one. This stable owner-wave mismatch explains the
long-sequence regression. All 35 long-sequence cells with `B >= 4` win.

## Comparison with the earlier B300 and B200 matrices

The earlier B300 matrix used uninitialized `torch.empty` query/KV allocations,
did not scatter the standard KV values into the localized pools, and used 25 ms
warmup plus 100 ms repeat windows. On the 72 common shapes, changing to
identical random data and the longer B200 timing windows changes the result as
follows:

| Matrix | Earlier B300 empty-data geomean | New B300 random-data geomean | B200 random-data geomean |
| --- | ---: | ---: | ---: |
| `Sq=1` | 1.0161x | 1.0347x | 1.0315x |
| `Sq=4` | 1.0310x | 0.9970x | 0.9932x |

Relative to the earlier B300 measurement, the geometric-mean absolute latency
increased by 11.7% standard / 9.7% localized for `Sq=1`, and by 27.4%
standard / 31.7% localized for `Sq=4`. The random-data `Sq=4, B=2` behavior
now agrees qualitatively with B200 and accounts for most of the all-matrix
geomean reduction.

Against B200 under the aligned protocol, B300 geometric-mean latency is 2.8%
lower standard / 3.1% lower localized for `Sq=1`, and 2.5% lower standard /
2.9% lower localized for `Sq=4`. The within-device speedup geomeans are also
close: 1.0347x versus 1.0315x for `Sq=1`, and 0.9970x versus 0.9932x for
`Sq=4`.

## Workload and environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA B300 SXM6 AC, SM103, 148 SMs, 275,040 MiB HBM |
| Localized topology | 35 / 39 physical 2-SM clusters (70 / 78 SMs) |
| Software | Python 3.12.3, PyTorch 2.13.0+cu130, CUDA runtime 13.0, Triton 3.7.1 |
| Decode shapes | `Sq=1` and `Sq=4`, `H=128`, BF16, latent/rope dimensions `512/64` |
| Paged KV | page size 64, fixed identity-contiguous logical page table |
| Batch axis | 2, 4, 8, 16, 32, 64 |
| `seqlen_k` axis | 512 through 1,048,576 (12 values) |
| Capacity endpoint | one KV layout 72 GiB; paired layouts 144 GiB at `B=64` |
| Standard | original modular MLA, static persistent scheduler, cudaMalloc KV |
| Localized | owner-local static persistent scheduler, two RM-localized KV pools |

Allocation, random initialization, scatter, and teardown are outside the timed
region. A representative random-data correctness check at
`B=4, Sk=16,384` passed for both `Sq=1` and `Sq=4`: standard and localized
output and LSE tensors were bitwise identical with maximum absolute difference
zero.

## Timing and stability

The retained `Sq=1` and `Sq=4` results are both under
`iter_000_evaluation`. They use the same blocked
`triton.testing.do_bench` cold-L2 policy as the B200 result:

- 20 paired warmups before measurement.
- Four balanced blocks in AB, BA, BA, AB order.
- 500 ms warmup and 1,000 ms repeat window for each implementation in every
  block, with at least 20 samples.
- The reported latency is the median of four per-mode block medians; speedup is
  standard latency divided by localized latency.

Observed sustained board power was approximately 1.08-1.10 kW, with GPU
temperature at or below 61 C during the monitored long-sequence region. No
application-clock lock was applied.

Block-level stability for the retained matrices is:

| Matrix | Per-mode spread P95 / max | Paired-speedup spread P95 / max | Cells above 10% |
| --- | ---: | ---: | ---: |
| `Sq=1` | 0.76% / 6.13% | 1.13% / 6.13% | 0 |
| `Sq=4` | 2.05% / 8.44% | 3.39% / 8.86% | 0 |

Here spread is `max(block median) / min(block median) - 1`. Very short kernels,
including the roughly 25-us `Sq=4, B=16, Sk=1,024` point, remain sensitive to
discrete clock and event-timing shifts, so their individual speedups should not
be over-interpreted.

## Artifacts

- `sq1/iter_000_evaluation/post_flops.json`: retained `Sq=1` timings and
  per-block samples.
- `sq1/iter_000_evaluation/figures/performance_comparison.png`: retained
  `Sq=1` heatmap.
- `sq4/iter_000_evaluation/post_flops.json`: retained `Sq=4` timings and
  per-block samples.
- `sq4/iter_000_evaluation/figures/performance_comparison.png`: retained
  `Sq=4` heatmap and owner-wave overlay.

## Reproduction

Run the benchmark once for each query length, replacing `<sq>` and `<output>`:

```bash
MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2 \
  .venv/bin/python benchmarks/bench_cute_dsl_localized_mla.py \
  --device 0 --seqlen-q <sq> \
  --batch-sizes 2 4 8 16 32 64 \
  --seqlen-ks 512 1024 2048 4096 8192 16384 32768 65536 \
    131072 262144 524288 1048576 \
  --paired-warmups 20 --timing-warmup-ms 500 \
  --timing-repeat-ms 1000 --timing-blocks 4 --timing-min-samples 20 \
  --timing-method blocked-do-bench --data-initialization random \
  --output <output>
```
