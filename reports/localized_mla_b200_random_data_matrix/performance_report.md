# FlashInfer localized MLA on B200: random-data performance matrix

## Result

The random-data matrix completed all 144 requested shapes: 72 for `Sq=1` and
72 for `Sq=4`. Query and KV tensors contain deterministic random BF16 values
(`torch.manual_seed(42)` followed by `torch.randn`). Each shape first creates
the standard KV tensor, then scatters those exact values into the localized
pools, so the two layouts read identical KV data.

The main results are:

| Workload region | Wins | Geomean | Median | Range |
| --- | ---: | ---: | ---: | ---: |
| `Sq=1`, all cells | 52/72, 3 ties | 1.0315x | 1.0456x | 0.8924x-1.1151x |
| `Sq=1`, `Sk >= 16,384` | 41/42, 1 tie | 1.0652x | 1.0685x | 1.0000x-1.1151x |
| `Sq=1`, `Sk >= 65,536` | 30/30 | 1.0771x | 1.0816x | 1.0297x-1.1151x |
| `Sq=1`, `Sk >= 524,288` | 12/12 | 1.0885x | 1.0928x | 1.0616x-1.1130x |
| `Sq=4`, all cells | 54/72, 2 ties | 0.9932x | 1.0244x | 0.7033x-1.1461x |
| `Sq=4`, `B >= 4` | 54/60, 2 ties | 1.0343x | 1.0335x | 0.9163x-1.1461x |
| `Sq=4`, `B >= 4, Sk >= 16,384` | 35/35 | 1.0512x | 1.0391x | 1.0146x-1.1461x |
| `Sq=4`, `B = 2` | 0/12 | 0.8110x | 0.7748x | 0.7033x-0.9990x |

The capacity endpoints are:

- `Sq=1, B=64, Sk=1,048,576`: 18.3938 ms standard versus 16.8242 ms
  localized, or 1.0933x.
- `Sq=4, B=64, Sk=1,048,576`: 71.4599 ms standard versus 63.0331 ms
  localized, or 1.1337x.

For `Sq=4`, the all-matrix geometric mean is dragged below one entirely by
`B=2`. Its localized owner work has one wave on one partition and two waves
on the other, which is marked by the purple outline in the heatmap. Excluding
that known owner-wave mismatch, all 35 long-sequence cells at `B >= 4` win.

## Workload and environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA B200, SM100, 148 SMs, 178.34 GiB HBM |
| Localized topology | 39 / 35 physical 2-SM clusters (78 / 70 SMs) |
| Software | Python 3.12.3, PyTorch 2.12.1+cu132, CUDA runtime 13.2 |
| Decode shapes | `Sq=1` and `Sq=4`, `H=128`, BF16, latent/rope dimensions `512/64` |
| Paged KV | page size 64, fixed identity-contiguous logical page table |
| Batch axis | 2, 4, 8, 16, 32, 64 |
| `seqlen_k` axis | 512 through 1,048,576 (12 values) |
| Capacity endpoint | one KV layout 72 GiB; paired layouts 144 GiB at `B=64` |
| Standard | original modular MLA, static persistent scheduler, cudaMalloc KV |
| Localized | owner-local static persistent scheduler, two RM-localized KV pools |

Allocation, random initialization, scatter, and teardown are outside the
timed region; this remains a kernel-only comparison.

A representative correctness spot check at `B=4, Sk=16,384` passed for both
`Sq=1` and `Sq=4`: standard and localized output tensors and LSE tensors were
bitwise identical (maximum absolute difference 0).

## Timing and stability

The retained result is `iter_000_evaluation`. It uses the blocked
`triton.testing.do_bench` method and cold-L2 policy with a long steady-state
window for random data:

- 20 paired warmups before measurement.
- Four balanced blocks in AB, BA, BA, AB order.
- 500 ms warmup and 1,000 ms repeat window for each implementation in every
  block, with at least 20 samples.
- The reported latency is the median of the four block medians; the speedup is
  standard latency divided by localized latency.

Random BF16 inputs drove high sustained power. During measurement preparation,
board power reached approximately 900-978 W and software power capping was
observed; application clocks could not be locked without administrator
permission. The long timing windows reduce order-sensitive clock drift.

With the longer final windows, block-level stability was:

| Matrix | Per-mode spread P95 / max | Paired-speedup spread P95 / max | Cells above 10% |
| --- | ---: | ---: | ---: |
| `Sq=1` | 1.51% / 8.20% | 2.34% / 8.20% | 0 |
| `Sq=4` | 2.69% / 5.58% | 3.23% / 7.91% | 0 |

Here spread is `max(block median) / min(block median) - 1`. The preliminary
runs had maximum paired-speedup spreads of 55.52% (`Sq=1`) and 30.96%
(`Sq=4`), which is why they were discarded before the retained matrices were
renumbered.

## Artifacts

- `sq1/iter_000_evaluation/post_flops.json`: authoritative `Sq=1` timings,
  per-block samples, scheduler geometry, and initialization metadata.
- `sq1/iter_000_evaluation/figures/performance_comparison.png`: `Sq=1`
  heatmap.
- `sq4/iter_000_evaluation/post_flops.json`: authoritative `Sq=4` timings,
  per-block samples, scheduler geometry, and initialization metadata.
- `sq4/iter_000_evaluation/figures/performance_comparison.png`: `Sq=4`
  heatmap and owner-wave mismatch overlay.
