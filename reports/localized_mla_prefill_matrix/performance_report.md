# B200 Monolithic MLA Prefill Localized-Allocation Results

## Scope

This report compares the standard BF16 monolithic MLA prefill kernel with its
batch-owner partition-aware localized-allocation specialization on one NVIDIA
B200 GPU.

- `B`: 2, 4, 8, 16, 32, 64
- `Sq`: 2, 4, 8, 16, 32, 64, 128
- `Sk`: 512, 4,096, 32,768, 131,072, 1,008,576
- Heads: 128
- Latent/RoPE dimensions: 512/64
- Page size: 64
- BF16 input/output, FP32 LSE
- Fixed Q/KV lengths, bottom-right causal, `split_kv=1`, PDL disabled
- Deterministic `torch.randn` input data with seed 42

All 210 shapes passed bitwise equality for both output and LSE before timing.

## Timing Method

The measurement follows the existing localized MLA decode experiment:

- kernel-only timing of precompiled callables;
- 20 alternating paired warmups;
- cold-L2 `triton.testing.do_bench` samples;
- 25 ms warmup and 100 ms initial repeat budget;
- repeat budget expansion until at least 20 samples;
- four balanced blocks: AB, BA, BA, AB;
- final time is the median of the four per-mode block medians.

## Results

Across all 210 shapes, the geometric-mean speedup is **1.0209x**, the median is
**1.0071x**, and 130 shapes are faster with localized allocation.

| Sq | Geometric mean | Median | Faster shapes |
| ---: | ---: | ---: | ---: |
| 2 | 1.0702x | 1.0858x | 29 / 30 |
| 4 | 1.0618x | 1.0768x | 29 / 30 |
| 8 | 1.0305x | 1.0361x | 26 / 30 |
| 16 | 1.0099x | 1.0111x | 23 / 30 |
| 32 | 0.9948x | 0.9987x | 12 / 30 |
| 64 | 0.9911x | 0.9926x | 5 / 30 |
| 128 | 0.9911x | 0.9926x | 6 / 30 |

The best shape is `B=64, Sq=8, Sk=1,008,576`: 121.176 ms standard versus
105.591 ms localized, a **1.1476x** speedup. The worst shape is
`B=2, Sq=8, Sk=512`: 0.01654 ms standard versus 0.01829 ms localized, or
**0.9046x**.

At the largest `Sq=128, B=64, Sk=1,008,576` point, standard takes 1580.271 ms
and localized takes 1559.159 ms, a **1.0135x** speedup. The `Sq=64` equivalent
is essentially neutral at **1.0048x**.

## Capacity

The maximum sequence length uses 69.25 GiB per logical KV layout. During the
formal `Sq=64, B=64` capacity case, recorded HBM usage reached 168.99 GiB with
9.35 GiB free. The RM allocator has a sharp physical-allocation step near the
next sequence-length interval, so 1,008,576 is the stable capacity point used
for the long-lived matrix process.

## Conclusion

Localized allocation helps when KV streaming dominates and `Sq` is modest.
The benefit falls steadily as `Sq` grows: at `Sq=32` it is neutral on average,
and at `Sq=64/128` it is about 0.9% slower on average. This optimization should
therefore be gated by workload shape rather than enabled unconditionally for
all prefill requests.
