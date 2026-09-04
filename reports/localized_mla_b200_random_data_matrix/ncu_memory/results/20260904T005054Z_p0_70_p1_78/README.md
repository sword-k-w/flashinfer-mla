# Localized MLA NCU L2/HBM profile

## Scope

This run compares the standard MLA decode path with the two-partition localized
MLA path on one NVIDIA B200. It collects hardware-counter data only; the prior
timing JSON files are used solely as shape and `split_kv` manifests, and none of
their timing values are read into this result.

- GPU: NVIDIA B200, UUID `GPU-3b1dff74-18aa-23eb-c4c8-25f3684ac029`
- Localized partition SM counts: P0 = 70, P1 = 78; total = 148
- Grid: `Sq in {1, 4}`, `B in {2, 4, 8, 16, 32, 64}`, and
  `Sk in {512, ..., 1,048,576}` (powers of two)
- Coverage: 144 paired configurations, 288 NCU reports
- Effective collection time: 35.13 minutes from the first successful pair to
  the final pair; NCU subprocess wall-time sum was 33.02 minutes
- Nsight Compute: 2026.1.1

## Protocol

Each mode runs in its own process with the same logical random inputs (seed 42).
Every process performs three warmup launches, clearing L2 before each attention
launch, and then profiles exactly one target kernel after another cold-L2 clear.
Mode order alternates by configuration to reduce order bias.

NCU uses `--cache-control all`, `--clock-control boost`, and kernel replay. The
ten requested metrics require exactly four replay passes for every report:

- `gpu__time_duration.avg`
- `dram__bytes_read.sum` and `dram__bytes_write.sum`
- `dram__bytes_read.sum.per_second` and
  `dram__bytes_write.sum.per_second`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__t_sector_hit_rate.pct`
- `lts__t_sectors.sum`
- `lts__t_sectors_lookup_hit.sum`
- `lts__t_sectors_lookup_miss.sum`

## Aggregate results

Ratios below are localized / standard unless explicitly stated. Geometric means
weight every matrix cell equally.

| Group | Median L2 hit rate, standard | Median L2 hit rate, localized | Median L2 delta | Cells with higher L2 | HBM bandwidth ratio, geomean | Cells with higher HBM bandwidth | HBM traffic ratio, geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sq=1 | 37.895% | 47.905% | +9.855 pp | 72/72 | 0.9647x | 36/72 | 0.9977x |
| Sq=4 | 72.900% | 84.945% | +9.505 pp | 72/72 | 0.7820x | 10/72 | 0.8325x |
| Overall | 58.000% | 66.475% | +9.735 pp | 144/144 | 0.8686x | 46/144 | 0.9114x |

The clearest observations are:

1. Localized MLA increases the reported L2 sector hit rate in every measured
   cell. For Sq=1 the gain settles near 10 pp at long `Sk`. For Sq=4 the gain
   grows with the large-batch, long-context regime and peaks at +31.32 pp for
   `B=32, Sk=1,048,576`.
2. Sq=1 moves essentially the same amount of HBM data overall (0.9977x
   geomean). Its achieved HBM bandwidth is lower for short contexts and higher
   for long contexts; the per-`Sk` geomean crosses 1x between `Sk=32,768` and
   `Sk=65,536`, reaching 1.1040x at `Sk=1,048,576`.
3. Sq=4 materially reduces HBM traffic in the long-context region. Across
   batches, the traffic-ratio geomean is 0.8301x at `Sk=65,536`, 0.7214x at
   `131,072`, 0.6422x at `262,144`, 0.5900x at `524,288`, and 0.5358x at
   `1,048,576`. The minimum cell is 0.4016x at `B=32, Sk=1,048,576`.
4. Achieved bandwidth is a rate, not a traffic-volume measurement. In the
   Sq=4 long-context region, lower bandwidth accompanies much lower HBM bytes;
   it should not by itself be interpreted as a latency or throughput result.
   This run intentionally does not repeat the timing matrix, and its single
   NCU duration sample per cell is diagnostic only.

## Artifacts

- [Machine-readable aggregate](aggregate_summary.json)
- [Full matrix summary and provenance](matrix_summary.json)
- [Flat per-cell table](memory_matrix.csv)
- [Sq=1 L2 hit-rate heatmaps](plots/sq1_l2_hit_rate.png)
- [Sq=1 HBM bandwidth heatmaps](plots/sq1_hbm_total_bandwidth.png)
- [Sq=1 HBM utilization heatmaps](plots/sq1_hbm_utilization.png)
- [Sq=1 HBM traffic heatmaps](plots/sq1_hbm_total_bytes.png)
- [Sq=4 L2 hit-rate heatmaps](plots/sq4_l2_hit_rate.png)
- [Sq=4 HBM bandwidth heatmaps](plots/sq4_hbm_total_bandwidth.png)
- [Sq=4 HBM utilization heatmaps](plots/sq4_hbm_utilization.png)
- [Sq=4 HBM traffic heatmaps](plots/sq4_hbm_total_bytes.png)
- Exported per-profile CSV, target metadata, and logs are under
  [`profiles/`](profiles/). Binary `.ncu-rep` files are retained in the local
  result tree but intentionally ignored by Git, matching the reference
  experiment's storage convention.

## Validation

- 144/144 paired configurations completed.
- All 288 reports contain one target-kernel row, all ten requested metrics, and
  exactly four replay passes.
- All 144 localized launches report the fixed `(70, 78)` SM split.
- Standard and localized profiles resolve to the same target-kernel name.
- Two setup failures for the first smoke configuration remain in its historical
  attempt records and logs; its third attempt and every final configuration
  record completed successfully.

Charts can be regenerated without profiling:

```bash
.venv/bin/python benchmarks/profile_cute_dsl_localized_mla_memory.py \
  --plot-only \
  --output-root reports/localized_mla_b200_random_data_matrix/ncu_memory/results/20260904T005054Z_p0_70_p1_78
```
