# Sq=4 localized MLA adaptation

## Scope

The experimental BF16 modular MLA path now accepts non-causal `Sq=1..4` while
retaining its deliberately narrow fixed-layout scope: `H=128`, latent/rope
dimensions `512/64`, page size 64, fixed equal KV lengths, and a fixed page
table. The original modular MLA path and the Sq=1 default are unchanged.

No new CUDA/CuTe kernel algorithm was required. The partition scheduler added
for Sq=1 already represented `s_idx`; the missing work was to propagate the
real query length through cache planning, splitKV/workspace sizing, validation,
benchmarks, profiling, and reports.

## Ownership and scheduling

KV ownership remains at `(batch, splitKV)` granularity, not query-tile
granularity. This is intentional: all four query tiles for one unit consume the
same K/V interval, so assigning that unit to one physical partition stores each
KV page exactly once.

For owner `p`, the scheduler now treats its local work range as:

```text
num_local_tiles = owner_work_units[p] * Sq
local_linear    = owner_cluster_rank + iteration * owner_cluster_count
local_unit, s   = divmod(local_linear, Sq)
global_unit     = owner_base[p] + local_unit
batch, split    = divmod(global_unit, splitKV)
```

The stride is still static and partition-local. There is no HBM atomic queue
and no work stealing.

The prefix/suffix ownership cut is still made in whole `(batch, splitKV)`
units. Its one-wave constraint now uses `work_units * Sq` rather than assuming
one tile per unit. This lets the 38/36-cluster B300 topology choose cuts that
balance actual query-tile work without splitting or duplicating a KV interval.

## Host/API changes

`LocalizedMLAKVCache` gained a keyword-only `seq_len_q` argument, defaulting to
1 and validated in `[1, 4]`. The value is stored in the cache because it changes
the splitKV decision and therefore changes physical KV ownership. A query must
match the Sq for which its cache was constructed.

The localized launch path now uses the configured Sq for:

- splitKV selection and workspace sizing;
- query, output, and LSE shapes;
- modular MLA capability validation;
- the work-cut one-wave calculation.

The two KV descriptor sets are unchanged. Each owner still prefetches and uses
only its local latent-K, rope-K, and transposed-latent-V TMA descriptors.

## Benchmark and profiler changes

The shared benchmark case and both command-line runners gained `--seqlen-q`
support. Sq=1 keeps the old report directory; Sq=N uses
`reports/localized_mla_sqN_capacity_matrix`, preventing accidental overwrite.

Timing and NCU metadata now record both:

- `owner_work_counts`: owned `(batch, splitKV)` units;
- `owner_tile_counts`: scheduled query tiles after multiplying by Sq.

Theoretical active-cluster counts also use total query tiles. The plotting
script derives Sq from each result document and remains compatible with older
Sq=1 NCU JSON files that lack a top-level `seqlen_q` field.

The correctness test now covers every supported Sq value. It verifies the
localized page placement/table and compares the kernel result with the
original modular MLA implementation.

## Files changed

- `flashinfer/cute_dsl/attention/experimental/localized_mla.py`
- `benchmarks/localized_mla_benchmark.py`
- `benchmarks/bench_cute_dsl_localized_mla.py`
- `benchmarks/profile_cute_dsl_localized_mla_ltc.py`
- `benchmarks/profile_cute_dsl_localized_mla_ltc_target.py`
- `benchmarks/plot_cute_dsl_localized_mla.py`
- `tests/attention/test_cute_dsl_mla_localized.py`

The kernel mainloop, TMA KV loader, softmax/MMA pipeline, splitKV reducer, and
RM allocator were not changed for Sq=4.
