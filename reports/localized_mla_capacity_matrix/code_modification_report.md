# Split-granular localized MLA design

## Scope

The implementation deliberately targets the narrow experiment requested here:
BF16 modular MLA decode, fixed `Sq=1`, `H=128`, latent/rope dimensions
`512/64`, page size 64, fixed equal sequence lengths, and a fixed page table.
It does not implement append, variable sequence lengths, variable split-KV,
FP8 localized storage, or production lifecycle integration.

The original modular MLA path remains the default. Partition awareness is a
separate compile-time experimental path selected by `partition_aware=True`.

## Data placement

CUDA RM reports the physical uGPU owner for each SM and allocates two virtual
memory pools with affinity to those owners. The logical `(batch, split_kv)`
space is flattened in batch-major order:

```text
global_work = batch_idx * split_kv + split_idx
```

If there are `W = B * split_kv` work units, P0 receives the contiguous prefix
`[0, work_p0)` and P1 receives `[work_p0, W)`. `work_p0` is chosen
proportionally to each partition's count of resident two-CTA clusters, with a
one-wave constraint when possible.

Each MLA split covers a contiguous range of 128-token K tiles. With the fixed
64-token page size, split boundaries are page-aligned. Therefore the work
prefix/suffix also induces a single flattened page prefix/suffix:

```text
P0 KV pool = global pages [0, p0_pages)
P1 KV pool = global pages [p0_pages, total_pages)
```

The global logical page table stays fixed in shape and ordering. Entries in
the P0 page range contain their P0-local page number; entries in the P1 range
contain their P1-local page number. No owner bit and no second page table are
needed because the executing physical partition already determines which KV
pool is legal for that work unit.

## Why there are two descriptor sets

Each logical datum exists in exactly one partition. The two descriptor sets do
not duplicate data: a TMA descriptor encodes the base address, shape, strides,
and element type of one allocation. P0 and P1 are distinct virtual allocations
with different base addresses and sizes, so they necessarily need distinct TMA
descriptors for latent K, rope K, and transposed latent V.

At kernel entry, the CTA reads `%smid`, maps it to P0 or P1, and prefetches only
that partition's three KV descriptors. The loader selects that descriptor set
once per CTA before entering its persistent loop. The hot per-K-tile TMA load
path is otherwise the original loader; it does not branch between P0/P1 for
each page or copy.

## Scheduler behavior

Both the original and experimental kernels use static persistent scheduling.
Neither fetches work from a public HBM atomic counter.

The original scheduler assigns the next flattened global work item using the
full resident-grid stride. The experimental scheduler first pins every
physical two-CTA cluster to the owner returned by RM, then uses a local rank and
local stride inside that owner's work interval.

For partition `p`, define:

```text
local_count(0) = work_p0
local_count(1) = B * split_kv - work_p0
base(0) = 0
base(1) = work_p0
```

A cluster with owner-local rank `r` starts at local work `r` and advances by
that owner's resident cluster count `C_p`:

```text
local_work = r, r + C_p, r + 2*C_p, ...
global_work = base(p) + local_work
batch_idx, split_idx = divmod(global_work, split_kv)
```

`Sq` tiles remain nested inside each `(batch, split)` work unit. In this
experiment `Sq=1`, so one work unit corresponds to one logical split-KV CTA
cluster. Invalid tail work exits through the existing persistent-loop validity
check.

This split-level ownership is finer than the earlier batch-only design. It can
use both partitions even for a small batch, because split-KV produces up to 32
independent work units per batch.

## Original versus localized path

| Component | Original modular MLA | Experimental localized MLA |
| --- | --- | --- |
| KV allocation | one ordinary cudaMalloc pool | two RM-affine owner-local pools |
| Logical data copy count | one | one; every page resides in one pool only |
| KV TMA descriptors | one set | one set per pool; one set selected per CTA |
| SM topology | ignored | `%smid` -> owner and owner-local cluster rank |
| Scheduler domain | global static stride | owner-local prefix/suffix static stride |
| Work granularity | `(batch, split, Sq tile)` | same, ownership cut at `(batch, split)` |
| Dynamic atomic queue | none | none |
| Page table | global physical page IDs | same fixed table shape, owner-local page IDs |
| Split-KV reducer | original | unchanged |

KV loading still uses TMA. Query TMA, KV TMA, the compute pipeline, softmax,
MMA, workspace layout, and split-KV reduction are not redesigned.

## Advantages over cudaMalloc partition-aware compact placement

With a compact cudaMalloc implementation, the page table and work scheduler
can be partition-aware, but the physical pages are not guaranteed to reside on
the uGPU that executes their work. A compact layout improves address locality
and reduces page-table indirection, yet RM may still back the allocation with
remote-partition memory, leaving substantial LTC fabric traffic.

Localized allocation adds the missing physical-placement guarantee:

- KV pages are backed by memory local to the intended SM partition.
- Static scheduling and physical placement use the same ownership map.
- The fixed page table can use small owner-local page IDs.
- NCU shows sequence-dependent remote traffic is almost eliminated: 99.96%
  fewer LTC fabric requests at B=64, Sk=1,754,432.

There is intentionally no partition-aware cudaMalloc control in this run.
Consequently, the physical-locality advantage is supported directly by the
LTC traffic result and by design, but its latency contribution is not isolated
from the split-granular scheduler change in a separate measured curve.

The corresponding costs are:

- CUDA driver/RM-specific allocation and topology-probe code.
- Two allocation lifetimes and two sets of TMA descriptors.
- Extra kernel arguments plus `%smid` and topology loads at kernel entry.
- A partition-local static stride can underfill one owner even if the other has
  idle clusters; there is no work stealing.
- Allocation sizes are rounded to RM granularity, which is visible for tiny KV
  pools.
- The prototype assumes stable 2-SM physical cluster pairing and rejects a
  topology it cannot prove safe.

The measured result is therefore the expected tradeoff: short kernels can lose
roughly 5-10% because constant overhead dominates, while long-sequence kernels
gain about 3-5% and nearly eliminate the targeted fabric traffic.

## Files

- `flashinfer/cute_dsl/attention/experimental/localized_mla.py`: narrow Python
  API, split/page cut, localized pools, page-table construction, topology data.
- `flashinfer/cute_dsl/attention/experimental/localized_mla_ext.cu`: CUDA
  driver/RM allocation and physical cluster probes.
- `flashinfer/cute_dsl/attention/scheduler/mla_persistent.py`: owner-local
  static persistent scheduler.
- `flashinfer/cute_dsl/attention/mla_decode.py`: partition lookup, descriptor
  construction/prefetch/selection, experimental grid launch.
- `flashinfer/cute_dsl/attention/wrappers/batch_mla.py`: compile and launch
  signature plumbing; original callers pass null experimental arguments.
- `tests/attention/test_cute_dsl_mla_localized.py`: correctness and placement
  checks.
- `benchmarks/bench_cute_dsl_localized_mla.py`: paired timing matrix.
- `benchmarks/profile_cute_dsl_localized_mla_ltc.py`: isolated NCU orchestration.
- `benchmarks/plot_cute_dsl_localized_mla.py`: report plots.
