# Repository-local H200 partition runtime

`runtime.cu` provides a TVM-FFI JIT module for cudaMalloc allocation, validated
hash recovery/SM affinity probing, and owner-local compact CKV/KPE copies.
`flashinfer/mla/experimental/partition_runtime.py` owns the arena lifetime and
constructs per-owner page slots. No installed `partition_kv` module is used.

`cuda_malloc_hash.cuh` and `include/flashinfer/partition/address.cuh` are local
source snapshots from vllm-fa. The latter was imported as `partition_kv.h` and
moved to the framework-independent include tree so attention and scatter/gather
share exactly the same address mapping. The old `partition_kv.h` is now a
compatibility include. `UPSTREAM.json` records the source revision, exact source
hashes, and current paths; `LICENSE.upstream` retains the BSD-3-Clause license.
These snapshots contain no
runtime path lookup and are compiled directly from this repository. The runtime
intentionally excludes the external RM/localized allocator and Torch extension
bindings. Its new bindings use FlashInfer's existing TVM-FFI build path.

This version supports H200, with the validated 64 GiB arena required for hash
recovery. Reserve additional space for ordinary inputs and scratch before
allocation; allocate once and reuse sequentially. A layout maps each physical KV
page to an owner and owner-local slot. CKV and KPE use separate compact pool spans,
with 4 KiB page-pair remapping selecting the owner's physical pages. Both dtype
views are opaque physical storage; they must not be passed as ordinary linear
KV tensors to ordinary attention. The experimental `compact_kv=True` SM90
specialization explicitly accepts the arena and slot metadata and remaps its
`cp.async` source addresses. It supports noncausal BF16, H128, CKV/KPE dimensions
512/64 and page size 64. Allocation, probing, owner planning, and scattering
take place before capture or kernel timing.

Initialize outside graph capture. `close()` waits for device work and frees the
arena. Metadata, runtime and layouts must remain alive until GPU use completes;
layouts share the same arena starting at logical offset zero, so scattering a
new layout replaces prior contents. See `docs/design_docs/batch_mla_backend_architecture.md`
for the static scheduling experiment and validation requirements.
