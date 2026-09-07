# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Independent H200 runtime and page-owner compact layout for MLA experiments."""

import functools
import weakref

import torch

from flashinfer.jit.partition_runtime import gen_partition_runtime_module


@functools.cache
def _module():
    return gen_partition_runtime_module().build_and_load()


class H200PartitionRuntime:
    """Own one validated cudaMalloc arena; initialize before large input tensors.

    Instances and layouts must remain alive until their CUDA work completes.
    The topology is measured for this allocation, not imported from another run.
    """

    def __init__(self, device=0, reserve_bytes=2 << 30):
        self.device = torch.device("cuda", device)
        self.module = _module()
        data = list(map(int, self.module.initialize(device, reserve_bytes)))
        (
            self.handle,
            self.arena_bytes,
            self.hash_base,
            self.mask,
            self.threshold_cycles,
        ) = data[:5]
        self._finalizer = weakref.finalize(self, self.module.release, self.handle)
        self.sm_partition_cpu = torch.tensor(data[5:], dtype=torch.int32)
        self.sm_counts_cpu = torch.bincount(
            self.sm_partition_cpu.long(), minlength=2
        ).to(torch.int32)
        self.sm_rank_cpu = torch.empty_like(self.sm_partition_cpu)
        for p in range(2):
            ids = torch.where(self.sm_partition_cpu == p)[0]
            self.sm_rank_cpu[ids] = torch.arange(ids.numel(), dtype=torch.int32)
        self.sm_partition = self.sm_partition_cpu.to(self.device)
        self.sm_rank = self.sm_rank_cpu.to(self.device)
        self.sm_counts = self.sm_counts_cpu.to(self.device)

    def close(self):
        if self._finalizer.alive:
            torch.cuda.synchronize(self.device)
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def compact_layout(self, page_owners, page_size=64):
        if not self._finalizer.alive:
            raise RuntimeError("Arena is closed")
        return CompactMLALayout(self, page_owners, page_size)


class CompactMLALayout:
    """Page-owner placement with independent CKV and KPE compact spans.

    Layouts reuse the arena from offset zero. Create/use them sequentially;
    scattering one layout invalidates previously scattered layout contents.
    """

    def __init__(self, runtime, page_owners, page_size):
        owners = torch.as_tensor(page_owners, dtype=torch.int32, device="cpu").clone()
        if (
            owners.ndim != 1
            or owners.numel() == 0
            or not ((owners == 0) | (owners == 1)).all()
        ):
            raise ValueError("Every physical KV page must have owner 0 or 1")
        if page_size <= 0 or page_size % 32:
            raise ValueError("page_size must be a positive multiple of 32")
        self.runtime, self.page_size = runtime, page_size
        self.owners_cpu = owners
        slots = torch.empty_like(owners)
        counts = []
        for p in range(2):
            ids = torch.where(owners == p)[0]
            slots[ids] = torch.arange(ids.numel(), dtype=torch.int32)
            counts.append(ids.numel())
        self.owner_page_counts = counts
        self.slot_capacity = max(counts)
        self.ckv_stride = page_size * 512 * 2
        self.kpe_stride = page_size * 64 * 2
        self.kpe_offset = self.slot_capacity * self.ckv_stride
        self.logical_span = self.kpe_offset + self.slot_capacity * self.kpe_stride
        if 2 * self.logical_span > runtime.arena_bytes:
            raise ValueError("Compact layout exceeds cudaMalloc arena capacity")
        self.owners = owners.to(runtime.device)
        self.slots_cpu = slots
        self.slots = slots.to(runtime.device)

    def _copy(self, ckv, kpe, gather):
        if not self.runtime._finalizer.alive:
            raise RuntimeError("Arena is closed")
        for tensor, dim, offset, stride in (
            (ckv, 512, 0, self.ckv_stride),
            (kpe, 64, self.kpe_offset, self.kpe_stride),
        ):
            if (
                tensor.shape != (self.owners.numel(), self.page_size, dim)
                or not tensor.is_contiguous()
            ):
                raise ValueError("Incorrect compact MLA tensor shape or strides")
            if tensor.device != self.runtime.device or tensor.dtype not in (
                torch.bfloat16,
                torch.float16,
            ):
                raise ValueError("Expected BF16/FP16 tensor on the arena device")
            self.runtime.module.copy(
                self.runtime.handle,
                tensor,
                self.owners,
                self.slots,
                offset,
                stride,
                self.slot_capacity,
                gather,
            )

    def scatter(self, ckv, kpe):
        self._copy(ckv, kpe, False)

    def gather(self, dtype=torch.bfloat16):
        ckv = torch.empty(
            (self.owners.numel(), self.page_size, 512),
            dtype=dtype,
            device=self.runtime.device,
        )
        kpe = torch.empty(
            (self.owners.numel(), self.page_size, 64),
            dtype=dtype,
            device=self.runtime.device,
        )
        self._copy(ckv, kpe, True)
        return ckv, kpe
