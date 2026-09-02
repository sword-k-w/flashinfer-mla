# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""B200/B300 split-granular localized KV experiment for modular MLA decode.

This module intentionally supports only the fixed-layout correctness and
performance experiment: BF16, Sq=1..4, H=128, L/R=512/64, fixed sequence
length, identity-contiguous pages, and two non-empty split-owner ranges.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import torch
from cutlass import Float32, Int32
from torch.utils.cpp_extension import load

from flashinfer.comm.dlpack_utils import create_dlpack_capsule
from flashinfer.cute_dsl.utils import _as_cute_dsl_workspace_i8, get_num_sm
from flashinfer.utils import device_support_pdl

from ..wrappers.batch_mla import (
    _check_can_implement,
    _compile_mla_kernel,
    _get_split_kv_and_workspace_size,
)


_LATENT_DIM = 512
_ROPE_DIM = 64
_HEADS = 128
_CLUSTER_SIZE = 2
_QK_TILE_TOKENS = 128


@functools.cache
def _load_localized_extension():
    source = Path(__file__).with_name("localized_mla_ext.cu")
    return load(
        name="flashinfer_localized_mla_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2", "-std=c++17", "-lineinfo"],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
        verbose=os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1",
    )


def _tensor_from_cuda_pointer(
    pointer: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    numel = 1
    for extent in shape:
        numel *= extent
    element_size = torch.empty((), dtype=dtype).element_size()
    capsule = create_dlpack_capsule(
        pointer,
        element_size,
        element_size,
        numel,
        dtype,
        device_id,
    )
    tensor = torch.utils.dlpack.from_dlpack(capsule.capsule)
    if dtype == torch.bfloat16:
        # The shared DLPack helper currently emits the generic float type code
        # for all floating-point types. Reinterpret its 16-bit view without a
        # copy so the external allocation is exposed as BF16.
        tensor = tensor.view(torch.bfloat16)
    tensor = tensor.view(shape)
    tensor._localized_mla_capsule = capsule
    return tensor


def _recover_cluster_metadata(
    extension,
    device_id: int,
    sm_partition_map: list[int],
) -> tuple[list[int], list[int]]:
    """Probe stable 2-SM cluster pairs and assign owner-local ranks."""
    sm_count = len(sm_partition_map)
    if sm_count % _CLUSTER_SIZE:
        raise RuntimeError(f"localized MLA requires an even SM count, got {sm_count}")

    observed_pair_sets = []
    canonical_pairs = None
    for _ in range(3):
        block_to_smid = list(extension.probe_cluster_smids(device_id, sm_count))
        if sorted(block_to_smid) != list(range(sm_count)):
            raise RuntimeError(
                "2-CTA cluster probe did not cover every SM exactly once: "
                f"observed {block_to_smid}"
            )
        pairs = [
            tuple(sorted(block_to_smid[i : i + _CLUSTER_SIZE]))
            for i in range(0, sm_count, _CLUSTER_SIZE)
        ]
        pair_set = frozenset(pairs)
        observed_pair_sets.append(pair_set)
        if canonical_pairs is None:
            canonical_pairs = pairs
    if any(pair_set != observed_pair_sets[0] for pair_set in observed_pair_sets[1:]):
        raise RuntimeError(
            "2-CTA physical cluster pairing was not stable across probes"
        )

    owner_pairs: list[list[tuple[int, int]]] = [[], []]
    for pair in sorted(canonical_pairs):
        owners = {sm_partition_map[smid] for smid in pair}
        if len(owners) != 1:
            raise RuntimeError(
                f"physical cluster {pair} crosses localized uGPU owners {owners}"
            )
        owner_pairs[owners.pop()].append(pair)
    if not owner_pairs[0] or not owner_pairs[1]:
        raise RuntimeError("cluster probe did not expose both localized owners")

    sm_cluster_rank = [-1] * sm_count
    for pairs in owner_pairs:
        for rank, pair in enumerate(pairs):
            for smid in pair:
                sm_cluster_rank[smid] = rank
    if any(rank < 0 for rank in sm_cluster_rank):
        raise RuntimeError("cluster-rank construction left an SM unassigned")
    return sm_cluster_rank, [len(owner_pairs[0]), len(owner_pairs[1])]


def _choose_work_cut(
    total_work: int,
    partition_clusters: tuple[int, int] | list[int],
    *,
    tiles_per_work: int = 1,
) -> int:
    """Choose a proportional prefix/suffix cut at split-work granularity.

    A split work unit owns all of its query tiles so that its KV interval has
    exactly one physical owner.  ``tiles_per_work`` is therefore used only to
    keep both owners within one persistent wave when that is possible.
    """
    if total_work < 2:
        raise ValueError("split-granular placement requires at least two work units")
    if tiles_per_work < 1:
        raise ValueError("tiles_per_work must be positive")
    p0_clusters, p1_clusters = partition_clusters
    total_clusters = p0_clusters + p1_clusters
    target = round(total_work * p0_clusters / total_clusters)
    if total_work * tiles_per_work <= total_clusters:
        # If possible, keep both owners within one persistent wave.
        lower = max(1, total_work - p1_clusters // tiles_per_work)
        upper = min(total_work - 1, p0_clusters // tiles_per_work)
        if lower <= upper:
            return min(upper, max(lower, target))
    return min(total_work - 1, max(1, target))


def _prefix_page_count(
    batch_size: int,
    pages_per_batch: int,
    split_kv: int,
    work_cut: int,
    page_size: int,
) -> int:
    """Return pages covered by a batch-major prefix of split work units."""
    full_batches, partial_splits = divmod(work_cut, split_kv)
    if full_batches >= batch_size:
        return batch_size * pages_per_batch
    k_tiles_per_batch = (
        pages_per_batch * page_size + _QK_TILE_TOKENS - 1
    ) // _QK_TILE_TOKENS
    k_tiles_per_split = (k_tiles_per_batch + split_kv - 1) // split_kv
    prefix_k_tiles = min(k_tiles_per_batch, partial_splits * k_tiles_per_split)
    pages_per_qk_tile = _QK_TILE_TOKENS // page_size
    partial_pages = min(pages_per_batch, prefix_k_tiles * pages_per_qk_tile)
    return full_batches * pages_per_batch + partial_pages


class LocalizedMLAKVCache:
    """Own two affine MLA KV pools partitioned at a logical split boundary."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        *,
        seq_len_q: int = 1,
        page_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("localized MLA requires a CUDA device")
        if batch_size < 2:
            raise ValueError("localized MLA requires at least two batches")
        if not 1 <= seq_len_q <= 4:
            raise ValueError("localized MLA requires 1 <= seq_len_q <= 4")
        if seq_len <= 0 or seq_len % page_size:
            raise ValueError("seq_len must be positive and page-aligned")
        if page_size != 64:
            raise ValueError("the initial localized MLA experiment fixes page_size=64")
        if dtype != torch.bfloat16:
            raise ValueError("the initial localized MLA experiment supports BF16 only")

        self.device_id = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.device = torch.device("cuda", self.device_id)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_len_q = seq_len_q
        self.page_size = page_size
        self.dtype = dtype
        self.pages_per_batch = seq_len // page_size
        self._closed = False

        extension = _load_localized_extension()
        self._context = extension.LocalizedContext(self.device_id)
        sm_partition_host = list(self._context.sm_partition_map())
        expected_sm_count = torch.cuda.get_device_properties(
            self.device_id
        ).multi_processor_count
        if len(sm_partition_host) != expected_sm_count:
            self._context.close()
            raise RuntimeError(
                f"RM returned {len(sm_partition_host)} SMs, expected {expected_sm_count}"
            )
        sm_cluster_rank_host, cluster_count_host = _recover_cluster_metadata(
            extension, self.device_id, sm_partition_host
        )

        self.split_kv, _ = _get_split_kv_and_workspace_size(
            batch_size,
            seq_len_q,
            _HEADS,
            _LATENT_DIM,
            expected_sm_count,
        )
        self.total_work = batch_size * self.split_kv
        self.work_p0 = _choose_work_cut(
            self.total_work,
            cluster_count_host,
            tiles_per_work=seq_len_q,
        )
        total_pages = batch_size * self.pages_per_batch
        while True:
            p0_pages = _prefix_page_count(
                batch_size,
                self.pages_per_batch,
                self.split_kv,
                self.work_p0,
                page_size,
            )
            if p0_pages < total_pages:
                break
            # Short sequences can have many empty logical splits. Move the cut
            # left until P1 owns at least one real page while retaining the
            # closest possible static-work balance.
            self.work_p0 -= 1
            if self.work_p0 == 0:
                self._context.close()
                raise RuntimeError("could not construct two non-empty KV owners")
        while p0_pages == 0:
            self.work_p0 += 1
            p0_pages = _prefix_page_count(
                batch_size,
                self.pages_per_batch,
                self.split_kv,
                self.work_p0,
                page_size,
            )
        self.work_p1 = self.total_work - self.work_p0
        p1_pages = total_pages - p0_pages
        self.owner_page_counts = (p0_pages, p1_pages)
        element_size = torch.empty((), dtype=dtype).element_size()
        bytes_per_page = page_size * (_LATENT_DIM + _ROPE_DIM) * element_size
        pointer_p0, mapped_p0, pointer_p1, mapped_p1 = self._context.allocate(
            p0_pages * bytes_per_page,
            p1_pages * bytes_per_page,
        )
        if pointer_p0 == 0 or pointer_p1 == 0:
            self._context.close()
            raise RuntimeError("localized MLA requires two non-empty allocations")
        self.mapped_bytes = (mapped_p0, mapped_p1)
        self.kv_p0 = _tensor_from_cuda_pointer(
            pointer_p0,
            (p0_pages, page_size, _LATENT_DIM + _ROPE_DIM),
            dtype,
            self.device_id,
        )
        self.kv_p1 = _tensor_from_cuda_pointer(
            pointer_p1,
            (p1_pages, page_size, _LATENT_DIM + _ROPE_DIM),
            dtype,
            self.device_id,
        )

        self.sm_partition_map = torch.tensor(
            sm_partition_host, dtype=torch.int32, device=device
        )
        self.sm_cluster_rank = torch.tensor(
            sm_cluster_rank_host, dtype=torch.int32, device=device
        )
        self.partition_cluster_count = torch.tensor(
            cluster_count_host, dtype=torch.int32, device=device
        )
        # Batch-major split work is cut into one prefix and one suffix. Since
        # every split covers a contiguous, page-aligned K interval, the owned
        # pages are also a prefix/suffix of the original flat cache. The page
        # table therefore needs only owner-local page numbers, not owner bits
        # or a second table; the physical partition selects the descriptor.
        flat_pages = torch.arange(
            batch_size * self.pages_per_batch, dtype=torch.int32, device=device
        )
        self.page_table = torch.where(
            flat_pages < p0_pages, flat_pages, flat_pages - p0_pages
        ).reshape(batch_size, self.pages_per_batch)

    def scatter_from(self, contiguous_kv: torch.Tensor) -> None:
        expected_shape = (
            self.batch_size * self.pages_per_batch,
            self.page_size,
            _LATENT_DIM + _ROPE_DIM,
        )
        if tuple(contiguous_kv.shape) != expected_shape:
            raise ValueError(
                f"expected contiguous KV shape {expected_shape}, "
                f"got {tuple(contiguous_kv.shape)}"
            )
        if contiguous_kv.dtype != self.dtype or contiguous_kv.device != self.device:
            raise ValueError(
                "contiguous KV dtype/device does not match localized pools"
            )
        p0_pages = self.owner_page_counts[0]
        self.kv_p0.copy_(contiguous_kv[:p0_pages])
        self.kv_p1.copy_(contiguous_kv[p0_pages:])

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self.kv_p0 = None
        self.kv_p1 = None
        self._context.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def localized_mla_decode(
    query: torch.Tensor,
    cache: LocalizedMLAKVCache,
    workspace_buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Run the narrow owner-split modular MLA experiment."""
    if cache._closed:
        raise RuntimeError("localized MLA cache is closed")
    if query.dtype != torch.bfloat16 or query.device != cache.device:
        raise ValueError("query must be BF16 on the localized cache device")
    if tuple(query.shape) != (
        cache.batch_size,
        cache.seq_len_q,
        _HEADS,
        _LATENT_DIM + _ROPE_DIM,
    ):
        raise ValueError(
            "localized MLA requires query shape "
            f"({cache.batch_size}, {cache.seq_len_q}, {_HEADS}, "
            f"{_LATENT_DIM + _ROPE_DIM})"
        )
    if (
        seq_lens.dtype != torch.int32
        or tuple(seq_lens.shape) != (cache.batch_size,)
        or not bool(torch.all(seq_lens == cache.seq_len).item())
    ):
        raise ValueError("localized MLA requires one fixed int32 sequence length")

    workspace_buffer = _as_cute_dsl_workspace_i8(workspace_buffer)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        cache.batch_size,
        cache.seq_len_q,
        _HEADS,
        _LATENT_DIM,
        get_num_sm(query.device),
    )
    if split_kv != cache.split_kv:
        raise RuntimeError(
            f"cache was partitioned for split_kv={cache.split_kv}, got {split_kv}"
        )
    if workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace has {workspace_buffer.numel()} bytes, need {workspace_size}"
        )
    workspace = None if workspace_size == 0 else workspace_buffer[:workspace_size]
    if out is None:
        out = torch.empty(
            (cache.batch_size, cache.seq_len_q, _HEADS, _LATENT_DIM),
            dtype=torch.bfloat16,
            device=query.device,
        )
    lse = torch.empty(
        (cache.batch_size, cache.seq_len_q, _HEADS),
        dtype=torch.float32,
        device=query.device,
    )

    _check_can_implement(
        torch_dtype=torch.bfloat16,
        torch_out_dtype=torch.bfloat16,
        page_size=cache.page_size,
        num_heads=_HEADS,
        seq_len_q=cache.seq_len_q,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
    )
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl
    compiled_kernel = _compile_mla_kernel(
        torch_dtype=torch.bfloat16,
        torch_out_dtype=torch.bfloat16,
        page_size=cache.page_size,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
        is_workspace_size_zero=workspace_size == 0,
        enable_pdl=enable_pdl,
        partition_aware=True,
    )

    q_latent = query[..., :_LATENT_DIM]
    q_rope = query[..., _LATENT_DIM:]
    c_latent_p0 = cache.kv_p0[..., :_LATENT_DIM]
    c_rope_p0 = cache.kv_p0[..., _LATENT_DIM:]
    c_latent_p1 = cache.kv_p1[..., :_LATENT_DIM]
    c_rope_p1 = cache.kv_p1[..., _LATENT_DIM:]
    compiled_kernel(
        q_latent,
        q_rope,
        c_latent_p0,
        c_rope_p0,
        c_latent_p1,
        c_rope_p1,
        cache.page_table,
        out,
        lse,
        workspace,
        Int32(split_kv),
        seq_lens,
        None,
        cache.sm_partition_map,
        cache.sm_cluster_rank,
        cache.partition_cluster_count,
        Int32(cache.work_p0),
        Float32(softmax_scale),
        Float32(output_scale),
        None,
    )
    return out


__all__ = ["LocalizedMLAKVCache", "localized_mla_decode"]
