# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Partition-localized fixed-length monolithic MLA prefill experiment."""

from __future__ import annotations

import math
from typing import Optional

import torch
from cutlass import Float32, Int32

from ..monolithic.mla_decode import (
    _check_can_implement,
    _get_compiled_mla_kernel,
)
from .localized_mla import (
    _load_localized_extension,
    _recover_cluster_metadata,
    _tensor_from_cuda_pointer,
)


_LATENT_DIM = 512
_ROPE_DIM = 64
_HEADS = 128
_PAGE_SIZE = 64


def _choose_batch_cut(
    batch_size: int, seq_len_q: int, partition_clusters: tuple[int, int]
) -> int:
    """Minimize the slower owner's persistent-wave count."""
    p0_clusters, p1_clusters = partition_clusters
    if batch_size < 2:
        raise ValueError("batch ownership requires at least two batches")
    if seq_len_q < 1 or p0_clusters < 1 or p1_clusters < 1:
        raise ValueError("query length and partition cluster counts must be positive")

    target_fraction = p0_clusters / (p0_clusters + p1_clusters)

    def score(cut: int) -> tuple[int, float]:
        waves_p0 = math.ceil(cut * seq_len_q / p0_clusters)
        waves_p1 = math.ceil((batch_size - cut) * seq_len_q / p1_clusters)
        return max(waves_p0, waves_p1), abs(cut / batch_size - target_fraction)

    return min(range(1, batch_size), key=score)


class LocalizedMLAPrefillKVCache:
    """Own two RM-localized MLA KV pools split at a batch boundary."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        *,
        seq_len_q: int,
        page_size: int = _PAGE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("localized MLA prefill requires a CUDA device")
        if batch_size < 2:
            raise ValueError("localized MLA prefill requires B >= 2")
        if seq_len_q < 1:
            raise ValueError("seq_len_q must be positive")
        if seq_len <= 0 or seq_len % page_size:
            raise ValueError("seq_len must be positive and page-aligned")
        if page_size != _PAGE_SIZE:
            raise ValueError(
                f"localized MLA prefill fixes page_size={_PAGE_SIZE}, got {page_size}"
            )
        if dtype != torch.bfloat16:
            raise ValueError("localized MLA prefill supports BF16 only")

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
        if set(sm_partition_host) != {0, 1}:
            self._context.close()
            raise RuntimeError(
                f"localized MLA prefill requires exactly owners 0 and 1, "
                f"got {sorted(set(sm_partition_host))}"
            )

        sm_cluster_rank_host, cluster_count_host = _recover_cluster_metadata(
            extension, self.device_id, sm_partition_host
        )
        self.partition_cluster_counts = tuple(cluster_count_host)
        self.total_resident_clusters = sum(self.partition_cluster_counts)
        if self.total_resident_clusters * 2 != expected_sm_count:
            self._context.close()
            raise RuntimeError(
                "stable 2-CTA topology did not cover exactly one cluster per SM pair"
            )

        self.batch_p0 = _choose_batch_cut(
            batch_size, seq_len_q, self.partition_cluster_counts
        )
        self.batch_p1 = batch_size - self.batch_p0
        p0_pages = self.batch_p0 * self.pages_per_batch
        p1_pages = self.batch_p1 * self.pages_per_batch
        self.owner_page_counts = (p0_pages, p1_pages)

        element_size = torch.empty((), dtype=dtype).element_size()
        bytes_per_page = page_size * (_LATENT_DIM + _ROPE_DIM) * element_size
        pointer_p0, mapped_p0, pointer_p1, mapped_p1 = self._context.allocate(
            p0_pages * bytes_per_page,
            p1_pages * bytes_per_page,
        )
        if pointer_p0 == 0 or pointer_p1 == 0:
            self._context.close()
            raise RuntimeError("localized MLA prefill requires two non-empty pools")
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
            sm_partition_host, dtype=torch.int32, device=self.device
        )
        self.sm_cluster_rank = torch.tensor(
            sm_cluster_rank_host, dtype=torch.int32, device=self.device
        )
        self.partition_cluster_count = torch.tensor(
            cluster_count_host, dtype=torch.int32, device=self.device
        )
        page_ids_p0 = torch.arange(
            p0_pages, dtype=torch.int32, device=self.device
        ).view(self.batch_p0, self.pages_per_batch)
        page_ids_p1 = torch.arange(
            p1_pages, dtype=torch.int32, device=self.device
        ).view(self.batch_p1, self.pages_per_batch)
        self.page_table = torch.cat((page_ids_p0, page_ids_p1), dim=0)

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
            raise ValueError("contiguous KV dtype/device does not match localized pools")
        p0_pages = self.owner_page_counts[0]
        self.kv_p0.copy_(contiguous_kv[:p0_pages])
        self.kv_p1.copy_(contiguous_kv[p0_pages:])

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self.kv_p0 = None
        self.kv_p1 = None
        self.page_table = None
        self.sm_partition_map = None
        self.sm_cluster_rank = None
        self.partition_cluster_count = None
        self._context.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def localized_mla_prefill(
    query: torch.Tensor,
    cache: LocalizedMLAPrefillKVCache,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-length causal monolithic MLA with owner-local KV pools."""
    if cache._closed:
        raise RuntimeError("localized MLA prefill cache is closed")
    expected_query_shape = (
        cache.batch_size,
        cache.seq_len_q,
        _HEADS,
        _LATENT_DIM + _ROPE_DIM,
    )
    if (
        query.dtype != torch.bfloat16
        or query.device != cache.device
        or tuple(query.shape) != expected_query_shape
    ):
        raise ValueError(
            f"query must be BF16 on {cache.device} with shape {expected_query_shape}"
        )
    if not query.is_contiguous():
        query = query.contiguous()
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.device != cache.device
        or tuple(seq_lens.shape) != (cache.batch_size,)
        or not bool(torch.all(seq_lens == cache.seq_len).item())
    ):
        raise ValueError("localized MLA prefill requires one fixed int32 KV length")

    output_shape = (
        cache.batch_size,
        cache.seq_len_q,
        _HEADS,
        _LATENT_DIM,
    )
    lse_shape = (cache.batch_size, cache.seq_len_q, _HEADS)
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bfloat16, device=cache.device)
    elif (
        out.dtype != torch.bfloat16
        or out.device != cache.device
        or tuple(out.shape) != output_shape
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous BF16 with shape {output_shape}")
    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=cache.device)
    elif (
        lse.dtype != torch.float32
        or lse.device != cache.device
        or tuple(lse.shape) != lse_shape
        or not lse.is_contiguous()
    ):
        raise ValueError(f"lse must be contiguous FP32 with shape {lse_shape}")

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
    compiled_kernel = _get_compiled_mla_kernel(
        torch_dtype=torch.bfloat16,
        torch_out_dtype=torch.bfloat16,
        page_size=cache.page_size,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        num_heads=_HEADS,
        seq_len_q=cache.seq_len_q,
        is_persistent=True,
        is_var_seq=False,
        is_var_q=False,
        is_var_split_kv=False,
        is_workspace_size_zero=True,
        enable_pdl=False,
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
        None,
        Int32(1),
        seq_lens,
        None,
        None,
        Int32(0),
        None,
        cache.sm_partition_map,
        cache.sm_cluster_rank,
        cache.partition_cluster_count,
        Int32(cache.total_resident_clusters),
        Int32(cache.batch_p0),
        Float32(softmax_scale),
        Float32(output_scale),
    )
    return (out, lse) if return_lse else out


__all__ = ["LocalizedMLAPrefillKVCache", "localized_mla_prefill"]
