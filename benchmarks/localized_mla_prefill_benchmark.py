#!/usr/bin/env python3
"""Shared setup for partition-localized monolithic MLA prefill benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.attention.experimental.localized_mla_prefill import (
    LocalizedMLAPrefillKVCache,
)
from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
    _get_compiled_mla_kernel,
)


LATENT_DIM = 512
ROPE_DIM = 64
HEADS = 128
PAGE_SIZE = 64
DTYPE = torch.bfloat16
CLUSTER_SIZE = 2
DEFAULT_BATCH_SIZES = (2, 4, 8, 16, 32, 64)
DEFAULT_SEQLEN_QS = (2, 4, 8, 16, 32)
DEFAULT_SEQLEN_KS = (512, 4096, 32768, 131072)


def softmax_scale() -> float:
    return 1.0 / math.sqrt(LATENT_DIM)


def kv_bytes(batch_size: int, seqlen_k: int) -> int:
    return batch_size * seqlen_k * (LATENT_DIM + ROPE_DIM) * DTYPE.itemsize


def fixed_tensor_bytes(batch_size: int, seqlen_q: int) -> int:
    query = (
        batch_size
        * seqlen_q
        * HEADS
        * (LATENT_DIM + ROPE_DIM)
        * DTYPE.itemsize
    )
    outputs = 2 * batch_size * seqlen_q * HEADS * LATENT_DIM * DTYPE.itemsize
    lse = 2 * batch_size * seqlen_q * HEADS * torch.float32.itemsize
    return query + outputs + lse


def compiled_kernel(*, seq_len_q: int, partition_aware: bool):
    return _get_compiled_mla_kernel(
        torch_dtype=DTYPE,
        torch_out_dtype=DTYPE,
        page_size=PAGE_SIZE,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        num_heads=HEADS,
        seq_len_q=seq_len_q,
        is_persistent=True,
        is_var_seq=False,
        is_var_q=False,
        is_var_split_kv=False,
        is_workspace_size_zero=True,
        enable_pdl=False,
        partition_aware=partition_aware,
    )


@dataclass(frozen=True)
class SchedulerGeometry:
    seq_len_q: int
    owner_batch_counts: tuple[int, int]
    resident_partition_clusters: tuple[int, int]
    standard_active_clusters: int
    localized_active_clusters: int

    @property
    def total_resident_clusters(self) -> int:
        return sum(self.resident_partition_clusters)

    @property
    def owner_tile_counts(self) -> tuple[int, int]:
        return tuple(count * self.seq_len_q for count in self.owner_batch_counts)

    @property
    def owner_wave_counts(self) -> tuple[int, int]:
        return tuple(
            math.ceil(tiles / clusters)
            for tiles, clusters in zip(
                self.owner_tile_counts,
                self.resident_partition_clusters,
                strict=True,
            )
        )

    @property
    def localized_active_fraction(self) -> float:
        return self.localized_active_clusters / self.total_resident_clusters


class PreparedMLAPrefillCase:
    """Own both KV layouts and expose kernel-only standard/localized calls."""

    _TENSOR_NAMES = (
        "query",
        "q_latent",
        "q_rope",
        "standard_kv",
        "standard_c_latent",
        "standard_c_rope",
        "localized_c_latent_p0",
        "localized_c_rope_p0",
        "localized_c_latent_p1",
        "localized_c_rope_p1",
        "standard_page_table",
        "seq_lens",
        "standard_out",
        "localized_out",
        "standard_lse",
        "localized_lse",
    )

    def __init__(
        self,
        batch_size: int,
        seqlen_q: int,
        seqlen_k: int,
        *,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        if batch_size < 2:
            raise ValueError("localized MLA prefill benchmark requires B >= 2")
        if seqlen_q <= 0:
            raise ValueError("seqlen_q must be positive")
        if seqlen_k <= 0 or seqlen_k % PAGE_SIZE:
            raise ValueError("seqlen_k must be a positive multiple of page size")

        self.batch_size = batch_size
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.device = device
        self._closed = False
        self.localized_cache = None
        self.standard_kernel = None
        self.localized_kernel = None
        for name in self._TENSOR_NAMES:
            setattr(self, name, None)

        try:
            # Match the modular MLA experiment lifecycle: reserve physically
            # localized VA first, then allocate and initialize ordinary KV.
            self.localized_cache = LocalizedMLAPrefillKVCache(
                batch_size,
                seqlen_k,
                seq_len_q=seqlen_q,
                page_size=PAGE_SIZE,
                dtype=DTYPE,
                device=device,
            )
            pages_per_batch = seqlen_k // PAGE_SIZE
            num_pages = batch_size * pages_per_batch
            torch.manual_seed(seed)
            self.standard_kv = torch.randn(
                num_pages,
                PAGE_SIZE,
                LATENT_DIM + ROPE_DIM,
                dtype=DTYPE,
                device=device,
            )
            self.localized_cache.scatter_from(self.standard_kv)
            self.query = torch.randn(
                batch_size,
                seqlen_q,
                HEADS,
                LATENT_DIM + ROPE_DIM,
                dtype=DTYPE,
                device=device,
            )
            self.standard_page_table = torch.arange(
                num_pages, dtype=torch.int32, device=device
            ).reshape(batch_size, pages_per_batch)
            self.seq_lens = torch.full(
                (batch_size,), seqlen_k, dtype=torch.int32, device=device
            )
            output_shape = (batch_size, seqlen_q, HEADS, LATENT_DIM)
            lse_shape = (batch_size, seqlen_q, HEADS)
            self.standard_out = torch.empty(output_shape, dtype=DTYPE, device=device)
            self.localized_out = torch.empty_like(self.standard_out)
            self.standard_lse = torch.empty(
                lse_shape, dtype=torch.float32, device=device
            )
            self.localized_lse = torch.empty_like(self.standard_lse)

            self.q_latent = self.query[..., :LATENT_DIM]
            self.q_rope = self.query[..., LATENT_DIM:]
            self.standard_c_latent = self.standard_kv[..., :LATENT_DIM]
            self.standard_c_rope = self.standard_kv[..., LATENT_DIM:]
            self.localized_c_latent_p0 = self.localized_cache.kv_p0[
                ..., :LATENT_DIM
            ]
            self.localized_c_rope_p0 = self.localized_cache.kv_p0[
                ..., LATENT_DIM:
            ]
            self.localized_c_latent_p1 = self.localized_cache.kv_p1[
                ..., :LATENT_DIM
            ]
            self.localized_c_rope_p1 = self.localized_cache.kv_p1[
                ..., LATENT_DIM:
            ]
            self.standard_kernel = compiled_kernel(
                seq_len_q=seqlen_q, partition_aware=False
            )
            self.localized_kernel = compiled_kernel(
                seq_len_q=seqlen_q, partition_aware=True
            )
        except BaseException:
            self.close()
            raise

    def standard_call(self):
        return self.standard_kernel(
            self.q_latent,
            self.q_rope,
            self.standard_c_latent,
            self.standard_c_rope,
            None,
            None,
            self.standard_page_table,
            self.standard_out,
            self.standard_lse,
            None,
            Int32(1),
            self.seq_lens,
            None,
            None,
            Int32(0),
            None,
            None,
            None,
            None,
            Int32(0),
            Int32(0),
            Float32(softmax_scale()),
            Float32(1.0),
        )

    def localized_call(self):
        cache = self.localized_cache
        return self.localized_kernel(
            self.q_latent,
            self.q_rope,
            self.localized_c_latent_p0,
            self.localized_c_rope_p0,
            self.localized_c_latent_p1,
            self.localized_c_rope_p1,
            cache.page_table,
            self.localized_out,
            self.localized_lse,
            None,
            Int32(1),
            self.seq_lens,
            None,
            None,
            Int32(0),
            None,
            cache.sm_partition_map,
            cache.sm_cluster_rank,
            cache.partition_cluster_count,
            Int32(cache.total_resident_clusters),
            Int32(cache.batch_p0),
            Float32(softmax_scale()),
            Float32(1.0),
        )

    def scheduler_geometry(self) -> SchedulerGeometry:
        cache = self.localized_cache
        partition_clusters = tuple(cache.partition_cluster_counts)
        owner_batches = (cache.batch_p0, cache.batch_p1)
        total_clusters = sum(partition_clusters)
        standard_active = min(self.batch_size * self.seqlen_q, total_clusters)
        localized_active = sum(
            min(batch_count * self.seqlen_q, cluster_count)
            for batch_count, cluster_count in zip(
                owner_batches, partition_clusters, strict=True
            )
        )
        return SchedulerGeometry(
            seq_len_q=self.seqlen_q,
            owner_batch_counts=owner_batches,
            resident_partition_clusters=partition_clusters,
            standard_active_clusters=standard_active,
            localized_active_clusters=localized_active,
        )

    def check_exact(self) -> dict:
        self.standard_call()
        self.localized_call()
        torch.cuda.synchronize(self.device)
        return {
            "output_exact": torch.equal(self.standard_out, self.localized_out),
            "lse_exact": torch.equal(self.standard_lse, self.localized_lse),
        }

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        for name in self._TENSOR_NAMES:
            setattr(self, name, None)
        if self.localized_cache is not None:
            self.localized_cache.close()
            self.localized_cache = None
        self.standard_kernel = None
        self.localized_kernel = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
