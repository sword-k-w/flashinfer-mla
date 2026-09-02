#!/usr/bin/env python3
"""Shared setup for the localized partition-aware modular MLA experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.attention.experimental.localized_mla import (
    LocalizedMLAKVCache,
)
from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
    _compile_mla_kernel,
    _get_split_kv_and_workspace_size,
)
from flashinfer.cute_dsl.utils import get_num_sm


LATENT_DIM = 512
ROPE_DIM = 64
HEADS = 128
PAGE_SIZE = 64
DTYPE = torch.bfloat16
CLUSTER_SIZE = 2
DEFAULT_BATCH_SIZES = (2, 4, 8, 16, 32, 64)
# Same 128-GiB-localized-arena capacity axis used by the vllm-fa B300 MLA
# experiment.  At the final B=64 point, one KV copy is 120.47 GiB.
DEFAULT_SEQLEN_KS = (
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    1754432,
)


def deepseek_v3_effective_softmax_scale() -> float:
    yarn_scale = 0.1 * math.log(40.0) + 1.0
    return yarn_scale * yarn_scale / math.sqrt(128 + 64)


def kv_bytes(batch_size: int, seqlen_k: int) -> int:
    return batch_size * seqlen_k * (LATENT_DIM + ROPE_DIM) * DTYPE.itemsize


def compiled_kernel(*, partition_aware: bool, workspace_size_zero: bool):
    return _compile_mla_kernel(
        torch_dtype=DTYPE,
        torch_out_dtype=DTYPE,
        page_size=PAGE_SIZE,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
        is_workspace_size_zero=workspace_size_zero,
        enable_pdl=False,
        partition_aware=partition_aware,
    )


@dataclass(frozen=True)
class SchedulerGeometry:
    split_kv: int
    owner_work_counts: tuple[int, int]
    resident_partition_clusters: tuple[int, int]
    standard_active_clusters: int
    localized_active_clusters: int

    @property
    def total_resident_clusters(self) -> int:
        return sum(self.resident_partition_clusters)

    @property
    def localized_active_fraction(self) -> float:
        return self.localized_active_clusters / self.total_resident_clusters


class PreparedMLACase:
    """Own both KV layouts and expose kernel-only standard/localized calls."""

    def __init__(
        self,
        batch_size: int,
        seqlen_k: int,
        *,
        device: torch.device,
        initialize_for_correctness: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size < 2:
            raise ValueError("localized MLA benchmark requires B >= 2")
        if seqlen_k <= 0 or seqlen_k % PAGE_SIZE:
            raise ValueError("seqlen_k must be a positive multiple of page size")
        self.batch_size = batch_size
        self.seqlen_k = seqlen_k
        self.device = device
        self._closed = False
        self.softmax_scale = deepseek_v3_effective_softmax_scale()
        self.split_kv, workspace_size = _get_split_kv_and_workspace_size(
            batch_size,
            1,
            HEADS,
            LATENT_DIM,
            get_num_sm(device),
        )
        self.workspace_size = workspace_size

        # Reserve the physically localized allocation first, before the large
        # ordinary cudaMalloc allocation can fragment the virtual address space.
        self.localized_cache = LocalizedMLAKVCache(
            batch_size,
            seqlen_k,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            device=device,
        )
        if self.localized_cache.split_kv != self.split_kv:
            raise RuntimeError("localized cache and kernel split_kv disagree")
        pages_per_batch = seqlen_k // PAGE_SIZE
        num_pages = batch_size * pages_per_batch
        if initialize_for_correctness:
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
                1,
                HEADS,
                LATENT_DIM + ROPE_DIM,
                dtype=DTYPE,
                device=device,
            )
        else:
            # KV values do not affect the memory access pattern.  Avoid spending
            # minutes initializing up to 241 GiB across the paired layouts.
            self.standard_kv = torch.empty(
                num_pages,
                PAGE_SIZE,
                LATENT_DIM + ROPE_DIM,
                dtype=DTYPE,
                device=device,
            )
            self.query = torch.empty(
                batch_size,
                1,
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
        self.workspace_buffer = (
            None
            if workspace_size == 0
            else torch.empty(workspace_size, dtype=torch.int8, device=device)
        )
        output_shape = (batch_size, 1, HEADS, LATENT_DIM)
        lse_shape = (batch_size, 1, HEADS)
        self.standard_out = torch.empty(output_shape, dtype=DTYPE, device=device)
        self.localized_out = torch.empty_like(self.standard_out)
        self.standard_lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
        self.localized_lse = torch.empty_like(self.standard_lse)

        self.q_latent = self.query[..., :LATENT_DIM]
        self.q_rope = self.query[..., LATENT_DIM:]
        self.standard_c_latent = self.standard_kv[..., :LATENT_DIM]
        self.standard_c_rope = self.standard_kv[..., LATENT_DIM:]
        self.localized_c_latent_p0 = self.localized_cache.kv_p0[..., :LATENT_DIM]
        self.localized_c_rope_p0 = self.localized_cache.kv_p0[..., LATENT_DIM:]
        self.localized_c_latent_p1 = self.localized_cache.kv_p1[..., :LATENT_DIM]
        self.localized_c_rope_p1 = self.localized_cache.kv_p1[..., LATENT_DIM:]

        workspace_size_zero = workspace_size == 0
        self.standard_kernel = compiled_kernel(
            partition_aware=False, workspace_size_zero=workspace_size_zero
        )
        self.localized_kernel = compiled_kernel(
            partition_aware=True, workspace_size_zero=workspace_size_zero
        )

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
            self.workspace_buffer,
            Int32(self.split_kv),
            self.seq_lens,
            None,
            None,
            None,
            None,
            Int32(0),
            Float32(self.softmax_scale),
            Float32(1.0),
            None,
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
            self.workspace_buffer,
            Int32(self.split_kv),
            self.seq_lens,
            None,
            cache.sm_partition_map,
            cache.sm_cluster_rank,
            cache.partition_cluster_count,
            Int32(cache.work_p0),
            Float32(self.softmax_scale),
            Float32(1.0),
            None,
        )

    def scheduler_geometry(self) -> SchedulerGeometry:
        cache = self.localized_cache
        partition_clusters = tuple(
            int(value) for value in cache.partition_cluster_count.cpu().tolist()
        )
        owner_work = (cache.work_p0, cache.work_p1)
        total_clusters = sum(partition_clusters)
        standard_active = min(self.batch_size * self.split_kv, total_clusters)
        localized_active = sum(
            min(work_count, cluster_count)
            for work_count, cluster_count in zip(
                owner_work, partition_clusters, strict=True
            )
        )
        return SchedulerGeometry(
            split_kv=self.split_kv,
            owner_work_counts=owner_work,
            resident_partition_clusters=partition_clusters,
            standard_active_clusters=standard_active,
            localized_active_clusters=localized_active,
        )

    def check_exact(self) -> dict:
        self.standard_call()
        self.localized_call()
        torch.cuda.synchronize(self.device)
        output_equal = torch.equal(self.standard_out, self.localized_out)
        lse_equal = torch.equal(self.standard_lse, self.localized_lse)
        output_max_abs = (
            0.0
            if output_equal
            else float(
                (self.standard_out.float() - self.localized_out.float())
                .abs()
                .max()
                .item()
            )
        )
        lse_max_abs = (
            0.0
            if lse_equal
            else float((self.standard_lse - self.localized_lse).abs().max().item())
        )
        return {
            "output_exact": output_equal,
            "lse_exact": lse_equal,
            "output_max_abs": output_max_abs,
            "lse_max_abs": lse_max_abs,
        }

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        # Drop every external-allocation view before unmapping its RM handles.
        self.localized_c_latent_p0 = None
        self.localized_c_rope_p0 = None
        self.localized_c_latent_p1 = None
        self.localized_c_rope_p1 = None
        self.localized_cache.close()
        for name in (
            "query",
            "q_latent",
            "q_rope",
            "standard_kv",
            "standard_c_latent",
            "standard_c_rope",
            "standard_page_table",
            "seq_lens",
            "workspace_buffer",
            "standard_out",
            "localized_out",
            "standard_lse",
            "localized_lse",
        ):
            setattr(self, name, None)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def run_single_call(case: PreparedMLACase, mode: str) -> Callable[[], object]:
    if mode == "standard":
        return case.standard_call
    if mode == "localized":
        return case.localized_call
    raise ValueError(f"unknown mode {mode!r}")
