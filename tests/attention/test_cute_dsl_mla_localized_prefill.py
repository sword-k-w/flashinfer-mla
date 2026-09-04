# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness checks for partition-localized monolithic MLA prefill."""

import math

import pytest
import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.attention.experimental import (
    LocalizedMLAPrefillKVCache,
    localized_mla_prefill,
)
from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
    _get_compiled_mla_kernel,
)
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported


_LATENT_DIM = 512
_ROPE_DIM = 64
_HEADS = 128
_PAGE_SIZE = 64


def _skip_if_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if not is_cute_dsl_arch_supported(major, minor):
        pytest.skip("installed CuTe DSL does not support this GPU")
    name = torch.cuda.get_device_name()
    if "B200" not in name and "B300" not in name:
        pytest.skip("localized allocation currently supports only B200/B300")


def _run_standard(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len_q, _, _ = query.shape
    out = torch.empty(
        batch_size,
        seq_len_q,
        _HEADS,
        _LATENT_DIM,
        dtype=torch.bfloat16,
        device=query.device,
    )
    lse = torch.empty(
        batch_size,
        seq_len_q,
        _HEADS,
        dtype=torch.float32,
        device=query.device,
    )
    kernel = _get_compiled_mla_kernel(
        torch_dtype=torch.bfloat16,
        torch_out_dtype=torch.bfloat16,
        page_size=_PAGE_SIZE,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        num_heads=_HEADS,
        seq_len_q=seq_len_q,
        is_persistent=True,
        is_var_seq=False,
        is_var_q=False,
        is_var_split_kv=False,
        is_workspace_size_zero=True,
        enable_pdl=False,
        partition_aware=False,
    )
    kernel(
        query[..., :_LATENT_DIM],
        query[..., _LATENT_DIM:],
        kv_cache[..., :_LATENT_DIM],
        kv_cache[..., _LATENT_DIM:],
        None,
        None,
        page_table,
        out,
        lse,
        None,
        Int32(1),
        seq_lens,
        None,
        None,
        Int32(0),
        None,
        None,
        None,
        None,
        Int32(0),
        Int32(0),
        Float32(softmax_scale),
        Float32(1.0),
    )
    return out, lse


@pytest.mark.parametrize(
    "batch_size,seq_len_q,seq_len_k",
    [(2, 1, 64), (3, 2, 192), (4, 4, 512)],
)
def test_localized_mla_prefill_is_bitwise_exact(
    batch_size: int, seq_len_q: int, seq_len_k: int
) -> None:
    _skip_if_unsupported()
    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    pages_per_batch = seq_len_k // _PAGE_SIZE
    softmax_scale = 1.0 / math.sqrt(_LATENT_DIM)

    query = torch.randn(
        batch_size,
        seq_len_q,
        _HEADS,
        _LATENT_DIM + _ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        batch_size * pages_per_batch,
        _PAGE_SIZE,
        _LATENT_DIM + _ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    page_table = torch.arange(
        batch_size * pages_per_batch, dtype=torch.int32, device=device
    ).view(batch_size, pages_per_batch)
    seq_lens = torch.full(
        (batch_size,), seq_len_k, dtype=torch.int32, device=device
    )
    expected_out, expected_lse = _run_standard(
        query, kv_cache, page_table, seq_lens, softmax_scale
    )

    with LocalizedMLAPrefillKVCache(
        batch_size,
        seq_len_k,
        seq_len_q=seq_len_q,
        device=device,
    ) as localized_cache:
        localized_cache.scatter_from(kv_cache)
        p0_pages, p1_pages = localized_cache.owner_page_counts
        torch.testing.assert_close(
            localized_cache.kv_p0, kv_cache[:p0_pages], rtol=0, atol=0
        )
        torch.testing.assert_close(
            localized_cache.kv_p1, kv_cache[p0_pages:], rtol=0, atol=0
        )
        expected_page_table = torch.cat(
            (
                torch.arange(p0_pages, dtype=torch.int32, device=device).view(
                    localized_cache.batch_p0, pages_per_batch
                ),
                torch.arange(p1_pages, dtype=torch.int32, device=device).view(
                    localized_cache.batch_p1, pages_per_batch
                ),
            ),
            dim=0,
        )
        assert torch.equal(localized_cache.page_table, expected_page_table)

        actual_out, actual_lse = localized_mla_prefill(
            query,
            localized_cache,
            seq_lens,
            softmax_scale,
            return_lse=True,
        )
        torch.cuda.synchronize(device)
        assert torch.equal(actual_out, expected_out)
        assert torch.equal(actual_lse, expected_lse)
