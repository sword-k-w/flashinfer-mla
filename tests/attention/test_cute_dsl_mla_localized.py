# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness checks for the experimental localized modular MLA path."""

import math

import pytest
import torch

from flashinfer.cute_dsl.attention import cute_dsl_mla_decode
from flashinfer.cute_dsl.attention.experimental import (
    LocalizedMLAKVCache,
    localized_mla_decode,
)
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported


def _skip_if_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if not is_cute_dsl_arch_supported(major, minor):
        pytest.skip("installed CuTe DSL does not support this GPU")
    name = torch.cuda.get_device_name()
    if "B200" not in name and "B300" not in name:
        pytest.skip("localized allocation currently supports only B200/B300")


@pytest.mark.parametrize("batch_size", [2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("seq_len", [128, 512])
@pytest.mark.parametrize("seq_len_q", [1, 2, 3, 4])
def test_localized_mla_matches_modular(
    batch_size: int, seq_len: int, seq_len_q: int
) -> None:
    _skip_if_unsupported()
    torch.manual_seed(42)

    device = torch.device("cuda", torch.cuda.current_device())
    page_size = 64
    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    pages_per_batch = seq_len // page_size
    softmax_scale = 1.0 / math.sqrt(latent_dim)

    query = torch.randn(
        batch_size,
        seq_len_q,
        num_heads,
        latent_dim + rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        batch_size * pages_per_batch,
        page_size,
        latent_dim + rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    global_page_table = torch.arange(
        batch_size * pages_per_batch, dtype=torch.int32, device=device
    ).view(batch_size, pages_per_batch)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    reference = cute_dsl_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=global_page_table,
        seq_lens=seq_lens,
        max_seq_len=seq_len,
        softmax_scale=softmax_scale,
        is_var_seq=False,
        enable_pdl=False,
        cute_dsl_impl="modular",
    )

    with LocalizedMLAKVCache(
        batch_size,
        seq_len,
        seq_len_q=seq_len_q,
        page_size=page_size,
        device=device,
    ) as localized_cache:
        localized_cache.scatter_from(kv_cache)
        p0_pages, p1_pages = localized_cache.owner_page_counts
        assert p0_pages + p1_pages == batch_size * pages_per_batch
        assert localized_cache.work_p0 + localized_cache.work_p1 == (
            batch_size * localized_cache.split_kv
        )
        torch.testing.assert_close(
            localized_cache.kv_p0, kv_cache[:p0_pages], rtol=0, atol=0
        )
        torch.testing.assert_close(
            localized_cache.kv_p1, kv_cache[p0_pages:], rtol=0, atol=0
        )
        expected_local_pages = torch.arange(
            batch_size * pages_per_batch, dtype=torch.int32, device=device
        )
        expected_local_pages[p0_pages:] -= p0_pages
        torch.testing.assert_close(
            localized_cache.page_table.flatten(),
            expected_local_pages,
            rtol=0,
            atol=0,
        )
        actual = localized_mla_decode(
            query,
            localized_cache,
            workspace,
            seq_lens,
            softmax_scale,
            enable_pdl=False,
        )
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
