#!/usr/bin/env python3
"""Check FlashInfer's existing SM90 MLA backend against absorbed-MLA math."""

import json
import math
from pathlib import Path

import torch

import flashinfer


def main():
    torch.manual_seed(42)
    batch, sk, heads, page = 2, 1024, 128, 64
    scale = (0.1 * math.log(40) + 1) ** 2 / math.sqrt(192)
    dtype, device = torch.bfloat16, "cuda:0"
    ckv = torch.randn(batch * sk // page, page, 512, device=device, dtype=dtype)
    kpe = torch.randn(batch * sk // page, page, 64, device=device, dtype=dtype)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8),
        backend="fa3",
    )
    records = []
    for sq in (1, 4, 128):
        qn = torch.randn(batch * sq, heads, 512, device=device, dtype=dtype)
        qp = torch.randn(batch * sq, heads, 64, device=device, dtype=dtype)
        wrapper.plan(
            metadata=flashinfer.mla.MLAPlanMetadata.csr(
                torch.arange(batch + 1, device=device, dtype=torch.int32) * sq,
                torch.arange(batch + 1, device=device, dtype=torch.int32)
                * (sk // page),
                torch.arange(batch * sk // page, device=device, dtype=torch.int32),
                torch.full((batch,), sk, device=device, dtype=torch.int32),
            ),
            num_heads=heads,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=page,
            causal=False,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
            query_layout="split",
            kv_cache_layout="split",
            lse_mode="base2",
        )
        out, lse = wrapper.run(query=(qn, qp), kv_cache=(ckv, kpe), return_lse=True)
        scores = (
            torch.einsum(
                "bqhd,bkd->bhqk",
                qn.reshape(batch, sq, heads, 512).float(),
                ckv.reshape(batch, sk, 512).float(),
            )
            + torch.einsum(
                "bqhd,bkd->bhqk",
                qp.reshape(batch, sq, heads, 64).float(),
                kpe.reshape(batch, sk, 64).float(),
            )
        ) * scale
        ref = torch.einsum(
            "bhqk,bkd->bqhd", scores.softmax(-1), ckv.reshape(batch, sk, 512).float()
        ).reshape_as(out)
        ref_lse = (scores.logsumexp(-1).transpose(1, 2) / math.log(2)).reshape_as(lse)
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3)
        records.append(
            {
                "status": "passed",
                "batch": batch,
                "seqlen_q": sq,
                "seqlen_k": sk,
                "max_abs_output_error": (out.float() - ref).abs().max().item(),
                "max_abs_lse_error": (lse - ref_lse).abs().max().item(),
            }
        )
        print(records[-1], flush=True)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "backend": "flashinfer fa3 SM90",
        "same_kernel_as_blackwell_experiment": False,
        "dtype": str(dtype),
        "heads": heads,
        "page_size": page,
        "softmax_scale": scale,
        "causal": False,
        "cases": records,
    }
    Path(__file__).with_name("flashinfer_hopper.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
