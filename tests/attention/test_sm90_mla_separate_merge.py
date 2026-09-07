# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""GPU regression tests for the experimental separate-merge SM90 baseline."""

import math

import pytest
import torch

from benchmarks.sm90_mla_separate_merge import SM90MLAExperiment, make_uniform_case
from flashinfer.jit.attention import gen_batch_mla_module

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires SM90",
)


def ragged_case(q_lens, kv_lens, **kwargs):
    torch.manual_seed(42)
    dtype = kwargs.pop("dtype", torch.bfloat16)
    pages = [(x + 63) // 64 for x in kv_lens]
    qo_indptr = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    kv_indptr = torch.tensor([0, *pages], dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    n_pages = sum(pages)
    case = SM90MLAExperiment(
        torch.randn(sum(q_lens), 128, 512, device="cuda", dtype=dtype),
        torch.randn(sum(q_lens), 128, 64, device="cuda", dtype=dtype),
        torch.randn(n_pages, 64, 512, device="cuda", dtype=dtype),
        torch.randn(n_pages, 64, 64, device="cuda", dtype=dtype),
        qo_indptr,
        kv_indptr,
        torch.randperm(n_pages, device="cuda", dtype=torch.int32),
        torch.tensor(kv_lens, dtype=torch.int32),
        **kwargs,
    )
    return case, qo_indptr, kv_indptr


def reference(case, qo_indptr, kv_indptr, kv_lens, causal, base_e):
    qn, qp, ckv, kpe, indices = case._args[3:8]
    out = torch.empty_like(qn, dtype=torch.float32)
    lse = torch.empty(qn.shape[:2], device=qn.device, dtype=torch.float32)
    for b, sk in enumerate(kv_lens):
        qstart, qend = map(int, qo_indptr[b : b + 2])
        selected = indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].long()
        ck = ckv[selected].reshape(-1, 512)[:sk].float()
        kp = kpe[selected].reshape(-1, 64)[:sk].float()
        for start in range(qstart, qend, 16):
            end = min(start + 16, qend)
            score = (
                torch.einsum("qhd,kd->qhk", qn[start:end].float(), ck)
                + torch.einsum("qhd,kd->qhk", qp[start:end].float(), kp)
            ) * case._args[13]
            if causal:
                qpos = torch.arange(start - qstart, end - qstart, device=qn.device)
                kpos = torch.arange(sk, device=qn.device)
                score.masked_fill_(
                    kpos[None, None, :] > (sk - (qend - qstart) + qpos)[:, None, None],
                    -torch.inf,
                )
            out[start:end] = torch.einsum("qhk,kd->qhd", score.softmax(-1), ck)
            lse[start:end] = score.logsumexp(-1) / (1 if base_e else math.log(2))
    return out, lse


def merge_mask(case):
    p = [int(x) for x in case.plan]
    host = case.int_workspace.cpu()
    count = p[0] * p[1]
    starts = host[p[5] : p[5] + count * 4].view(torch.int32)
    ends = host[p[6] : p[6] + count * 4].view(torch.int32)
    mask = torch.zeros(case.out.shape[:2].numel(), dtype=torch.bool, device="cuda")
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        mask[start:end] = True
    return mask.reshape(case.out.shape[:2])


@pytest.mark.parametrize(
    "q_lens,kv_lens,causal,base_e,dtype,expected",
    [
        ([1, 1], [1024, 1024], False, False, torch.bfloat16, "split"),
        ([4, 4], [1025, 1025], True, True, torch.bfloat16, "split"),
        ([128, 128], [1024, 1024], False, False, torch.bfloat16, "direct"),
        ([1, 128], [32769, 129], False, True, torch.bfloat16, "mixed"),
        ([1, 128], [32769, 129], True, False, torch.bfloat16, "mixed"),
        ([1, 4], [1025, 257], True, True, torch.float16, "split"),
    ],
)
def test_separate_matches_fused_and_reference(
    q_lens, kv_lens, causal, base_e, dtype, expected
):
    case, qi, ki = ragged_case(
        q_lens, kv_lens, dtype=dtype, causal=causal, lse_base_e=base_e
    )
    summary = case.plan_summary()
    if expected == "split":
        assert summary["split_work_count"] > 0
    elif expected == "direct":
        assert summary["split_work_count"] == 0
    else:
        assert summary["split_work_count"] > 0 and summary["direct_work_count"] > 0
    case.original()
    ref, ref_lse = reference(case, qi, ki, kv_lens, causal, base_e)
    torch.testing.assert_close(case.original_out.float(), ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(case.original_lse, ref_lse, rtol=1e-3, atol=1e-3)

    # Force missing writes/stale workspace reads to show up as NaNs.
    case.float_workspace.fill_(255)
    case.out.fill_(torch.nan)
    case.lse.fill_(torch.nan)
    mask = merge_mask(case)
    case.attention()
    assert case.out[mask].isnan().all()
    assert case.lse[mask].isnan().all()
    torch.testing.assert_close(
        case.out[~mask], case.original_out[~mask], rtol=0, atol=0
    )
    case.merge()
    torch.testing.assert_close(case.out, case.original_out, rtol=0, atol=0)
    torch.testing.assert_close(case.lse, case.original_lse, rtol=0, atol=0)
    case.out.fill_(torch.nan)
    case.run()
    torch.testing.assert_close(case.out, case.original_out, rtol=0, atol=0)
    torch.testing.assert_close(case.lse, case.original_lse, rtol=0, atol=0)


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("explicit_phases", [False, True])
def test_cuda_graph_replay(return_lse, explicit_phases):
    case = make_uniform_case(2, 1, 1024, return_lse=return_lse)
    case.original()
    case.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if explicit_phases:
            case.attention()
            case.merge()
        else:
            case.run()
    for _ in range(3):
        # Change the queries so replay cannot pass by reusing old partials.
        case._args[3].add_(0.01)
        case.original()
        case.float_workspace.fill_(255)
        case.out.fill_(torch.nan)
        graph.replay()
        torch.testing.assert_close(case.out, case.original_out, rtol=0, atol=0)
        if return_lse:
            torch.testing.assert_close(case.lse, case.original_lse, rtol=0, atol=0)


def test_invalid_experimental_modes():
    for backend, profiler in (("fa2", False), ("fa3", True)):
        with pytest.raises(ValueError, match="separate_merge requires"):
            gen_batch_mla_module(
                backend,
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                512,
                64,
                profiler,
                separate_merge=True,
            )
    case = make_uniform_case(2, 1, 1024)
    with pytest.raises(Exception, match="MLA phase must"):
        case.separate_module.run(*case._args, 3)
