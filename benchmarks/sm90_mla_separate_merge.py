# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Experimental SM90 MLA baseline with independently callable attention and merge.

Run from the repository root:
    python -m benchmarks.sm90_mla_separate_merge --case 2,1,1024

This uses the existing planner and BF16/FP16 cp.async attention path. Split
outputs remain in workspace until merge() executes on the same CUDA stream.
The wrapper is for experiments; it is not a public FlashInfer API.
"""

import argparse
import functools
import json
import math
import statistics
from pathlib import Path

import torch

from flashinfer.jit.attention import gen_batch_mla_module


@functools.cache
def _module(dtype, separate_merge):
    return gen_batch_mla_module(
        "fa3",
        dtype,
        dtype,
        dtype,
        torch.int32,
        512,
        64,
        False,
        separate_merge=separate_merge,
    ).build_and_load()


class SM90MLAExperiment:
    """Own a fixed plan, scratch buffers and outputs for repeated launches.

    Inputs must be split, contiguous BF16/FP16 tensors with CKV/KPE dimensions
    512/64. CSR indptrs and lengths are CPU int32 tensors; indices are CUDA
    int32. All CUDA tensors must be on the query device. Plan construction and
    JIT compilation happen here, outside graph capture and timed regions.
    """

    def __init__(
        self,
        q_nope,
        q_pe,
        ckv,
        kpe,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len,
        *,
        causal=False,
        sm_scale=None,
        return_lse=True,
        lse_base_e=False,
        workspace_bytes=128 * 1024 * 1024,
    ):
        if q_nope.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("The experimental baseline supports BF16/FP16 inputs")
        if (
            not q_nope.is_cuda
            or torch.cuda.get_device_capability(q_nope.device)[0] != 9
        ):
            raise ValueError("The experimental baseline requires SM90")
        tensors = (q_nope, q_pe, ckv, kpe)
        if any(
            t.device != q_nope.device
            or t.dtype != q_nope.dtype
            or not t.is_contiguous()
            for t in tensors
        ):
            raise ValueError(
                "Inputs must be contiguous, with the same CUDA device and dtype"
            )
        if (
            q_nope.ndim != 3
            or q_nope.shape[-1] != 512
            or q_pe.shape != (*q_nope.shape[:2], 64)
        ):
            raise ValueError(
                "Expected Q shapes [tokens, heads, 512] and [tokens, heads, 64]"
            )
        if ckv.ndim != 3 or ckv.shape[-1] != 512 or kpe.shape != (*ckv.shape[:2], 64):
            raise ValueError(
                "Expected KV shapes [pages, page_size, 512] and [pages, page_size, 64]"
            )
        for t in (qo_indptr, kv_indptr, kv_len):
            if (
                t.device.type != "cpu"
                or t.dtype != torch.int32
                or not t.is_contiguous()
            ):
                raise ValueError(
                    "Plan indptrs and lengths must be contiguous CPU int32"
                )
        if (
            kv_indices.device != q_nope.device
            or kv_indices.dtype != torch.int32
            or not kv_indices.is_contiguous()
        ):
            raise ValueError("KV indices must be contiguous int32 on the query device")
        if len(qo_indptr) != len(kv_len) + 1 or len(kv_indptr) != len(qo_indptr):
            raise ValueError("CSR metadata lengths do not match")
        if (
            qo_indptr[0] != 0
            or qo_indptr[-1] != q_nope.shape[0]
            or (qo_indptr.diff() < 0).any()
        ):
            raise ValueError("Invalid query indptr")
        if (
            kv_indptr[0] != 0
            or kv_indptr[-1] != kv_indices.numel()
            or (kv_indptr.diff() < 0).any()
        ):
            raise ValueError("Invalid KV indptr")
        if (kv_len < 0).any() or (kv_len > kv_indptr.diff() * ckv.shape[1]).any():
            raise ValueError("KV lengths exceed the allocated pages")
        if causal and (qo_indptr.diff() > kv_len).any():
            raise ValueError("Causal queries must not exceed KV length")
        self.original_module = _module(q_nope.dtype, False)
        self.separate_module = _module(q_nope.dtype, True)
        device = q_nope.device
        self.int_workspace = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.float_workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        self.original_workspace = torch.empty_like(self.float_workspace)
        self.host_workspace = torch.empty(
            self.int_workspace.numel(), dtype=torch.uint8, pin_memory=True
        )
        self.plan = self.separate_module.plan(
            self.float_workspace,
            self.int_workspace,
            self.host_workspace,
            qo_indptr,
            kv_indptr,
            kv_len,
            q_nope.shape[1],
            512,
            causal,
        )
        self.out = torch.empty_like(q_nope)
        self.lse = (
            torch.empty(q_nope.shape[:2], dtype=torch.float32, device=device)
            if return_lse
            else None
        )
        self.original_out = torch.empty_like(self.out)
        self.original_lse = torch.empty_like(self.lse) if return_lse else None
        if sm_scale is None:
            sm_scale = (0.1 * math.log(40) + 1) ** 2 / math.sqrt(192)
        common = (
            int(causal),
            q_nope.shape[1],
            ckv.shape[1],
            sm_scale,
            lse_base_e,
            1.0,
            1.0,
            None,
        )
        inputs = (q_nope, q_pe, ckv, kpe, kv_indices)
        self._args = (
            self.float_workspace,
            self.int_workspace,
            self.plan,
            *inputs,
            self.out,
            self.lse,
            *common,
        )
        self._original_args = (
            self.original_workspace,
            self.int_workspace,
            self.plan,
            *inputs,
            self.original_out,
            self.original_lse,
            *common,
        )

    def original(self):
        self.original_module.run(*self._original_args)
        return self.original_out, self.original_lse

    def attention(self):
        """Launch attention only; split rows in final output are not yet valid."""
        self.separate_module.run(*self._args, 1)

    def merge(self):
        """Merge the most recent attention partials, using the same stream/plan."""
        self.separate_module.run(*self._args, 2)
        return self.out, self.lse

    def run(self):
        """Launch attention followed by merge on the current CUDA stream."""
        self.separate_module.run(*self._args, 0)
        return self.out, self.lse

    def plan_summary(self):
        """Inspect the fixed planner ABI outside capture/timing."""
        p = [int(x) for x in self.plan]
        host = self.int_workspace.cpu()

        def read(offset, count):
            return host[offset : offset + count * 4].view(torch.int32)

        work_indptr = read(p[15], p[1] + 1)
        work_count = int(work_indptr[-1])
        partial_indptr = read(p[4], work_count)
        starts, ends = read(p[5], p[0] * p[1]), read(p[6], p[0] * p[1])
        return dict(
            grid=[p[0], p[1]],
            work_count=work_count,
            split_work_count=int((partial_indptr >= 0).sum()),
            direct_work_count=int((partial_indptr == -1).sum()),
            merge_rows=int((ends - starts).sum()),
        )


def make_uniform_case(batch, sq, sk, *, dtype=torch.bfloat16, causal=False, **kwargs):
    page, heads, device = 64, 128, torch.device("cuda")
    pages = (sk + page - 1) // page
    return SM90MLAExperiment(
        torch.randn(batch * sq, heads, 512, device=device, dtype=dtype),
        torch.randn(batch * sq, heads, 64, device=device, dtype=dtype),
        torch.randn(batch * pages, page, 512, device=device, dtype=dtype),
        torch.randn(batch * pages, page, 64, device=device, dtype=dtype),
        torch.arange(batch + 1, dtype=torch.int32) * sq,
        torch.arange(batch + 1, dtype=torch.int32) * pages,
        torch.arange(batch * pages, dtype=torch.int32, device=device),
        torch.full((batch,), sk, dtype=torch.int32),
        causal=causal,
        **kwargs,
    )


def graph_time_us(fn, *, launches=20, repeats=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            fn()
    graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / launches)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", help="B,Sq,Sk; repeat for multiple cases"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for value in args.case or ["2,1,1024", "64,1,32768", "64,4,32768", "2,128,1024"]:
        batch, sq, sk = map(int, value.split(","))
        case = make_uniform_case(batch, sq, sk)
        case.original()
        case.run()
        torch.testing.assert_close(case.out, case.original_out, rtol=0, atol=0)
        torch.testing.assert_close(case.lse, case.original_lse, rtol=0, atol=0)
        result = dict(batch=batch, sq=sq, sk=sk, **case.plan_summary())
        for name, fn in (
            ("fused", case.original),
            ("attention", case.attention),
            ("merge", case.merge),
            ("full", case.run),
        ):
            result[name + "_us"] = graph_time_us(fn)
        result["correctness"] = "bitwise equal to fused output and LSE"
        results.append(result)
        print(json.dumps(result), flush=True)
        del case
    report = dict(
        gpu=torch.cuda.get_device_name(),
        dtype="bfloat16",
        heads=128,
        page_size=64,
        causal=False,
        timing="CUDA events around graph replay, 20 invocations/graph, median of 10 replays; resident reused inputs",
        cases=results,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
