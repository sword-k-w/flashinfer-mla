#!/usr/bin/env python3
"""Profile one unchanged SM90 attention launch, with setup and merge excluded."""

import argparse
import json
import math

import torch
from triton import runtime as triton_runtime

from benchmarks.bench_sm90_mla_partition import assert_result
from benchmarks.sm90_mla_partition import SM90PartitionExperiment
from benchmarks.sm90_mla_separate_merge import make_uniform_case
from flashinfer.mla.experimental.partition_runtime import H200PartitionRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "compact"), required=True)
    parser.add_argument("--seqlen-q", type=int, choices=(1, 4, 128), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_name() == "NVIDIA H200"
    torch.manual_seed(42)
    with H200PartitionRuntime() as runtime:
        assert runtime.sm_counts_cpu.tolist() == [66, 66]
        scale = 1 / math.sqrt(512) if args.seqlen_q == 128 else None
        baseline = make_uniform_case(64, args.seqlen_q, 32768, sm_scale=scale)
        # Match the previous matrix's setup/allocation order, including its ablation.
        linear = SM90PartitionExperiment(baseline, runtime)
        compact = SM90PartitionExperiment(
            baseline, runtime, compact=True, schedule=linear.schedule
        )
        baseline.original()
        baseline.run()
        assert_result(baseline, baseline)
        audits = {}
        for name, case in (("schedule_only", linear), ("compact", compact)):
            case.run_checked()
            audits[name] = case.validate()
            assert_result(case, baseline)
            case.set_audit(False)
            case.run()
            assert_result(case, baseline)
        selected = baseline if args.mode == "baseline" else compact
        cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()
        for _ in range(3):
            triton_runtime.driver.active.clear_cache(cache)
            selected.attention()
        torch.cuda.synchronize()
        metadata = dict(
            mode=args.mode,
            batch_size=64,
            seqlen_q=args.seqlen_q,
            seqlen_k=32768,
            heads=128,
            latent_dim=512,
            rope_dim=64,
            dtype="bfloat16",
            page_size=64,
            causal=False,
            seed=42,
            scale=baseline._args[13],
            device=torch.cuda.get_device_name(),
            plan=baseline.plan_summary(),
            audit=False,
            audits=audits,
            partition_sm_counts=runtime.sm_counts_cpu.tolist(),
            sm_partition=runtime.sm_partition_cpu.tolist(),
            sm_rank=runtime.sm_rank_cpu.tolist(),
            arena_bytes=runtime.arena_bytes,
            hash_base=runtime.hash_base,
            hash_mask=runtime.mask,
            compact_physical_span=2 * compact.layout.logical_span,
            warmup_launches=3,
            profiled_launches=0 if args.validate_only else 1,
            l2_eviction_bytes=cache.numel() * cache.element_size(),
            correctness="audit/no-audit output and LSE bitwise equal to fused",
        )
        print("TARGET_METADATA " + json.dumps(metadata), flush=True)
        if not args.validate_only:
            triton_runtime.driver.active.clear_cache(cache)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            selected.attention()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        selected.merge()
        assert_result(selected, baseline)
        print("POST_PROFILE_CORRECTNESS bitwise equal output and LSE", flush=True)


if __name__ == "__main__":
    main()
