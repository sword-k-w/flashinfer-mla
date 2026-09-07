# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Rectangular H200 MLA matrix, cold-L2 attention-kernel-only timings.

Uses the B200 B/Sq axes, random BF16 data, ABBA order, warmup/repeat budgets.
CUDA events are captured immediately around each single attention kernel so
Python/FFI submission and L2 eviction are outside the measured interval.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from triton import runtime as triton_runtime

from benchmarks.sm90_mla_partition import SM90PartitionExperiment
from benchmarks.sm90_mla_separate_merge import make_uniform_case
from flashinfer.mla.experimental.partition_runtime import H200PartitionRuntime

MODES = ("baseline", "schedule_only", "compact")
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


class TimedAttention:
    """Reusable small graph batches with an event pair around each kernel."""

    def __init__(self, fn, cache, warmup_ms, repeat_ms, min_samples):
        self.warmup_ms, self.repeat_ms, self.min_samples = (
            warmup_ms,
            repeat_ms,
            min_samples,
        )
        self.fn = fn
        fn()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.first, self.last = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
        ]
        with torch.cuda.graph(self.graph):
            self.first.record()
            triton_runtime.driver.active.clear_cache(cache)
            fn()
            self.last.record()
        estimates = []
        for _ in range(3):
            self.graph.replay()
            self.last.synchronize()
            estimates.append(self.first.elapsed_time(self.last))
        self.estimate_ms = statistics.median(estimates)
        # Limit each graph replay to roughly 8 ms, except a single long kernel.
        self.batch = max(1, min(128, math.ceil(8 / self.estimate_ms)))
        self.starts = [
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(self.batch)
        ]
        self.ends = [
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(self.batch)
        ]
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for start, end in zip(self.starts, self.ends, strict=True):
                triton_runtime.driver.active.clear_cache(cache)
                start.record()
                fn()
                end.record()
        self.warm_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.warm_graph):
            for _ in range(self.batch):
                fn()

    def measure(self):
        # Same budget-to-iteration conversion as the existing B200 do_bench:
        # estimate includes eviction, warmup invokes attention without eviction.
        warmups = max(1, int(self.warmup_ms / self.estimate_ms))
        for _ in range(math.ceil(warmups / self.batch)):
            self.warm_graph.replay()
        torch.cuda.synchronize()
        repeats = max(self.min_samples, math.ceil(self.repeat_ms / self.estimate_ms))
        samples = []
        while len(samples) < repeats:
            self.graph.replay()
            self.ends[-1].synchronize()
            values = [
                s.elapsed_time(e) for s, e in zip(self.starts, self.ends, strict=True)
            ]
            samples.extend(values[: repeats - len(samples)])
        return dict(
            median_ms=statistics.median(samples),
            min_ms=min(samples),
            max_ms=max(samples),
            p20_ms=quantile(samples, 0.2),
            p80_ms=quantile(samples, 0.8),
            sample_count=len(samples),
            estimate_with_eviction_ms=self.estimate_ms,
            graph_batch=self.batch,
            requested_repeat_ms=self.repeat_ms,
            effective_attention_ms=sum(samples),
            warmup_iterations=math.ceil(warmups / self.batch) * self.batch,
        )


def assert_equal_chunked(actual, expected):
    # Avoid multi-GiB comparison temporaries at the capacity point.
    if actual is None:
        assert expected is None
        return
    a, b = actual.reshape(-1), expected.reshape(-1)
    for offset in range(0, a.numel(), 1 << 20):
        torch.testing.assert_close(
            a[offset : offset + (1 << 20)],
            b[offset : offset + (1 << 20)],
            rtol=0,
            atol=0,
        )


def assert_result(case, baseline):
    assert_equal_chunked(case.out, baseline.original_out)
    assert_equal_chunked(case.lse, baseline.original_lse)


def trace_calls(calls, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for mode, fn in calls.items():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            fn()
            torch.cuda.synchronize()
        path = directory / f"{mode}.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        memory = [
            e["name"] for e in events if e.get("cat") in ("gpu_memset", "gpu_memcpy")
        ]
        assert len(kernels) == 1 and "HopperMergeKernel" not in kernels[0], kernels
        assert not memory, memory
        result[mode] = dict(kernels=kernels, memory_operations=memory)
    write_json(directory / "summary.json", result)


def run_case(runtime, batch, sq, sk, args, trace=False):
    torch.manual_seed(42)
    # B200 prefill used 1/sqrt(512); decode used the DeepSeek effective scale.
    scale = 1 / math.sqrt(512) if sq == 128 else None
    before = time.monotonic()
    baseline = make_uniform_case(batch, sq, sk, sm_scale=scale)
    linear = SM90PartitionExperiment(baseline, runtime)
    compact = SM90PartitionExperiment(
        baseline, runtime, compact=True, schedule=linear.schedule
    )
    setup_s = time.monotonic() - before
    baseline.original()
    baseline.run()
    assert_equal_chunked(baseline.out, baseline.original_out)
    assert_equal_chunked(baseline.lse, baseline.original_lse)
    audits = {}
    for mode, case in (("schedule_only", linear), ("compact", compact)):
        case.run_checked()
        audits[mode] = case.validate()
        assert_result(case, baseline)
        case.set_audit(False)
        case.run()
        assert_result(case, baseline)
    calls = dict(
        baseline=baseline.attention,
        schedule_only=linear.attention,
        compact=compact.attention,
    )
    rec = dict(
        batch=batch,
        sq=sq,
        sk=sk,
        status="passed",
        kv_bytes=batch * sk * 1152,
        setup_seconds=setup_s,
        compact_physical_span=2 * compact.layout.logical_span,
        cuda_free_bytes=torch.cuda.mem_get_info()[0],
        plan=baseline.plan_summary(),
        audits=audits,
        correctness="all variants bitwise equal output/LSE; audit and no-audit checked",
        scale=baseline._args[13],
    )
    if args.validate_only:
        return rec
    if trace:
        trace_calls(calls, args.output.parent / "attention_traces")
    for i in range(args.paired_warmups):
        for mode in MODES if i % 2 == 0 else MODES[::-1]:
            calls[mode]()
    torch.cuda.synchronize()
    cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    timers = {
        mode: TimedAttention(
            fn, cache, args.warmup_ms, args.repeat_ms, args.min_samples
        )
        for mode, fn in calls.items()
    }
    rec["l2_eviction_bytes"] = cache.numel() * cache.element_size()
    blocks = []
    for index in range(args.blocks):
        order = MODES if index % 4 in (0, 3) else MODES[::-1]
        block = dict(index=index, order=list(order))
        for mode in order:
            block[mode] = timers[mode].measure()
        blocks.append(block)
        print(f"BLOCK B={batch} Sq={sq} Sk={sk} {index + 1}/{args.blocks}", flush=True)
    rec["blocks"] = blocks
    rec["median_ms"] = {
        mode: statistics.median(b[mode]["median_ms"] for b in blocks) for mode in MODES
    }
    t = rec["median_ms"]
    rec["speedup"] = dict(
        compact_vs_baseline=t["baseline"] / t["compact"],
        schedule_vs_baseline=t["baseline"] / t["schedule_only"],
        compact_vs_schedule=t["schedule_only"] / t["compact"],
    )
    # Merge is outside every timed interval; verify latest partials after timing.
    baseline.separate_module.run(*baseline._args, 2)
    assert_equal_chunked(baseline.out, baseline.original_out)
    for case in (linear, compact):
        case.merge()
        assert_result(case, baseline)
    return rec


def capacity(runtime, batches, sqs):
    free, total = torch.cuda.mem_get_info()
    b, sq = max(batches), max(sqs)
    query = b * sq * 128 * 576 * 2
    outputs = 4 * b * sq * 128 * 512 * 2
    lse = 4 * b * sq * 128 * 4
    # Four 128 MiB workspaces, integer workspace, flush buffer, and launch metadata.
    fixed = query + outputs + lse + (4 * 128 + 8 + 256 + 64) * (1 << 20)
    reserve = 4 << 30
    arena_limit = runtime.arena_bytes // (b * 1152) // 64 * 64
    memory_limit = (free - fixed - reserve) // (b * 1152) // 64 * 64
    maximum = min(arena_limit, memory_limit)
    if maximum < 512:
        raise RuntimeError("Insufficient memory for the requested rectangle")
    return dict(
        max_seqlen_k=maximum,
        arena_limit=arena_limit,
        hbm_limit=memory_limit,
        max_batch=b,
        max_sq=sq,
        free_after_runtime=free,
        device_total_bytes=total,
        estimated_fixed_bytes=fixed,
        reserve_bytes=reserve,
        arena_bytes=runtime.arena_bytes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--seqlen-qs", type=int, nargs="+", default=[1, 4, 128])
    parser.add_argument("--seqlen-ks", type=int, nargs="+")
    parser.add_argument("--max-seqlen-k", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-warmups", type=int, default=20)
    parser.add_argument("--warmup-ms", type=float, default=500)
    parser.add_argument("--repeat-ms", type=float, default=1000)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if (
        args.blocks <= 0
        or args.blocks % 4
        or args.min_samples < 1
        or args.warmup_ms < 0
        or args.repeat_ms <= 0
    ):
        parser.error(
            "Invalid timing settings; blocks must be a positive multiple of four"
        )
    if any(b < 2 for b in args.batch_sizes) or any(
        sq not in (1, 4, 128) for sq in args.seqlen_qs
    ):
        parser.error("Use B>=2 and Sq=1/4/128")
    with H200PartitionRuntime() as runtime:
        cap = capacity(runtime, args.batch_sizes, args.seqlen_qs)
        maximum = args.max_seqlen_k or cap["max_seqlen_k"]
        if maximum > cap["max_seqlen_k"] or maximum % 64:
            raise ValueError(
                "Requested maximum exceeds safe rectangular capacity or page alignment"
            )
        axis = args.seqlen_ks or [2**p for p in range(9, 31) if 2**p < maximum] + [
            maximum
        ]
        if any(k < 512 or k > maximum or k % 64 for k in axis) or len(set(axis)) != len(
            axis
        ):
            raise ValueError("Invalid common KV length axis")
        timing = dict(
            method="paired-cold-l2-captured-events-v1",
            paired_warmups=args.paired_warmups,
            warmup_ms=args.warmup_ms,
            repeat_ms=args.repeat_ms,
            blocks=args.blocks,
            min_samples=args.min_samples,
            audit=False,
            measured="one attention kernel between captured CUDA events; eviction outside",
            order=[list(MODES), list(MODES[::-1]), list(MODES[::-1]), list(MODES)],
        )
        source_paths = [
            "benchmarks/bench_sm90_mla_partition.py",
            "benchmarks/sm90_mla_partition.py",
            "include/flashinfer/attention/mla_hopper.cuh",
            "include/flashinfer/attention/mla_partition.cuh",
            "include/flashinfer/partition/address.cuh",
            "csrc/batch_mla_sm90_run.cu",
        ]
        source_hashes = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_paths
        }
        document = dict(
            source_hashes=source_hashes,
            status="running",
            batch_sizes=args.batch_sizes,
            seqlen_qs=args.seqlen_qs,
            seqlen_ks=axis,
            capacity=cap,
            timing=timing,
            configuration=dict(
                dtype="bfloat16",
                heads=128,
                ckv_dim=512,
                kpe_dim=64,
                page=64,
                causal=False,
                data="random",
                seed=42,
                identical_kv_across_variants=True,
                shape="rectangular",
            ),
            environment=dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                sm_counts=runtime.sm_counts_cpu.tolist(),
            ),
            segments=[],
            results=[],
        )
        if args.resume and args.output.exists():
            old = json.loads(args.output.read_text())
            for key in (
                "batch_sizes",
                "seqlen_qs",
                "seqlen_ks",
                "timing",
                "configuration",
                "source_hashes",
            ):
                if old[key] != document[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            document = old
            document["status"] = "running"
        segment = len(document["segments"])
        document["segments"].append(
            dict(
                hash_base=runtime.hash_base,
                sm_partition=runtime.sm_partition_cpu.tolist(),
                sm_rank=runtime.sm_rank_cpu.tolist(),
                start_time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            )
        )
        completed = {
            (r["batch"], r["sq"], r["sk"])
            for r in document["results"]
            if r["status"] == "passed"
        }
        write_json(args.output, document)
        print("CAPACITY", json.dumps(cap), "AXIS", axis, flush=True)
        total = len(axis) * len(args.batch_sizes) * len(args.seqlen_qs)
        for sq in args.seqlen_qs:
            for b in args.batch_sizes:
                for sk in axis:
                    if (b, sq, sk) in completed:
                        continue
                    started = time.monotonic()
                    print(
                        f"START {len(completed) + 1}/{total} B={b} Sq={sq} Sk={sk}",
                        flush=True,
                    )
                    try:
                        rec = run_case(
                            runtime, b, sq, sk, args, trace=args.trace and not completed
                        )
                    except Exception as e:
                        document["status"] = "failed"
                        document["failure"] = dict(batch=b, sq=sq, sk=sk, error=str(e))
                        write_json(args.output, document)
                        raise
                    rec.update(segment=segment, wall_seconds=time.monotonic() - started)
                    document["results"].append(rec)
                    completed.add((b, sq, sk))
                    write_json(args.output, document)
                    print(
                        "RESULT",
                        json.dumps(
                            {
                                k: rec[k]
                                for k in (
                                    "batch",
                                    "sq",
                                    "sk",
                                    "status",
                                    "wall_seconds",
                                    "median_ms",
                                    "speedup",
                                )
                                if k in rec
                            }
                        ),
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
        document["status"] = "passed"
        document.pop("failure", None)
        document["completed_configurations"] = len(completed)
        write_json(args.output, document)


if __name__ == "__main__":
    main()
