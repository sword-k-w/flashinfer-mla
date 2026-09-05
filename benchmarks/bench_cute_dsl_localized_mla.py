#!/usr/bin/env python3
"""Benchmark standard versus localized partition-aware modular MLA decode."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from triton import runtime as triton_runtime
from triton.testing import do_bench

from localized_mla_benchmark import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQLEN_KS,
    HEADS,
    LATENT_DIM,
    PAGE_SIZE,
    ROPE_DIM,
    PreparedMLACase,
    compiled_kernel,
    kv_bytes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TIMING_POLICY = "paired-cold-l2-v2"
INTERLEAVED_TIMING_POLICY = "paired-cold-l2-interleaved-events-v1"
BALANCED_BLOCK_ORDER = (
    ("standard", "localized"),
    ("localized", "standard"),
    ("localized", "standard"),
    ("standard", "localized"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-partition-sm-counts", type=int, nargs=2)
    parser.add_argument("--seqlen-q", type=int, choices=range(1, 5), default=1)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--seqlen-ks", type=int, nargs="+", default=list(DEFAULT_SEQLEN_KS)
    )
    parser.add_argument("--paired-warmups", type=int, default=20)
    parser.add_argument("--timing-warmup-ms", type=int, default=25)
    parser.add_argument("--timing-repeat-ms", type=int, default=100)
    parser.add_argument("--timing-blocks", type=int, default=4)
    parser.add_argument("--timing-min-samples", type=int, default=20)
    parser.add_argument(
        "--timing-method",
        choices=("blocked-do-bench", "interleaved-events"),
        default="blocked-do-bench",
        help=(
            "Use the historical per-mode Triton blocks or alternate standard and "
            "localized CUDA-event samples to control power/clock drift."
        ),
    )
    parser.add_argument(
        "--data-initialization",
        choices=("empty", "random"),
        default="empty",
        help=(
            "Initialize query/KV with deterministic random BF16 values or retain "
            "the historical uninitialized access-pattern benchmark."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.batch_sizes = [2, 4]
        args.seqlen_ks = [512, 1024]
        args.paired_warmups = 2
        args.timing_warmup_ms = 5
        args.timing_repeat_ms = 10
        args.timing_min_samples = 5
    if any(batch < 2 for batch in args.batch_sizes):
        parser.error("all batch sizes must be >= 2")
    if any(seqlen <= 0 or seqlen % PAGE_SIZE for seqlen in args.seqlen_ks):
        parser.error(f"all seqlen_k values must be positive multiples of {PAGE_SIZE}")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch sizes must not contain duplicates")
    if len(set(args.seqlen_ks)) != len(args.seqlen_ks):
        parser.error("seqlen_k values must not contain duplicates")
    if args.timing_blocks <= 0 or args.timing_blocks % 4:
        parser.error("--timing-blocks must be a positive multiple of four")
    if args.output is None:
        report_name = (
            "localized_mla_capacity_matrix"
            if args.seqlen_q == 1
            else f"localized_mla_sq{args.seqlen_q}_capacity_matrix"
        )
        args.output = (
            SCRIPT_DIR.parent
            / "reports"
            / report_name
            / "iter_000_evaluation"
            / "post_flops.json"
        )
    return args


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_cold_l2_block(fn, args: argparse.Namespace) -> dict:
    effective_repeat_ms = args.timing_repeat_ms
    attempted_sample_counts = []
    while True:
        samples = [
            float(value)
            for value in do_bench(
                fn,
                warmup=args.timing_warmup_ms,
                rep=effective_repeat_ms,
                return_mode="all",
            )
        ]
        attempted_sample_counts.append(len(samples))
        if len(samples) >= args.timing_min_samples:
            break
        scale = args.timing_min_samples / max(1, len(samples))
        effective_repeat_ms = max(
            effective_repeat_ms + 1,
            math.ceil(effective_repeat_ms * scale * 1.10),
        )
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p20_ms": percentile(samples, 0.20),
        "p80_ms": percentile(samples, 0.80),
        "max_ms": max(samples),
        "sample_count": len(samples),
        "requested_repeat_ms": args.timing_repeat_ms,
        "effective_repeat_ms": effective_repeat_ms,
        "attempted_sample_counts": attempted_sample_counts,
    }


def benchmark_cold_l2_interleaved_block(
    calls: dict,
    base_order: tuple[str, str],
    args: argparse.Namespace,
    cache,
) -> dict:
    device_interface = triton_runtime.driver.active.get_device_interface()

    estimate_samples = {"standard": [], "localized": []}
    for sample_index in range(5):
        order = base_order if sample_index % 2 == 0 else base_order[::-1]
        for mode in order:
            triton_runtime.driver.active.clear_cache(cache)
            start = device_interface.Event(enable_timing=True)
            end = device_interface.Event(enable_timing=True)
            start.record()
            calls[mode]()
            end.record()
            estimate_samples[mode].append((start, end))
    device_interface.synchronize()
    estimates = {
        mode: statistics.median(start.elapsed_time(end) for start, end in events)
        for mode, events in estimate_samples.items()
    }
    pair_estimate_ms = estimates["standard"] + estimates["localized"]
    n_warmup = max(1, math.ceil(args.timing_warmup_ms / pair_estimate_ms))
    n_repeat = max(
        args.timing_min_samples,
        math.ceil(args.timing_repeat_ms / min(estimates.values())),
    )

    for sample_index in range(n_warmup):
        order = base_order if sample_index % 2 == 0 else base_order[::-1]
        for mode in order:
            calls[mode]()
    device_interface.synchronize()

    start_events = {
        mode: [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
        for mode in calls
    }
    end_events = {
        mode: [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
        for mode in calls
    }
    for sample_index in range(n_repeat):
        order = base_order if sample_index % 2 == 0 else base_order[::-1]
        for mode in order:
            triton_runtime.driver.active.clear_cache(cache)
            start_events[mode][sample_index].record()
            calls[mode]()
            end_events[mode][sample_index].record()
    device_interface.synchronize()

    results = {"sampling_order": "per-sample AB/BA alternation"}
    for mode in calls:
        samples = [
            start.elapsed_time(end)
            for start, end in zip(start_events[mode], end_events[mode], strict=True)
        ]
        results[mode] = {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "p20_ms": percentile(samples, 0.20),
            "p80_ms": percentile(samples, 0.80),
            "max_ms": max(samples),
            "sample_count": len(samples),
            "requested_repeat_ms": args.timing_repeat_ms,
            "effective_repeat_ms": sum(samples),
            "attempted_sample_counts": [len(samples)],
        }
    return results


def paired_timing(case: PreparedMLACase, args: argparse.Namespace) -> dict:
    calls = {"standard": case.standard_call, "localized": case.localized_call}
    for warmup_index in range(args.paired_warmups):
        order = (
            ("standard", "localized")
            if warmup_index % 2 == 0
            else ("localized", "standard")
        )
        for mode in order:
            calls[mode]()
    torch.cuda.synchronize(case.device)

    blocks = []
    mode_medians = {"standard": [], "localized": []}
    orders = list(BALANCED_BLOCK_ORDER) * (args.timing_blocks // 4)
    cache = (
        triton_runtime.driver.active.get_empty_cache_for_benchmark()
        if args.timing_method == "interleaved-events"
        else None
    )
    for block_index, order in enumerate(orders):
        block = {"block": block_index, "order": list(order)}
        if args.timing_method == "interleaved-events":
            measurements = benchmark_cold_l2_interleaved_block(
                calls, order, args, cache
            )
            block.update(measurements)
        else:
            for mode in order:
                block[mode] = benchmark_cold_l2_block(calls[mode], args)
        for mode in order:
            mode_medians[mode].append(block[mode]["median_ms"])
        block["paired_speedup"] = (
            block["standard"]["median_ms"] / block["localized"]["median_ms"]
        )
        blocks.append(block)

    standard_ms = statistics.median(mode_medians["standard"])
    localized_ms = statistics.median(mode_medians["localized"])
    return {
        "standard_ms": standard_ms,
        "localized_ms": localized_ms,
        "speedup": standard_ms / localized_ms,
        "median_paired_speedup": statistics.median(
            block["paired_speedup"] for block in blocks
        ),
        "standard_block_medians_ms": mode_medians["standard"],
        "localized_block_medians_ms": mode_medians["localized"],
        "blocks": blocks,
    }


def write_json_atomic(path: Path, document: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def make_document(args: argparse.Namespace, device: torch.device) -> dict:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    maximum_shape_kv_bytes = kv_bytes(max(args.batch_sizes), max(args.seqlen_ks))
    capacity_axis_source = (
        "vllm-fa B300 128-GiB localized arena capacity boundary"
        if tuple(args.seqlen_ks) == DEFAULT_SEQLEN_KS
        else "explicit command-line --seqlen-ks axis"
    )
    return {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "timing_policy": (
            INTERLEAVED_TIMING_POLICY
            if args.timing_method == "interleaved-events"
            else TIMING_POLICY
        ),
        "comparison": {
            "standard": "modular MLA static persistent scheduler + ordinary cudaMalloc KV",
            "localized": (
                "split-granular partition-aware static persistent scheduler + "
                "two RM-localized KV pools"
            ),
        },
        "batch_sizes": args.batch_sizes,
        "seqlen_ks": args.seqlen_ks,
        "configuration": {
            "workload": "decode",
            "seqlen_q": args.seqlen_q,
            "num_heads": HEADS,
            "latent_dim": LATENT_DIM,
            "rope_dim": ROPE_DIM,
            "page_size": PAGE_SIZE,
            "dtype": "bfloat16",
            "data_initialization": args.data_initialization,
            "data_seed": 42 if args.data_initialization == "random" else None,
            "identical_kv_values_between_layouts": (
                args.data_initialization == "random"
            ),
            "fixed_page_table": True,
            "enable_pdl": False,
            "maximum_shape_one_kv_bytes": maximum_shape_kv_bytes,
            "maximum_shape_two_kv_gib": 2 * maximum_shape_kv_bytes / 2**30,
            "capacity_axis_source": capacity_axis_source,
        },
        "timing": {
            "method": args.timing_method,
            "paired_warmups": args.paired_warmups,
            "warmup_ms": args.timing_warmup_ms,
            "repeat_ms": args.timing_repeat_ms,
            "blocks": args.timing_blocks,
            "min_samples": args.timing_min_samples,
            "balanced_block_order": [list(order) for order in BALANCED_BLOCK_ORDER],
            "l2_policy": "triton.testing.do_bench clears L2 before each sample",
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": properties.name,
            "sm_count": properties.multi_processor_count,
            "total_hbm_bytes": total_bytes,
            "free_hbm_bytes_at_start": free_bytes,
        },
        "results": [],
    }


def load_or_make_document(args: argparse.Namespace, device: torch.device) -> dict:
    if args.resume and args.output.is_file():
        document = json.loads(args.output.read_text())
        if document["batch_sizes"] != args.batch_sizes:
            raise ValueError("resume file batch axis differs from command line")
        if document["seqlen_ks"] != args.seqlen_ks:
            raise ValueError("resume file seqlen axis differs from command line")
        if document["configuration"]["seqlen_q"] != args.seqlen_q:
            raise ValueError("resume file seqlen_q differs from command line")
        if document["configuration"].get("data_initialization", "empty") != (
            args.data_initialization
        ):
            raise ValueError(
                "resume file data initialization differs from command line"
            )
        document["status"] = "running"
        document.pop("error", None)
        document.pop("finished_at", None)
        return document
    return make_document(args, device)


def run_case(
    batch_size: int,
    seqlen_k: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    started = time.monotonic()
    case = PreparedMLACase(
        batch_size,
        seqlen_k,
        device=device,
        seq_len_q=args.seqlen_q,
        initialize_for_correctness=args.data_initialization == "random",
    )
    try:
        geometry = case.scheduler_geometry()
        observed_sm_counts = [2 * n for n in geometry.resident_partition_clusters]
        if (
            args.expected_partition_sm_counts is not None
            and observed_sm_counts != args.expected_partition_sm_counts
        ):
            raise RuntimeError(
                f"partition SM counts {observed_sm_counts} != expected "
                f"{args.expected_partition_sm_counts}"
            )
        mapped_bytes = tuple(case.localized_cache.mapped_bytes)
        free_after_alloc, _ = torch.cuda.mem_get_info(device)
        timing = paired_timing(case, args)
        torch.cuda.synchronize(device)
        return {
            "batch_size": batch_size,
            "seqlen_q": args.seqlen_q,
            "seqlen_k": seqlen_k,
            "kv_bytes_per_layout": kv_bytes(batch_size, seqlen_k),
            "kv_gib_per_layout": kv_bytes(batch_size, seqlen_k) / 2**30,
            "localized_mapped_bytes": list(mapped_bytes),
            "localized_mapped_gib": sum(mapped_bytes) / 2**30,
            "split_kv": geometry.split_kv,
            "owner_work_counts": list(geometry.owner_work_counts),
            "owner_tile_counts": list(geometry.owner_tile_counts),
            "owner_page_counts": list(case.localized_cache.owner_page_counts),
            "resident_partition_clusters": list(geometry.resident_partition_clusters),
            "standard_theoretical_active_clusters": geometry.standard_active_clusters,
            "localized_theoretical_active_clusters": geometry.localized_active_clusters,
            "localized_theoretical_active_cluster_fraction": (
                geometry.localized_active_fraction
            ),
            "free_hbm_bytes_before": free_before,
            "free_hbm_bytes_after_alloc": free_after_alloc,
            "total_hbm_bytes": total_bytes,
            "wall_seconds": time.monotonic() - started,
            **timing,
        }
    finally:
        case.close()
        del case
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    document = load_or_make_document(args, device)
    args.output = args.output.expanduser().resolve()
    if args.plan_only:
        document["status"] = "planned"
        write_json_atomic(args.output, document)
        print(json.dumps(document["configuration"], indent=2))
        print(args.output)
        return

    # Compile both workspace variants before allocating capacity-scale KV.
    for workspace_size_zero in (False, True):
        compiled_kernel(partition_aware=False, workspace_size_zero=workspace_size_zero)
        compiled_kernel(partition_aware=True, workspace_size_zero=workspace_size_zero)
    torch.cuda.synchronize(device)

    completed = {(row["batch_size"], row["seqlen_k"]) for row in document["results"]}
    try:
        for seqlen_k in args.seqlen_ks:
            for batch_size in args.batch_sizes:
                if (batch_size, seqlen_k) in completed:
                    continue
                print(
                    f"START B={batch_size} Sq={args.seqlen_q} Sk={seqlen_k:,}",
                    flush=True,
                )
                row = run_case(batch_size, seqlen_k, device, args)
                document["results"].append(row)
                write_json_atomic(args.output, document)
                print(
                    f"DONE  B={batch_size} Sk={seqlen_k:,} "
                    f"standard={row['standard_ms']:.6f} ms "
                    f"localized={row['localized_ms']:.6f} ms "
                    f"speedup={row['speedup']:.5f}x",
                    flush=True,
                )
    except BaseException as error:
        document["status"] = "failed"
        document["error"] = f"{type(error).__name__}: {error}"
        document["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(args.output, document)
        raise
    document["status"] = "passed"
    document["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(args.output, document)
    print(args.output)


if __name__ == "__main__":
    main()
