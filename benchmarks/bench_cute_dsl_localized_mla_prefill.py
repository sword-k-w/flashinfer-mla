#!/usr/bin/env python3
"""Benchmark standard versus partition-localized monolithic MLA prefill."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from bench_cute_dsl_localized_mla import (
    BALANCED_BLOCK_ORDER,
    TIMING_POLICY,
    paired_timing,
)
from localized_mla_prefill_benchmark import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQLEN_KS,
    DEFAULT_SEQLEN_QS,
    HEADS,
    LATENT_DIM,
    PAGE_SIZE,
    ROPE_DIM,
    PreparedMLAPrefillCase,
    compiled_kernel,
    fixed_tensor_bytes,
    kv_bytes,
)
from flashinfer.cute_dsl.attention.experimental.localized_mla_prefill import (
    LocalizedMLAPrefillKVCache,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent
    / "reports"
    / "localized_mla_prefill_matrix"
    / "iter_000_evaluation"
    / "timing.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-partition-sm-counts", type=int, nargs=2)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--seqlen-qs", type=int, nargs="+", default=list(DEFAULT_SEQLEN_QS)
    )
    parser.add_argument(
        "--seqlen-ks", type=int, nargs="+", default=list(DEFAULT_SEQLEN_KS)
    )
    parser.add_argument(
        "--capacity-point",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a page-aligned Sk that targets the requested HBM utilization.",
    )
    parser.add_argument(
        "--hbm-utilization",
        type=float,
        default=0.90,
        help="Fraction of free HBM usable by both KV layouts and fixed buffers.",
    )
    parser.add_argument("--paired-warmups", type=int, default=20)
    parser.add_argument("--timing-warmup-ms", type=int, default=25)
    parser.add_argument("--timing-repeat-ms", type=int, default=100)
    parser.add_argument("--timing-blocks", type=int, default=4)
    parser.add_argument("--timing-min-samples", type=int, default=20)
    parser.add_argument(
        "--timing-method", choices=("blocked-do-bench",), default="blocked-do-bench"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.batch_sizes = [2, 4]
        args.seqlen_qs = [2, 4]
        args.seqlen_ks = [512, 4096]
        args.capacity_point = False
        args.paired_warmups = 2
        args.timing_warmup_ms = 5
        args.timing_repeat_ms = 10
        args.timing_min_samples = 5
    if any(batch < 2 for batch in args.batch_sizes):
        parser.error("all batch sizes must be >= 2")
    if any(seqlen_q <= 0 for seqlen_q in args.seqlen_qs):
        parser.error("all seqlen_q values must be positive")
    if any(seqlen_k <= 0 or seqlen_k % PAGE_SIZE for seqlen_k in args.seqlen_ks):
        parser.error(f"all seqlen_k values must be positive multiples of {PAGE_SIZE}")
    for name in ("batch_sizes", "seqlen_qs", "seqlen_ks"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if args.timing_blocks <= 0 or args.timing_blocks % 4:
        parser.error("--timing-blocks must be a positive multiple of four")
    if not 0.5 <= args.hbm_utilization <= 0.95:
        parser.error("--hbm-utilization must be between 0.5 and 0.95")
    return args


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


def write_csv(path: Path, rows: list[dict]) -> Path:
    csv_path = path.with_suffix(".csv")
    fields = (
        "batch_size",
        "seqlen_q",
        "seqlen_k",
        "standard_ms",
        "localized_ms",
        "speedup",
        "kv_gib_per_layout",
        "owner_batch_counts",
        "owner_tile_counts",
        "owner_wave_counts",
        "localized_theoretical_active_cluster_fraction",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return csv_path


def choose_capacity_seqlen_k(
    args: argparse.Namespace, device: torch.device
) -> tuple[int, dict]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    max_batch = max(args.batch_sizes)
    max_seqlen_q = max(args.seqlen_qs)
    fixed_bytes = fixed_tensor_bytes(max_batch, max_seqlen_q)
    reserve_bytes = int(free_bytes * (1.0 - args.hbm_utilization))
    usable_bytes = free_bytes - reserve_bytes - fixed_bytes
    bytes_per_k_token_for_pair = 2 * kv_bytes(max_batch, 1)
    upper_seqlen_k = usable_bytes // bytes_per_k_token_for_pair // PAGE_SIZE * PAGE_SIZE
    if upper_seqlen_k < PAGE_SIZE:
        raise RuntimeError("not enough free HBM for the requested benchmark matrix")

    # RM-localized allocations can consume more physical HBM than their mapped
    # byte count near the per-partition capacity boundary. Probe only the
    # localized half, then require enough reported free HBM for cudaMalloc KV,
    # all fixed tensors, and the requested reserve. This never attempts the
    # unsafe standard allocation while searching the boundary.
    probes = []
    lower = PAGE_SIZE
    upper = upper_seqlen_k
    best = None
    for _ in range(10):
        if lower > upper:
            break
        midpoint = ((lower + upper) // (2 * PAGE_SIZE)) * PAGE_SIZE
        midpoint = max(PAGE_SIZE, midpoint)
        cache = LocalizedMLAPrefillKVCache(
            max_batch,
            midpoint,
            seq_len_q=max_seqlen_q,
            page_size=PAGE_SIZE,
            device=device,
        )
        try:
            free_after_localized, _ = torch.cuda.mem_get_info(device)
            standard_kv_bytes = kv_bytes(max_batch, midpoint)
            projected_remaining = free_after_localized - standard_kv_bytes - fixed_bytes
            safe = projected_remaining >= reserve_bytes
            probe = {
                "seqlen_k": midpoint,
                "logical_localized_kv_bytes": standard_kv_bytes,
                "localized_mapped_bytes": sum(cache.mapped_bytes),
                "free_hbm_bytes_after_localized": free_after_localized,
                "projected_free_hbm_bytes_after_full_case": projected_remaining,
                "required_reserve_bytes": reserve_bytes,
                "safe": safe,
            }
            probes.append(probe)
        finally:
            cache.close()
            del cache
            torch.cuda.empty_cache()
        if safe:
            best = probe
            lower = midpoint + PAGE_SIZE
        else:
            upper = midpoint - PAGE_SIZE

    if best is None:
        raise RuntimeError("localized HBM calibration found no safe capacity point")
    seqlen_k = best["seqlen_k"]
    return seqlen_k, {
        "free_hbm_bytes_after_compile": free_bytes,
        "total_hbm_bytes": total_bytes,
        "target_hbm_utilization": args.hbm_utilization,
        "required_reserve_bytes": reserve_bytes,
        "fixed_tensor_bytes_at_max_shape": fixed_bytes,
        "capacity_batch_size": max_batch,
        "capacity_seqlen_q": max_seqlen_q,
        "capacity_seqlen_k": seqlen_k,
        "capacity_pair_kv_bytes": 2 * kv_bytes(max_batch, seqlen_k),
        "calibration_method": "localized-only allocation with projected full-case reserve",
        "selected_probe": best,
        "probes": probes,
    }


def make_document(
    args: argparse.Namespace,
    device: torch.device,
    seqlen_ks: list[int],
    capacity: dict | None,
) -> dict:
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "timing_policy": TIMING_POLICY,
        "comparison": {
            "standard": "monolithic BF16 MLA + ordinary cudaMalloc KV",
            "localized": (
                "monolithic BF16 MLA + batch-owner partition scheduler + "
                "two RM-localized KV pools"
            ),
        },
        "batch_sizes": args.batch_sizes,
        "seqlen_qs": args.seqlen_qs,
        "seqlen_ks": seqlen_ks,
        "configuration": {
            "workload": "prefill",
            "num_heads": HEADS,
            "latent_dim": LATENT_DIM,
            "rope_dim": ROPE_DIM,
            "page_size": PAGE_SIZE,
            "dtype": "bfloat16",
            "data_initialization": "torch.randn",
            "data_seed": 42,
            "split_kv": 1,
            "fixed_page_table": True,
            "enable_pdl": False,
            "capacity": capacity,
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
        },
        "results": [],
    }


def load_or_make_document(
    args: argparse.Namespace,
    device: torch.device,
    seqlen_ks: list[int],
    capacity: dict | None,
) -> dict:
    if args.resume and args.output.is_file():
        document = json.loads(args.output.read_text())
        for key, values in (
            ("batch_sizes", args.batch_sizes),
            ("seqlen_qs", args.seqlen_qs),
            ("seqlen_ks", seqlen_ks),
        ):
            if document[key] != values:
                raise ValueError(f"resume file {key} axis differs from command line")
        document["status"] = "running"
        document.pop("error", None)
        document.pop("finished_at", None)
        return document
    return make_document(args, device, seqlen_ks, capacity)


def run_case(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    started = time.monotonic()
    case = PreparedMLAPrefillCase(batch_size, seqlen_q, seqlen_k, device=device)
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
        correctness = case.check_exact()
        if not correctness["output_exact"] or not correctness["lse_exact"]:
            raise AssertionError(f"bitwise correctness failed: {correctness}")
        timing = paired_timing(case, args)
        torch.cuda.synchronize(device)
        return {
            "batch_size": batch_size,
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "kv_bytes_per_layout": kv_bytes(batch_size, seqlen_k),
            "kv_gib_per_layout": kv_bytes(batch_size, seqlen_k) / 2**30,
            "localized_mapped_bytes": list(mapped_bytes),
            "localized_mapped_gib": sum(mapped_bytes) / 2**30,
            "owner_batch_counts": list(geometry.owner_batch_counts),
            "owner_tile_counts": list(geometry.owner_tile_counts),
            "owner_wave_counts": list(geometry.owner_wave_counts),
            "owner_page_counts": list(case.localized_cache.owner_page_counts),
            "resident_partition_clusters": list(geometry.resident_partition_clusters),
            "standard_grid_clusters": geometry.standard_active_clusters,
            "localized_grid_clusters": geometry.total_resident_clusters,
            "standard_theoretical_active_clusters": geometry.standard_active_clusters,
            "localized_theoretical_active_clusters": (
                geometry.localized_active_clusters
            ),
            "localized_theoretical_active_cluster_fraction": (
                geometry.localized_active_fraction
            ),
            "correctness": correctness,
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
    args.output = args.output.expanduser().resolve()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    # Compile before capacity planning so compiler/context memory is already
    # reflected in the free-HBM reading used to choose the largest Sk.
    for seqlen_q in args.seqlen_qs:
        compiled_kernel(seq_len_q=seqlen_q, partition_aware=False)
        compiled_kernel(seq_len_q=seqlen_q, partition_aware=True)
    torch.cuda.synchronize(device)

    existing_document = None
    if args.resume and args.output.is_file():
        existing_document = json.loads(args.output.read_text())
        seqlen_ks = list(existing_document["seqlen_ks"])
        capacity = existing_document["configuration"].get("capacity")
    else:
        capacity = None
        seqlen_ks = list(args.seqlen_ks)
        if args.capacity_point:
            capacity_seqlen_k, capacity = choose_capacity_seqlen_k(args, device)
            if capacity_seqlen_k not in seqlen_ks:
                seqlen_ks.append(capacity_seqlen_k)
        seqlen_ks.sort()
    document = load_or_make_document(args, device, seqlen_ks, capacity)
    if args.plan_only:
        document["status"] = "planned"
        write_json_atomic(args.output, document)
        print(json.dumps(document["configuration"], indent=2))
        print(args.output)
        return

    completed = {
        (row["batch_size"], row["seqlen_q"], row["seqlen_k"])
        for row in document["results"]
    }
    try:
        for seqlen_q in args.seqlen_qs:
            for seqlen_k in seqlen_ks:
                # Allocate the capacity corner before the other capacity cases.
                # Large back-to-back RM allocations can otherwise fragment the
                # physical per-partition allocator within a long-lived process.
                batches = (
                    list(reversed(args.batch_sizes))
                    if capacity is not None
                    and seqlen_k == capacity["capacity_seqlen_k"]
                    else args.batch_sizes
                )
                for batch_size in batches:
                    shape = (batch_size, seqlen_q, seqlen_k)
                    if shape in completed:
                        continue
                    print(
                        f"START B={batch_size} Sq={seqlen_q} Sk={seqlen_k:,}",
                        flush=True,
                    )
                    row = run_case(batch_size, seqlen_q, seqlen_k, device, args)
                    document["results"].append(row)
                    write_json_atomic(args.output, document)
                    write_csv(args.output, document["results"])
                    print(
                        f"DONE  B={batch_size} Sq={seqlen_q} Sk={seqlen_k:,} "
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
        write_csv(args.output, document["results"])
        raise
    document["status"] = "passed"
    document["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(args.output, document)
    csv_path = write_csv(args.output, document["results"])
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
