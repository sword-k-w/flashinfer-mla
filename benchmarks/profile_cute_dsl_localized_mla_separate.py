#!/usr/bin/env python3
"""Collect one independent LTC-only or L2/HBM experiment on MLA boundary points.

Both experiments use random seed 42 and the established cold-L2 target protocol.
LTC requests exactly one metric; memory requests the original ten metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import profile_cute_dsl_localized_mla_boundary as boundary
import profile_cute_dsl_localized_mla_memory as memory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("ltc", "memory"), required=True)
    parser.add_argument("--timing-sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--expected-partition-sm-counts", type=int, nargs=2, default=[74, 74]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def collect(args, ncu, metrics, config, mode, directory):
    stem = directory / mode
    command = memory.profile_command(args, ncu, config, mode, stem)
    command[command.index("--metrics") + 1] = ",".join(metrics)
    command += [
        "--workload",
        config["workload"],
        "--expected-partition-sm-counts",
        *map(str, args.expected_partition_sm_counts),
    ]
    environment = os.environ.copy()
    for key, value in {
        "MAX_JOBS": "5",
        "FLASHINFER_NVCC_THREADS": "1",
        "OMP_NUM_THREADS": "5",
    }.items():
        environment.setdefault(key, value)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(memory.REPO_ROOT), environment.get("PYTHONPATH")])
    )
    memory.write_json_atomic(
        stem.with_name(f"{mode}_command.json"),
        {
            "argv": command,
            "environment": {
                key: environment[key]
                for key in (
                    "MAX_JOBS",
                    "FLASHINFER_NVCC_THREADS",
                    "OMP_NUM_THREADS",
                    "PYTHONPATH",
                )
            },
        },
    )
    started = time.monotonic()
    with stem.with_suffix(".log").open("w") as log:
        subprocess.run(
            command,
            cwd=memory.REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    metadata = memory.parse_target_metadata(stem.with_suffix(".log").read_text())
    memory.validate_target(metadata, config, mode)
    if (
        metadata["workload"] != config["workload"]
        or metadata["device"] != "NVIDIA B200"
    ):
        raise RuntimeError(f"unexpected workload/device: {metadata}")
    if (
        mode == "localized"
        and metadata["partition_sm_counts"] != args.expected_partition_sm_counts
    ):
        raise RuntimeError(f"unexpected SM split: {metadata}")
    memory.write_json_atomic(stem.with_name(f"{mode}_metadata.json"), metadata)
    profile = memory.parse_ncu_report(
        ncu, stem.with_suffix(".ncu-rep"), stem.with_suffix(".csv"), metrics=metrics
    )
    if profile["ncu_replay_passes"] < 1:
        raise RuntimeError("missing replay passes")
    if args.experiment == "memory":
        # Preserve raw anomalous readings explicitly; collection is not validity.
        memory.derive_metrics(profile, allow_inconsistent_l2=True)
    else:
        count = memory.metric_value(profile, boundary.LTC_METRIC)
        if count < 0:
            raise RuntimeError("negative LTC count")
        profile["ltc_fabric_requests"] = count
    profile.update(
        target_metadata=metadata,
        command=command,
        log=str(stem.with_suffix(".log")),
        wall_seconds=time.monotonic() - started,
    )
    return profile


def summarize(root, document):
    warnings, rows = [], []
    for config in document["configs"]:
        if config["status"] != "complete":
            continue
        row = {
            key: config[key]
            for key in ("workload", "seqlen_q", "batch_size", "seqlen_k")
        }
        for mode, profile in config["profiles"].items():
            row[f"{mode}_replay_passes"] = profile["ncu_replay_passes"]
            for metric, record in profile["metrics"].items():
                row[f"{mode}_{metric}"] = record["value"]
            if document["settings"]["experiment"] == "memory":
                l2 = profile["l2"]
                row[f"{mode}_l2_hit_rate_usable"] = l2["hit_rate_usable"]
                row[f"{mode}_l2_counter_sum_relative_error"] = l2[
                    "counter_sum_relative_error"
                ]
                row[f"{mode}_l2_warnings"] = "; ".join(l2["validation_warnings"])
                if not l2["hit_rate_usable"]:
                    warnings.append(
                        {
                            "workload": config["workload"],
                            "shape": memory.config_key(config),
                            "mode": mode,
                            "l2": l2,
                        }
                    )
        row.update(config["comparison"])
        rows.append(row)
    if rows:
        with (root / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    document["metric_quality_warnings"] = warnings
    document["measurement_validity"] = (
        "L2 warnings present" if warnings else "no automated quality flags"
    )
    document["quality_policy"] = (
        "Raw L2 rate outside [0,100] or abs((hits+misses)/total-1)>5% is flagged, never repaired. 5% is an experimental QA heuristic, not an NVIDIA specification."
    )


def main():
    args = parse_args()
    args.cache_control, args.clock_control = "all", "boost"
    root = args.output_root.resolve()
    summary_path = root / "matrix_summary.json"
    if args.summarize_only:
        document = json.loads(summary_path.read_text())
        if document["settings"]["experiment"] != args.experiment:
            raise ValueError("experiment differs")
        summarize(root, document)
        memory.write_json_atomic(summary_path, document)
        return
    configs, sources = boundary.load_boundary_configs(args.timing_sources)
    metrics = (
        (boundary.LTC_METRIC,) if args.experiment == "ltc" else memory.MEMORY_METRICS
    )
    settings = {
        "experiment": args.experiment,
        "metrics": list(metrics),
        "sources": sources,
        "data_initialization": "random",
        "seed": 42,
        "cold_l2": True,
        "warmup_launches": 3,
        "profiled_launches": 1,
        "cache_control": "all",
        "clock_control": "boost",
        "replay_mode": "kernel",
        "expected_partition_sm_counts": args.expected_partition_sm_counts,
        "device": args.device,
        "timing_values_used": False,
    }
    if args.dry_run:
        print(json.dumps({"settings": settings, "configs": configs}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    ncu = memory.resolve_ncu(args.ncu)
    memory.validate_metrics(ncu, args.device, metrics=metrics)
    scripts = {
        str(p.relative_to(memory.REPO_ROOT)): memory.file_sha256(p)
        for p in (
            Path(__file__).resolve(),
            memory.TARGET,
            Path(memory.__file__),
            Path(boundary.__file__),
        )
    }
    if args.resume:
        document = json.loads(summary_path.read_text())
        if document["settings"] != settings:
            raise ValueError("resume settings differ")
        document["environment"].setdefault("resume_scripts", []).append(
            {"time": memory.utc_now(), "scripts": scripts}
        )
    else:
        if summary_path.exists():
            raise FileExistsError("use --resume or a fresh output directory")
        document = {
            "schema": "flashinfer-localized-mla-separate-profile-v1",
            "created_at": memory.utc_now(),
            "settings": settings,
            "configs": configs,
            "environment": {
                "git_revision": memory.git_revision(),
                "ncu_version": memory.ncu_version(ncu),
                "gpu_initial": memory.query_gpu(args.device),
                "scripts": scripts,
            },
        }
    document["status"] = "running"
    memory.write_json_atomic(summary_path, document)
    for index, config in enumerate(document["configs"]):
        if config["status"] == "complete":
            continue
        directory = (
            root
            / "profiles"
            / config["workload"]
            / memory.config_key(config)
            / f"attempt_{len(config['attempts']) + 1:03d}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        order = (
            ["standard", "localized"] if index % 2 == 0 else ["localized", "standard"]
        )
        attempt = {
            "started_at": memory.utc_now(),
            "mode_order": order,
            "directory": str(directory.relative_to(root)),
            "status": "running",
        }
        config["attempts"].append(attempt)
        config["status"] = "running"
        memory.write_json_atomic(summary_path, document)
        print(
            f"[{index + 1}/{len(configs)}] {args.experiment} {config['workload']} {memory.config_key(config)}",
            flush=True,
        )
        try:
            profiles = {
                mode: collect(args, ncu, metrics, config, mode, directory)
                for mode in order
            }
            if args.experiment == "memory":
                comparison = memory.compare(profiles)
                if not all(p["l2"]["hit_rate_usable"] for p in profiles.values()):
                    comparison["localized_minus_standard_l2_hit_rate_pp"] = None
            else:
                standard = profiles["standard"]["ltc_fabric_requests"]
                localized = profiles["localized"]["ltc_fabric_requests"]
                comparison = {
                    "ltc_reduction_pct": 100 * (1 - localized / standard)
                    if standard
                    else None
                }
            config.update(profiles=profiles, comparison=comparison, status="complete")
            config.pop("error", None)
            attempt["status"] = "complete"
            print("  collected", flush=True)
        except BaseException as error:
            config["status"] = attempt["status"] = document["status"] = "failed"
            config["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            attempt["finished_at"] = memory.utc_now()
            summarize(root, document)
            memory.write_json_atomic(summary_path, document)
    document["status"] = "complete"
    document["finished_at"] = memory.utc_now()
    document["environment"]["gpu_final"] = memory.query_gpu(args.device)
    memory.write_json_atomic(summary_path, document)
    print(f"complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
