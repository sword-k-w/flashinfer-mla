#!/usr/bin/env python3
"""Profile sparse B/Sk boundary points for MLA decode and Sq=128 prefill with NCU.

Collect LTC fabric and L2/HBM counters together. Timing sources determine shapes
and split_kv only; their latencies are never treated as counter measurements.
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

import profile_cute_dsl_localized_mla_memory as memory


LTC_METRIC = "lts__t_requests_srcunit_ltcfabric.sum"
METRICS = (*memory.MEMORY_METRICS, LTC_METRIC)
DEFAULT_SOURCES = [
    *memory.DEFAULT_TIMING_SOURCES,
    memory.REPO_ROOT
    / "reports/localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing-sources", type=Path, nargs="+", default=DEFAULT_SOURCES
    )
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
    parser.add_argument("--max-new-configs", type=int)
    return parser.parse_args()


def load_boundary_configs(paths):
    configs, sources = [], []
    for path in paths:
        path = path.resolve()
        data = json.loads(path.read_text())
        if data.get("status") != "passed":
            raise ValueError(f"source is not complete: {path}")
        configuration = data["configuration"]
        if configuration["data_initialization"] not in ("random", "torch.randn"):
            raise ValueError(f"source requires random inputs: {path}")
        workload = configuration["workload"]
        rows = data["results"]
        sq_values = {row["seqlen_q"] for row in rows}
        if len(sq_values) != 1:
            raise ValueError(f"use one Sq per source: {path}")
        sq = next(iter(sq_values))
        if (workload, sq) not in (("decode", 1), ("decode", 4), ("prefill", 128)):
            raise ValueError(f"unsupported workload/Sq: {workload}/{sq}")
        max_batch = max(row["batch_size"] for row in rows)
        max_sk = max(row["seqlen_k"] for row in rows)
        # Four Sk points at maximum batch, plus two batches at maximum Sk.
        points = [(max_batch, sk) for sk in (512, 65536, 524288, max_sk)]
        points += [(batch, max_sk) for batch in (2, 16)]
        lookup = {(row["batch_size"], row["seqlen_k"]): row for row in rows}
        sources.append(
            {
                "path": str(path),
                "sha256": memory.file_sha256(path),
                "workload": workload,
                "seqlen_q": sq,
                "role": "shape_and_split_kv_manifest_only",
            }
        )
        for batch, sk in dict.fromkeys(points):
            row = lookup[(batch, sk)]
            configs.append(
                {
                    "workload": workload,
                    "seqlen_q": sq,
                    "batch_size": batch,
                    "seqlen_k": sk,
                    "split_kv": row.get("split_kv", configuration.get("split_kv")),
                    "source": str(path),
                    "status": "pending",
                    "attempts": [],
                }
            )
    keys = [(c["workload"], memory.config_key(c)) for c in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate configurations")
    return configs, sources


def profile_one(args, ncu, config, mode, directory):
    stem = directory / mode
    command = memory.profile_command(args, ncu, config, mode, stem)
    command[command.index("--metrics") + 1] = ",".join(METRICS)
    command += [
        "--workload",
        config["workload"],
        "--expected-partition-sm-counts",
        *map(str, args.expected_partition_sm_counts),
    ]
    environment = os.environ.copy()
    environment.setdefault("MAX_JOBS", "5")
    environment.setdefault("FLASHINFER_NVCC_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "5")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(memory.REPO_ROOT),
                environment.get("PYTHONPATH"),
            ),
        )
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
        completed = subprocess.run(
            command,
            cwd=memory.REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"NCU exited {completed.returncode}: {stem.with_suffix('.log')}"
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
        raise RuntimeError(f"unexpected partition: {metadata['partition_sm_counts']}")
    memory.write_json_atomic(stem.with_name(f"{mode}_metadata.json"), metadata)
    profile = memory.parse_ncu_report(
        ncu, stem.with_suffix(".ncu-rep"), stem.with_suffix(".csv"), metrics=METRICS
    )
    if profile["ncu_replay_passes"] < 1:
        raise RuntimeError("NCU did not record replay passes")
    memory.derive_metrics(profile, allow_inconsistent_l2=True)
    fabric = memory.metric_value(profile, LTC_METRIC)
    if fabric < 0:
        raise RuntimeError("negative LTC fabric request count")
    profile.update(
        {
            "target_metadata": metadata,
            "command": command,
            "log": str(stem.with_suffix(".log")),
            "wall_seconds": time.monotonic() - started,
            "ltc_fabric_requests": fabric,
        }
    )
    return profile


def refresh_quality(document):
    warnings = []
    for config in document["configs"]:
        if config["status"] != "complete":
            continue
        for mode, profile in config["profiles"].items():
            memory.derive_metrics(profile, allow_inconsistent_l2=True)
            if not profile["l2"]["hit_rate_usable"]:
                warnings.append(
                    {
                        "workload": config["workload"],
                        "shape": memory.config_key(config),
                        "mode": mode,
                        "warnings": profile["l2"]["validation_warnings"],
                        "raw_hit_rate_pct": profile["l2"]["hit_rate_pct"],
                        "counter_sum_relative_error": profile["l2"][
                            "counter_sum_relative_error"
                        ],
                    }
                )
        if not all(p["l2"]["hit_rate_usable"] for p in config["profiles"].values()):
            config["comparison"]["localized_minus_standard_l2_hit_rate_pp"] = None
    document["metric_quality_warnings"] = warnings
    document["metric_quality_policy"] = (
        "Retain raw counters without clamping; omit L2 hit-rate interpretation if "
        "outside [0,100]% or abs((hits+misses)/total-1)>5%. The 5% threshold is an "
        "experiment QA heuristic, not an NVIDIA accuracy specification. "
        "hit/(hit+miss) remains diagnostic and does not repair cross-pass disagreement."
    )


def summarize(root, document):
    refresh_quality(document)
    rows = []
    for config in document["configs"]:
        if config["status"] != "complete":
            continue
        row = {
            key: config[key]
            for key in ("workload", "seqlen_q", "batch_size", "seqlen_k")
        }
        for mode, profile in config["profiles"].items():
            row.update(
                {
                    f"{mode}_ltc_requests": profile["ltc_fabric_requests"],
                    f"{mode}_l2_hit_rate_pct": profile["l2"]["hit_rate_pct"],
                    f"{mode}_l2_hit_rate_usable": profile["l2"]["hit_rate_usable"],
                    f"{mode}_l2_counter_sum_relative_error": profile["l2"][
                        "counter_sum_relative_error"
                    ],
                    f"{mode}_l2_quality_warnings": "; ".join(
                        profile["l2"]["validation_warnings"]
                    ),
                    f"{mode}_hbm_bytes": profile["hbm"]["total_bytes"],
                    f"{mode}_hbm_gb_s": profile["hbm"]["total_gb_s"],
                    f"{mode}_hbm_utilization_pct": profile["hbm"][
                        "peak_sustained_utilization_pct"
                    ],
                    f"{mode}_ncu_duration_ns": profile["hbm"]["duration_ns"],
                    f"{mode}_replay_passes": profile["ncu_replay_passes"],
                }
            )
        row.update(config["comparison"])
        rows.append(row)
    if rows:
        with (root / "boundary_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot_profiles(root, rows, sm_counts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths = []
    for workload, sq in sorted({(row["workload"], row["seqlen_q"]) for row in rows}):
        group = [
            row for row in rows if (row["workload"], row["seqlen_q"]) == (workload, sq)
        ]
        labels = [f"B={row['batch_size']}\nSk={row['seqlen_k']:,}" for row in group]
        x = np.arange(len(group))
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        panels = (
            ("ltc_requests", "LTC fabric requests", "Requests (log scale)", 1, True),
            (
                "l2_hit_rate_pct",
                "L2 sector hit rate (flagged readings omitted)",
                "%",
                1,
                False,
            ),
            ("hbm_bytes", "HBM read + write traffic", "GiB", 2**30, False),
            ("hbm_gb_s", "HBM achieved bandwidth", "GB/s", 1, False),
        )
        for axis, (suffix, title, unit, scale, logarithmic) in zip(
            axes.flat, panels, strict=True
        ):
            for offset, mode, color in (
                (-0.19, "standard", "#5978a5"),
                (0.19, "localized", "#db8748"),
            ):
                values = [row[f"{mode}_{suffix}"] / scale for row in group]
                if suffix == "l2_hit_rate_pct":
                    for index, row in enumerate(group):
                        if not row[f"{mode}_l2_hit_rate_usable"]:
                            values[index] = np.nan
                            axis.text(
                                index + offset,
                                2,
                                "N/A",
                                ha="center",
                                fontsize=7,
                                rotation=90,
                            )
                axis.bar(x + offset, values, width=0.36, label=mode, color=color)
            if logarithmic:
                axis.set_yscale("log")
            axis.set_xticks(x, labels, fontsize=8)
            axis.set_xlim(-0.6, len(group) - 0.4)
            axis.set_ylabel(unit)
            axis.set_title(title)
            axis.legend()
        fig.suptitle(
            f"B200 {sm_counts[0]}/{sm_counts[1]} SM — MLA {workload}, "
            f"Sq={sq} — sparse boundary profiles"
        )
        path = root / "figures" / f"{workload}_sq{sq}_metrics.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(root)))
    return paths


def main():
    args = parse_args()
    args.cache_control, args.clock_control = "all", "boost"
    root = args.output_root.resolve()
    summary_path = root / "matrix_summary.json"
    if args.summarize_only:
        document = json.loads(summary_path.read_text())
        rows = summarize(root, document)
        plot_profiles(root, rows, document["settings"]["expected_partition_sm_counts"])
        return
    configs, sources = load_boundary_configs(args.timing_sources)
    settings = {
        "sources": sources,
        "metrics": list(METRICS),
        "seed": 42,
        "cache_control": "all",
        "clock_control": "boost",
        "replay_mode": "kernel",
        "warmup_launches": 3,
        "profiled_launches": 1,
        "cold_l2": True,
        "expected_partition_sm_counts": args.expected_partition_sm_counts,
        "device": args.device,
        "timing_values_used": False,
    }
    if args.dry_run:
        print(json.dumps({"settings": settings, "configs": configs}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    ncu = memory.resolve_ncu(args.ncu)
    memory.validate_metrics(ncu, args.device, metrics=METRICS)
    if args.resume:
        document = json.loads(summary_path.read_text())
        if document["settings"] != settings:
            raise ValueError("resume settings differ")
        document["environment"].setdefault("resume_scripts", []).append(
            {
                "time": memory.utc_now(),
                "scripts": {
                    str(p.relative_to(memory.REPO_ROOT)): memory.file_sha256(p)
                    for p in (Path(__file__), memory.TARGET, Path(memory.__file__))
                },
            }
        )
    else:
        if summary_path.exists():
            raise FileExistsError(
                f"use --resume or a new output directory: {summary_path}"
            )
        document = {
            "schema": "flashinfer-localized-mla-ncu-boundary-v1",
            "created_at": memory.utc_now(),
            "settings": settings,
            "configs": configs,
            "environment": {
                "git_revision": memory.git_revision(),
                "ncu_version": memory.ncu_version(ncu),
                "gpu_initial": memory.query_gpu(args.device),
                "scripts": {
                    str(p.relative_to(memory.REPO_ROOT)): memory.file_sha256(p)
                    for p in (Path(__file__), memory.TARGET, Path(memory.__file__))
                },
            },
        }
    document["status"] = "running"
    refresh_quality(document)
    document.pop("finished_at", None)
    memory.write_json_atomic(summary_path, document)
    new_count = 0
    for index, config in enumerate(document["configs"]):
        if config["status"] == "complete":
            continue
        attempt_dir = (
            root
            / "profiles"
            / config["workload"]
            / memory.config_key(config)
            / f"attempt_{len(config['attempts']) + 1:03d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        order = (
            ["standard", "localized"] if index % 2 == 0 else ["localized", "standard"]
        )
        attempt = {
            "started_at": memory.utc_now(),
            "mode_order": order,
            "directory": str(attempt_dir.relative_to(root)),
            "status": "running",
        }
        config["attempts"].append(attempt)
        config["status"] = "running"
        memory.write_json_atomic(summary_path, document)
        print(
            f"[{index + 1}/{len(document['configs'])}] {config['workload']} {memory.config_key(config)}",
            flush=True,
        )
        try:
            profiles = {
                mode: profile_one(args, ncu, config, mode, attempt_dir)
                for mode in order
            }
            comparison = memory.compare(profiles)
            standard = profiles["standard"]["ltc_fabric_requests"]
            localized = profiles["localized"]["ltc_fabric_requests"]
            comparison["ltc_reduction_pct"] = (
                100 * (1 - localized / standard) if standard else None
            )
            config.update(
                {"profiles": profiles, "comparison": comparison, "status": "complete"}
            )
            config.pop("error", None)
            attempt["status"] = "complete"
            print(f"  complete; LTC {standard:,.0f} -> {localized:,.0f}", flush=True)
        except BaseException as error:
            config["status"] = attempt["status"] = document["status"] = "failed"
            config["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            attempt["finished_at"] = memory.utc_now()
            document["updated_at"] = memory.utc_now()
            summarize(root, document)
            memory.write_json_atomic(summary_path, document)
        new_count += 1
        if args.max_new_configs is not None and new_count >= args.max_new_configs:
            break
    document["status"] = (
        "complete"
        if all(c["status"] == "complete" for c in document["configs"])
        else "partial"
    )
    document["finished_at"] = memory.utc_now()
    document["environment"]["gpu_final"] = memory.query_gpu(args.device)
    if document["status"] == "complete":
        document["figures"] = plot_profiles(
            root, summarize(root, document), args.expected_partition_sm_counts
        )
    memory.write_json_atomic(summary_path, document)
    print(f"{document['status']}: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
