#!/usr/bin/env python3
"""Collect cold-L2 NCU L2/HBM metrics for standard versus localized MLA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TARGET = SCRIPT_DIR / "profile_cute_dsl_localized_mla_ltc_target.py"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "localized_mla_b200_random_data_matrix"
DEFAULT_TIMING_SOURCES = (
    DEFAULT_REPORT_ROOT / "sq1" / "iter_000_evaluation" / "post_flops.json",
    DEFAULT_REPORT_ROOT / "sq4" / "iter_000_evaluation" / "post_flops.json",
)

DURATION_METRIC = "gpu__time_duration.avg"
HBM_READ_BYTES_METRIC = "dram__bytes_read.sum"
HBM_WRITE_BYTES_METRIC = "dram__bytes_write.sum"
HBM_READ_BANDWIDTH_METRIC = "dram__bytes_read.sum.per_second"
HBM_WRITE_BANDWIDTH_METRIC = "dram__bytes_write.sum.per_second"
HBM_UTILIZATION_METRIC = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
L2_HIT_RATE_METRIC = "lts__t_sector_hit_rate.pct"
L2_TOTAL_SECTORS_METRIC = "lts__t_sectors.sum"
L2_HIT_SECTORS_METRIC = "lts__t_sectors_lookup_hit.sum"
L2_MISS_SECTORS_METRIC = "lts__t_sectors_lookup_miss.sum"
MEMORY_METRICS = (
    DURATION_METRIC,
    HBM_READ_BYTES_METRIC,
    HBM_WRITE_BYTES_METRIC,
    HBM_READ_BANDWIDTH_METRIC,
    HBM_WRITE_BANDWIDTH_METRIC,
    HBM_UTILIZATION_METRIC,
    L2_HIT_RATE_METRIC,
    L2_TOTAL_SECTORS_METRIC,
    L2_HIT_SECTORS_METRIC,
    L2_MISS_SECTORS_METRIC,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing-sources",
        nargs="+",
        type=Path,
        default=list(DEFAULT_TIMING_SOURCES),
        help="Completed timing JSON files used only as shape/configuration manifests.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-sm-p0", type=int)
    parser.add_argument("--expected-sm-p1", type=int)
    parser.add_argument("--cache-control", choices=("all", "none"), default="all")
    parser.add_argument(
        "--clock-control",
        choices=("base", "boost", "force-boost", "none"),
        default="boost",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-metric-validation", action="store_true")
    parser.add_argument(
        "--max-new-configs",
        type=int,
        help="Checkpoint and stop after this many new standard/localized pairs.",
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if (args.expected_sm_p0 is None) != (args.expected_sm_p1 is None):
        parser.error("--expected-sm-p0 and --expected-sm-p1 must be passed together")
    if args.max_new_configs is not None and args.max_new_configs <= 0:
        parser.error("--max-new-configs must be positive")
    if args.resume and args.output_root is None:
        parser.error("--resume requires --output-root")
    if args.plot_only and args.output_root is None:
        parser.error("--plot-only requires --output-root")
    if args.plot_only and args.resume:
        parser.error("--plot-only and --resume cannot be combined")
    if args.dry_run and (args.resume or args.plot_only):
        parser.error("--dry-run cannot be combined with --resume or --plot-only")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve_ncu(explicit: Path | None) -> Path:
    candidates: list[str | Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if configured := os.environ.get("NCU"):
        candidates.append(configured)
    if cuda_home := os.environ.get("CUDA_HOME"):
        candidates.append(Path(cuda_home) / "bin" / "ncu")
    if found := shutil.which("ncu"):
        candidates.append(found)
    candidates.extend(("/usr/local/cuda-13.2/bin/ncu", "/usr/local/cuda/bin/ncu"))
    for value in candidates:
        path = Path(value).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError("ncu was not found; pass --ncu")


def run_text(
    command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ncu_version(ncu: Path) -> str:
    return run_text([str(ncu), "--version"]).stdout.strip()


def validate_metrics(ncu: Path, device: int) -> None:
    output = run_text(
        [str(ncu), "--query-metrics-mode", "all", "--devices", str(device)]
    ).stdout
    available = {
        line.split(maxsplit=1)[0]
        for line in output.splitlines()
        if line and not line[0].isspace() and "__" in line
    }
    missing = [name for name in MEMORY_METRICS if name not in available]
    if missing:
        raise RuntimeError("unsupported NCU metrics: " + ", ".join(missing))


def query_gpu(device: int) -> dict[str, str]:
    fields = (
        "index,name,uuid,pstate,memory.total,memory.used,memory.free,"
        "utilization.gpu,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(device),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()}
    names = (
        "index",
        "name",
        "uuid",
        "pstate",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_pct",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
        "power_w",
    )
    return dict(
        zip(names, (part.strip() for part in completed.stdout.split(",")), strict=True)
    )


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def config_key(config: dict[str, Any]) -> str:
    return f"sq{config['seqlen_q']}_b{config['batch_size']}_sk{config['seqlen_k']}"


def load_configs(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source_arg in paths:
        source = source_arg.expanduser().resolve()
        document = json.loads(source.read_text())
        if document.get("status") != "passed" or not document.get("results"):
            raise ValueError(f"timing source is not complete: {source}")
        configuration = document.get("configuration", {})
        if configuration.get("data_initialization") != "random":
            raise ValueError(f"timing source is not a random-data matrix: {source}")
        seqlen_q = int(configuration["seqlen_q"])
        sources.append(
            {
                "path": str(source),
                "sha256": file_sha256(source),
                "seqlen_q": seqlen_q,
                "shape_count": len(document["results"]),
                "role": "shape_and_configuration_manifest_only",
            }
        )
        for row in document["results"]:
            if int(row["seqlen_q"]) != seqlen_q:
                raise ValueError(f"inconsistent Sq in {source}")
            configs.append(
                {
                    "index": len(configs),
                    "status": "pending",
                    "batch_size": int(row["batch_size"]),
                    "seqlen_q": seqlen_q,
                    "seqlen_k": int(row["seqlen_k"]),
                    "split_kv": int(row["split_kv"]),
                    "source": str(source),
                    "attempts": [],
                }
            )
    keys = [config_key(config) for config in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("timing sources contain duplicate configurations")
    return configs, sources


def settings(args: argparse.Namespace, sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "flashinfer-localized-mla-ncu-memory-profile-v1",
        "timing_sources": sources,
        "timing_values_used": False,
        "modes": ["standard", "localized"],
        "data_initialization": "random",
        "data_seed": 42,
        "cold_l2": True,
        "warmup_launches": 3,
        "profiled_launches": 1,
        "metrics": list(MEMORY_METRICS),
        "cache_control": args.cache_control,
        "clock_control": args.clock_control,
        "replay_mode": "kernel",
        "expected_sm_p0": args.expected_sm_p0,
        "expected_sm_p1": args.expected_sm_p1,
        "device": args.device,
    }


def default_output_root(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    split = (
        "unknown"
        if args.expected_sm_p0 is None
        else f"{args.expected_sm_p0}_p1_{args.expected_sm_p1}"
    )
    return DEFAULT_REPORT_ROOT / "ncu_memory" / "results" / f"{timestamp}_p0_{split}"


def prepare_document(
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> tuple[Path, Path, dict[str, Any]]:
    root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else default_output_root(args).resolve()
    )
    summary = root / "matrix_summary.json"
    expected_settings = settings(args, sources)
    if args.resume:
        document = json.loads(summary.read_text())
        if document.get("settings") != expected_settings:
            raise ValueError("resume settings differ from existing matrix summary")
        if [config_key(item) for item in document["configs"]] != [
            config_key(item) for item in configs
        ]:
            raise ValueError("resume configuration grid differs")
        document["status"] = "running"
        document["updated_at"] = utc_now()
        document.pop("finished_at", None)
    else:
        root.mkdir(parents=True, exist_ok=True)
        if summary.exists():
            raise FileExistsError(f"output already exists: {summary}")
        document = {
            "schema": "flashinfer-localized-mla-ncu-memory-matrix-v1",
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "settings": expected_settings,
            "environment": {},
            "observed_partition_sm_counts": None,
            "configs": configs,
        }
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(summary, document)
    return root, summary, document


def parse_target_metadata(output: str) -> dict[str, Any]:
    match = re.search(r"^TARGET_METADATA (\{.*\})$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError("target output did not contain TARGET_METADATA")
    return json.loads(match.group(1))


def parse_number(value: str, name: str) -> float:
    normalized = value.replace(",", "").strip()
    if not normalized or normalized.upper() == "N/A":
        raise RuntimeError(f"NCU did not produce a numeric value for {name}")
    try:
        return float(normalized)
    except ValueError as error:
        raise RuntimeError(f"invalid NCU value for {name}: {value!r}") from error


def parse_ncu_report(ncu: Path, report: Path, raw_csv: Path) -> dict[str, Any]:
    completed = run_text(
        [
            str(ncu),
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ]
    )
    raw_csv.write_text(completed.stdout)
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    matches: list[tuple[dict[str, str], dict[str, str]]] = []
    for index, row in enumerate(rows):
        if not row or row[0] != "ID" or index + 2 >= len(rows):
            continue
        if any(metric not in row for metric in MEMORY_METRICS):
            continue
        header = row
        units = dict(zip(header, rows[index + 1], strict=True))
        values = rows[index + 2]
        if len(values) >= len(header) and values[0].strip():
            matches.append((dict(zip(header, values, strict=True)), units))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one NCU kernel row, found {len(matches)}")
    record, units = matches[0]
    metrics = {
        name: {"value": parse_number(record[name], name), "unit": units.get(name, "")}
        for name in MEMORY_METRICS
    }
    passes = parse_number(record.get("profiler__replayer_passes", "1"), "replay passes")
    return {
        "report": str(report),
        "raw_csv": str(raw_csv),
        "kernel_name": record["Kernel Name"],
        "ncu_replay_passes": int(passes),
        "metrics": metrics,
    }


def metric_value(profile: dict[str, Any], name: str) -> float:
    value = float(profile["metrics"][name]["value"])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric {name}: {value}")
    return value


def derive_metrics(profile: dict[str, Any]) -> None:
    duration = metric_value(profile, DURATION_METRIC)
    read_bytes = metric_value(profile, HBM_READ_BYTES_METRIC)
    write_bytes = metric_value(profile, HBM_WRITE_BYTES_METRIC)
    read_per_second = metric_value(profile, HBM_READ_BANDWIDTH_METRIC)
    write_per_second = metric_value(profile, HBM_WRITE_BANDWIDTH_METRIC)
    hit_rate = metric_value(profile, L2_HIT_RATE_METRIC)
    total_sectors = metric_value(profile, L2_TOTAL_SECTORS_METRIC)
    hits = metric_value(profile, L2_HIT_SECTORS_METRIC)
    misses = metric_value(profile, L2_MISS_SECTORS_METRIC)
    values = (
        duration,
        read_bytes,
        write_bytes,
        read_per_second,
        write_per_second,
        total_sectors,
        hits,
        misses,
    )
    if any(value < 0 for value in values):
        raise RuntimeError("NCU returned a negative duration/traffic/count metric")
    lookup = hits + misses
    if not 0.0 <= hit_rate <= 100.0 or lookup <= 0:
        raise RuntimeError(f"invalid L2 counters: hit_rate={hit_rate}, lookup={lookup}")
    profile["hbm"] = {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": read_bytes + write_bytes,
        "read_gb_s": read_per_second / 1e9,
        "write_gb_s": write_per_second / 1e9,
        "total_gb_s": (read_per_second + write_per_second) / 1e9,
        "peak_sustained_utilization_pct": metric_value(profile, HBM_UTILIZATION_METRIC),
        "duration_ns": duration,
    }
    profile["l2"] = {
        "hit_rate_pct": hit_rate,
        "hit_rate_from_lookup_counts_pct": 100.0 * hits / lookup,
        "total_sectors": total_sectors,
        "lookup_hit_sectors": hits,
        "lookup_miss_sectors": misses,
        "duration_ns": duration,
    }


def profile_command(
    args: argparse.Namespace,
    ncu: Path,
    config: dict[str, Any],
    mode: str,
    stem: Path,
) -> list[str]:
    python = args.python.expanduser()
    if not python.is_absolute():
        python = REPO_ROOT / python
    return [
        str(ncu),
        "--force-overwrite",
        "--export",
        str(stem),
        "--devices",
        str(args.device),
        "--profile-from-start",
        "off",
        "--launch-count",
        "1",
        "--kernel-name-base",
        "demangled",
        "--cache-control",
        args.cache_control,
        "--clock-control",
        args.clock_control,
        "--replay-mode",
        "kernel",
        "--metrics",
        ",".join(MEMORY_METRICS),
        str(python.absolute()),
        "-P",
        str(TARGET),
        "--mode",
        mode,
        "--batch",
        str(config["batch_size"]),
        "--seqlen-q",
        str(config["seqlen_q"]),
        "--seqlen-k",
        str(config["seqlen_k"]),
        "--device",
        str(args.device),
        "--warmups",
        "3",
        "--data-initialization",
        "random",
        "--seed",
        "42",
        "--cold-l2",
    ]


def validate_target(
    metadata: dict[str, Any], config: dict[str, Any], mode: str
) -> None:
    expected = {
        "mode": mode,
        "batch_size": config["batch_size"],
        "seqlen_q": config["seqlen_q"],
        "seqlen_k": config["seqlen_k"],
        "split_kv": config["split_kv"],
        "warmup_launches": 3,
        "profiled_launches": 1,
        "data_initialization": "random",
        "data_seed": 42,
        "l2_cache_policy": "triton_clear_cache_before_each_attention_launch",
        "total_sm_count": 148,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"target metadata mismatch: {mismatches}")
    split = metadata.get("partition_sm_counts")
    if mode == "standard" and split is not None:
        raise RuntimeError("standard target unexpectedly reported a partition split")
    if mode == "localized" and (
        not isinstance(split, list) or len(split) != 2 or sum(split) != 148
    ):
        raise RuntimeError(f"invalid localized partition split: {split}")


def profile_one(
    args: argparse.Namespace,
    ncu: Path,
    config: dict[str, Any],
    mode: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    stem = attempt_dir / mode
    report = stem.with_suffix(".ncu-rep")
    log = stem.with_suffix(".log")
    raw_csv = stem.with_suffix(".csv")
    metadata_path = attempt_dir / f"{mode}_metadata.json"
    command = profile_command(args, ncu, config, mode, stem)
    environment = os.environ.copy()
    environment.setdefault("MAX_JOBS", "2")
    environment.setdefault("FLASHINFER_NVCC_THREADS", "2")
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not python_path
        else os.pathsep.join((str(REPO_ROOT), python_path))
    )
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.monotonic() - started
    log.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"NCU {mode} failed; see {log}")
    if not report.is_file():
        raise RuntimeError(f"NCU did not create {report}")
    metadata = parse_target_metadata(completed.stdout)
    validate_target(metadata, config, mode)
    write_json_atomic(metadata_path, metadata)
    profile = parse_ncu_report(ncu, report, raw_csv)
    if profile["ncu_replay_passes"] != 4:
        raise RuntimeError(
            f"expected 4 NCU replay passes, found {profile['ncu_replay_passes']}"
        )
    profile.update(
        {
            "mode": mode,
            "wall_seconds": wall_seconds,
            "log": str(log),
            "target_metadata": metadata,
            "command": command,
        }
    )
    derive_metrics(profile)
    return profile


def compare(profiles: dict[str, dict[str, Any]]) -> dict[str, float]:
    standard = profiles["standard"]
    localized = profiles["localized"]
    standard_bw = standard["hbm"]["total_gb_s"]
    localized_bw = localized["hbm"]["total_gb_s"]
    standard_bytes = standard["hbm"]["total_bytes"]
    localized_bytes = localized["hbm"]["total_bytes"]
    localized_duration = localized["hbm"]["duration_ns"]
    return {
        "localized_minus_standard_l2_hit_rate_pp": (
            localized["l2"]["hit_rate_pct"] - standard["l2"]["hit_rate_pct"]
        ),
        "localized_to_standard_hbm_bandwidth_ratio": localized_bw / standard_bw,
        "localized_minus_standard_hbm_utilization_pp": (
            localized["hbm"]["peak_sustained_utilization_pct"]
            - standard["hbm"]["peak_sustained_utilization_pct"]
        ),
        "localized_to_standard_hbm_total_bytes_ratio": (
            localized_bytes / standard_bytes
        ),
        "standard_to_localized_ncu_duration_ratio": (
            standard["hbm"]["duration_ns"] / localized_duration
        ),
    }


def next_attempt_dir(root: Path, config: dict[str, Any]) -> Path:
    base = (
        root
        / "profiles"
        / f"sq{config['seqlen_q']}"
        / f"b{config['batch_size']}_sk{config['seqlen_k']}"
    )
    attempt = len(config["attempts"]) + 1
    return base / f"attempt_{attempt:03d}"


def run_matrix(
    args: argparse.Namespace,
    ncu: Path,
    root: Path,
    summary: Path,
    document: dict[str, Any],
) -> bool:
    new_configs = 0
    failed = False
    total = len(document["configs"])
    for position, config in enumerate(document["configs"], start=1):
        if config.get("status") == "complete":
            continue
        attempt_dir = next_attempt_dir(root, config)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        order = (
            ("standard", "localized")
            if int(config["index"]) % 2 == 0
            else ("localized", "standard")
        )
        attempt = {
            "started_at": utc_now(),
            "mode_order": list(order),
            "output_dir": str(attempt_dir.relative_to(root)),
            "status": "running",
        }
        config["status"] = "running"
        config["attempts"].append(attempt)
        document["updated_at"] = utc_now()
        write_json_atomic(summary, document)
        print(
            f"[{position}/{total}] {config_key(config)} order={'/'.join(order)}",
            flush=True,
        )
        try:
            profiles = {
                mode: profile_one(args, ncu, config, mode, attempt_dir)
                for mode in order
            }
            actual_split = profiles["localized"]["target_metadata"][
                "partition_sm_counts"
            ]
            expected_split = (
                None
                if args.expected_sm_p0 is None
                else [args.expected_sm_p0, args.expected_sm_p1]
            )
            locked_split = document.get("observed_partition_sm_counts")
            if expected_split is not None and actual_split != expected_split:
                raise RuntimeError(
                    f"localized split {actual_split} != expected {expected_split}"
                )
            if locked_split is not None and actual_split != locked_split:
                raise RuntimeError(
                    f"localized split changed from {locked_split} to {actual_split}"
                )
            document["observed_partition_sm_counts"] = actual_split
            config["profiles"] = profiles
            config["comparison"] = compare(profiles)
            config["status"] = "complete"
            config.pop("error", None)
            attempt["status"] = "complete"
            attempt["finished_at"] = utc_now()
            attempt["profiles"] = {
                mode: {
                    "wall_seconds": profile["wall_seconds"],
                    "ncu_replay_passes": profile["ncu_replay_passes"],
                }
                for mode, profile in profiles.items()
            }
            l2_delta = config["comparison"]["localized_minus_standard_l2_hit_rate_pp"]
            hbm_ratio = config["comparison"][
                "localized_to_standard_hbm_bandwidth_ratio"
            ]
            print(
                f"  done L2_delta={l2_delta:+.3f} pp HBM_BW={hbm_ratio:.4f}x",
                flush=True,
            )
        except Exception as error:
            failed = True
            message = f"{type(error).__name__}: {error}"
            config["status"] = "failed"
            config["error"] = message
            attempt["status"] = "failed"
            attempt["finished_at"] = utc_now()
            attempt["error"] = message
            print(f"  failed: {message}", flush=True)
        if not document["environment"]:
            document["environment"] = {
                "git_revision": git_revision(),
                "python": str(
                    (
                        args.python.expanduser()
                        if args.python.expanduser().is_absolute()
                        else REPO_ROOT / args.python.expanduser()
                    ).absolute()
                ),
                "ncu": str(ncu),
                "ncu_version": ncu_version(ncu),
                "gpu_initial": query_gpu(args.device),
            }
        document["updated_at"] = utc_now()
        write_json_atomic(summary, document)
        new_configs += 1
        if failed and args.fail_fast:
            break
        if args.max_new_configs is not None and new_configs >= args.max_new_configs:
            break
    return failed


def complete_configs(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        config for config in document["configs"] if config.get("status") == "complete"
    ]


def geometric_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value) and value > 0]
    return math.exp(sum(math.log(value) for value in finite) / len(finite))


def write_aggregate_csv(path: Path, configs: list[dict[str, Any]]) -> None:
    fields = (
        "seqlen_q",
        "batch_size",
        "seqlen_k",
        "standard_ncu_duration_ns",
        "localized_ncu_duration_ns",
        "standard_l2_hit_rate_pct",
        "localized_l2_hit_rate_pct",
        "localized_minus_standard_l2_hit_rate_pp",
        "standard_l2_count_hit_rate_pct",
        "localized_l2_count_hit_rate_pct",
        "standard_hbm_read_bytes",
        "standard_hbm_write_bytes",
        "localized_hbm_read_bytes",
        "localized_hbm_write_bytes",
        "standard_hbm_total_gb_s",
        "localized_hbm_total_gb_s",
        "localized_to_standard_hbm_bandwidth_ratio",
        "standard_hbm_utilization_pct",
        "localized_hbm_utilization_pct",
        "localized_minus_standard_hbm_utilization_pp",
        "localized_to_standard_hbm_total_bytes_ratio",
        "standard_to_localized_ncu_duration_ratio",
        "standard_replay_passes",
        "localized_replay_passes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for config in configs:
            standard = config["profiles"]["standard"]
            localized = config["profiles"]["localized"]
            comparison = config["comparison"]
            writer.writerow(
                {
                    "seqlen_q": config["seqlen_q"],
                    "batch_size": config["batch_size"],
                    "seqlen_k": config["seqlen_k"],
                    "standard_ncu_duration_ns": standard["hbm"]["duration_ns"],
                    "localized_ncu_duration_ns": localized["hbm"]["duration_ns"],
                    "standard_l2_hit_rate_pct": standard["l2"]["hit_rate_pct"],
                    "localized_l2_hit_rate_pct": localized["l2"]["hit_rate_pct"],
                    "localized_minus_standard_l2_hit_rate_pp": comparison[
                        "localized_minus_standard_l2_hit_rate_pp"
                    ],
                    "standard_l2_count_hit_rate_pct": standard["l2"][
                        "hit_rate_from_lookup_counts_pct"
                    ],
                    "localized_l2_count_hit_rate_pct": localized["l2"][
                        "hit_rate_from_lookup_counts_pct"
                    ],
                    "standard_hbm_read_bytes": standard["hbm"]["read_bytes"],
                    "standard_hbm_write_bytes": standard["hbm"]["write_bytes"],
                    "localized_hbm_read_bytes": localized["hbm"]["read_bytes"],
                    "localized_hbm_write_bytes": localized["hbm"]["write_bytes"],
                    "standard_hbm_total_gb_s": standard["hbm"]["total_gb_s"],
                    "localized_hbm_total_gb_s": localized["hbm"]["total_gb_s"],
                    "localized_to_standard_hbm_bandwidth_ratio": comparison[
                        "localized_to_standard_hbm_bandwidth_ratio"
                    ],
                    "standard_hbm_utilization_pct": standard["hbm"][
                        "peak_sustained_utilization_pct"
                    ],
                    "localized_hbm_utilization_pct": localized["hbm"][
                        "peak_sustained_utilization_pct"
                    ],
                    "localized_minus_standard_hbm_utilization_pp": comparison[
                        "localized_minus_standard_hbm_utilization_pp"
                    ],
                    "localized_to_standard_hbm_total_bytes_ratio": comparison[
                        "localized_to_standard_hbm_total_bytes_ratio"
                    ],
                    "standard_to_localized_ncu_duration_ratio": comparison[
                        "standard_to_localized_ncu_duration_ratio"
                    ],
                    "standard_replay_passes": standard["ncu_replay_passes"],
                    "localized_replay_passes": localized["ncu_replay_passes"],
                }
            )


def summarize_group(configs: list[dict[str, Any]]) -> dict[str, Any]:
    l2_standard = [
        item["profiles"]["standard"]["l2"]["hit_rate_pct"] for item in configs
    ]
    l2_localized = [
        item["profiles"]["localized"]["l2"]["hit_rate_pct"] for item in configs
    ]
    l2_deltas = [
        item["comparison"]["localized_minus_standard_l2_hit_rate_pp"]
        for item in configs
    ]
    bandwidth_ratios = [
        item["comparison"]["localized_to_standard_hbm_bandwidth_ratio"]
        for item in configs
    ]
    byte_ratios = [
        item["comparison"]["localized_to_standard_hbm_total_bytes_ratio"]
        for item in configs
    ]
    duration_ratios = [
        item["comparison"]["standard_to_localized_ncu_duration_ratio"]
        for item in configs
    ]
    return {
        "config_count": len(configs),
        "median_standard_l2_hit_rate_pct": median(l2_standard),
        "median_localized_l2_hit_rate_pct": median(l2_localized),
        "median_l2_delta_pp": median(l2_deltas),
        "l2_higher_count": sum(value > 0 for value in l2_deltas),
        "localized_to_standard_hbm_bandwidth_ratio_geomean": geometric_mean(
            bandwidth_ratios
        ),
        "localized_hbm_bandwidth_higher_count": sum(
            value > 1 for value in bandwidth_ratios
        ),
        "localized_to_standard_hbm_total_bytes_ratio_geomean": geometric_mean(
            byte_ratios
        ),
        "standard_to_localized_ncu_duration_ratio_geomean": geometric_mean(
            duration_ratios
        ),
    }


def write_aggregate_summary(path: Path, configs: list[dict[str, Any]]) -> None:
    if not configs:
        write_json_atomic(
            path,
            {
                "schema": "flashinfer-localized-mla-ncu-memory-aggregate-v1",
                "overall": None,
                "groups": {},
            },
        )
        return
    groups = {
        f"sq{sq}": summarize_group(
            [config for config in configs if config["seqlen_q"] == sq]
        )
        for sq in sorted({config["seqlen_q"] for config in configs})
    }
    write_json_atomic(
        path,
        {
            "schema": "flashinfer-localized-mla-ncu-memory-aggregate-v1",
            "overall": summarize_group(configs),
            "groups": groups,
        },
    )


def plot_three_panel(
    path: Path,
    configs: list[dict[str, Any]],
    sq: int,
    extractor: Callable[[dict[str, Any], str], float],
    *,
    title: str,
    unit: str,
    comparison: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    group = [config for config in configs if config["seqlen_q"] == sq]
    batches = sorted({config["batch_size"] for config in group})
    seqlens = sorted({config["seqlen_k"] for config in group})
    lookup = {(config["batch_size"], config["seqlen_k"]): config for config in group}
    arrays = {
        mode: np.asarray(
            [
                [extractor(lookup[(batch, seqlen)], mode) for seqlen in seqlens]
                for batch in batches
            ],
            dtype=float,
        )
        for mode in ("standard", "localized")
    }
    if comparison == "delta":
        arrays["comparison"] = arrays["localized"] - arrays["standard"]
        center = 0.0
        comparison_title = "Localized - standard"
    else:
        arrays["comparison"] = arrays["localized"] / arrays["standard"]
        center = 1.0
        comparison_title = "Localized / standard"
    base = np.concatenate((arrays["standard"].ravel(), arrays["localized"].ravel()))
    base_min, base_max = float(base.min()), float(base.max())
    compared = arrays["comparison"]
    radius = max(float(np.max(np.abs(compared - center))), 1e-12)
    fig, axes = plt.subplots(3, 1, figsize=(20, 13.5), constrained_layout=True)
    specifications = (
        ("standard", "Standard (baseline)", "viridis", None),
        ("localized", "Localized", "viridis", None),
        ("comparison", comparison_title, "RdBu_r", center),
    )
    for axis, (name, subtitle, cmap, norm_center) in zip(
        axes, specifications, strict=True
    ):
        if norm_center is None:
            image = axis.imshow(
                arrays[name], aspect="auto", cmap=cmap, vmin=base_min, vmax=base_max
            )
        else:
            image = axis.imshow(
                arrays[name],
                aspect="auto",
                cmap=cmap,
                norm=TwoSlopeNorm(
                    vmin=norm_center - radius,
                    vcenter=norm_center,
                    vmax=norm_center + radius,
                ),
            )
        axis.set_xticks(
            range(len(seqlens)), [f"{value:,}" for value in seqlens], rotation=35
        )
        axis.set_yticks(range(len(batches)), [str(value) for value in batches])
        axis.set_xlabel("seqlen_k")
        axis.set_ylabel("Batch")
        axis.set_title(subtitle)
        for row in range(len(batches)):
            for column in range(len(seqlens)):
                value = float(arrays[name][row, column])
                if name == "comparison":
                    label = (
                        f"{value:+.2f}" if comparison == "delta" else f"{value:.4f}x"
                    )
                elif unit == "bytes":
                    label = f"{value / 1e9:.2f}G"
                elif unit == "GB/s":
                    label = f"{value:.0f}"
                else:
                    label = f"{value:.2f}"
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if luminance > 0.55 else "white",
                )
        fig.colorbar(
            image,
            ax=axis,
            pad=0.01,
            label=("ratio" if name == "comparison" and comparison == "ratio" else unit),
        )
    fig.suptitle(f"NVIDIA B200 localized MLA, Sq={sq}: {title}", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_plots(root: Path, configs: list[dict[str, Any]]) -> list[str]:
    specs = (
        (
            "l2_hit_rate",
            lambda config, mode: config["profiles"][mode]["l2"]["hit_rate_pct"],
            "L2 sector hit rate",
            "%",
            "delta",
        ),
        (
            "hbm_total_bandwidth",
            lambda config, mode: config["profiles"][mode]["hbm"]["total_gb_s"],
            "HBM total achieved bandwidth",
            "GB/s",
            "ratio",
        ),
        (
            "hbm_utilization",
            lambda config, mode: config["profiles"][mode]["hbm"][
                "peak_sustained_utilization_pct"
            ],
            "HBM peak-sustained utilization",
            "% of peak",
            "delta",
        ),
        (
            "hbm_total_bytes",
            lambda config, mode: config["profiles"][mode]["hbm"]["total_bytes"],
            "HBM total traffic",
            "bytes",
            "ratio",
        ),
    )
    paths: list[str] = []
    for sq in sorted({config["seqlen_q"] for config in configs}):
        for stem, extractor, title, unit, comparison in specs:
            path = root / "plots" / f"sq{sq}_{stem}.png"
            plot_three_panel(
                path,
                configs,
                sq,
                extractor,
                title=title,
                unit=unit,
                comparison=comparison,
            )
            paths.append(str(path.relative_to(root)))
    return paths


def finalize(root: Path, document: dict[str, Any]) -> None:
    configs = complete_configs(document)
    for config in configs:
        # A successful retry supersedes any prior top-level failure. Attempt
        # records retain the earlier error and its log path for provenance.
        config.pop("error", None)
    csv_path = root / "memory_matrix.csv"
    aggregate_path = root / "aggregate_summary.json"
    write_aggregate_csv(csv_path, configs)
    write_aggregate_summary(aggregate_path, configs)
    expected = len(document["configs"])
    document["artifacts"] = {
        "complete_config_count": len(configs),
        "expected_config_count": expected,
        "aggregate_csv": str(csv_path.relative_to(root)),
        "aggregate_summary": str(aggregate_path.relative_to(root)),
        "plots": generate_plots(root, configs) if len(configs) == expected else [],
    }


def main() -> None:
    args = parse_args()
    if args.plot_only:
        root = args.output_root.expanduser().resolve()
        summary = root / "matrix_summary.json"
        document = json.loads(summary.read_text())
        finalize(root, document)
        document["updated_at"] = utc_now()
        write_json_atomic(summary, document)
        print(root)
        return
    configs, sources = load_configs(args.timing_sources)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_count": len(configs),
                    "ncu_report_count": 2 * len(configs),
                    "expected_replay_passes": 8 * len(configs),
                    "metrics": list(MEMORY_METRICS),
                    "configs": [config_key(config) for config in configs],
                },
                indent=2,
            )
        )
        return
    root, summary, document = prepare_document(args, configs, sources)
    ncu = resolve_ncu(args.ncu)
    if not args.skip_metric_validation:
        print("Validating the ten NCU metrics on the selected GPU...", flush=True)
        validate_metrics(ncu, args.device)
    failed = run_matrix(args, ncu, root, summary, document)
    finalize(root, document)
    complete = len(complete_configs(document)) == len(document["configs"])
    document["environment"]["gpu_final"] = query_gpu(args.device)
    document["status"] = "complete" if complete else ("failed" if failed else "running")
    document["updated_at"] = utc_now()
    if complete:
        document["finished_at"] = utc_now()
    write_json_atomic(summary, document)
    print(root)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
