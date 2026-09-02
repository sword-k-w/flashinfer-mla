#!/usr/bin/env python3
"""Collect isolated NCU LTC-fabric profiles for localized MLA decode."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEQLEN_KS = (1024, 65536, 524288, 1754432)
METRIC = "lts__t_requests_srcunit_ltcfabric.sum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seqlen-q", type=int, choices=range(1, 5), default=1)
    parser.add_argument(
        "--seqlen-ks", type=int, nargs="+", default=list(DEFAULT_SEQLEN_KS)
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.ncu is None:
        resolved = shutil.which("ncu")
        if resolved is None:
            candidate = Path("/usr/local/cuda-13.2/bin/ncu")
            if not candidate.is_file():
                parser.error("ncu was not found; pass --ncu")
            resolved = str(candidate)
        args.ncu = Path(resolved)
    if args.report_dir is None:
        report_name = (
            "localized_mla_capacity_matrix"
            if args.seqlen_q == 1
            else f"localized_mla_sq{args.seqlen_q}_capacity_matrix"
        )
        args.report_dir = (
            SCRIPT_DIR.parent / "reports" / report_name / "iter_000_evaluation"
        )
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


def ncu_version(ncu: Path) -> str:
    result = subprocess.run(
        [str(ncu), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def parse_target_metadata(output: str) -> dict:
    match = re.search(r"^TARGET_METADATA (\{.*\})$", output, re.MULTILINE)
    if match is None:
        raise ValueError("target output did not contain TARGET_METADATA")
    return json.loads(match.group(1))


def parse_ncu_raw(raw: str) -> tuple[str, float]:
    rows = list(csv.reader(raw.splitlines()))
    header = next((row for row in rows if "Kernel Name" in row and METRIC in row), None)
    if header is None:
        raise ValueError("NCU raw output has no metric header")
    data = next(
        (
            row
            for row in rows[rows.index(header) + 1 :]
            if len(row) == len(header) and row and row[0].strip() == "0"
        ),
        None,
    )
    if data is None:
        raise ValueError("NCU raw output has no launch record")
    kernel = data[header.index("Kernel Name")]
    value = float(data[header.index(METRIC)].replace(",", ""))
    return kernel, value


def profile_one(
    args: argparse.Namespace,
    report_dir: Path,
    mode: str,
    seqlen_k: int,
) -> dict:
    shape_dir = report_dir / "ncu_ltc" / f"b{args.batch}_sk{seqlen_k}"
    shape_dir.mkdir(parents=True, exist_ok=True)
    report_stem = shape_dir / mode
    report = report_stem.with_suffix(".ncu-rep")
    log = shape_dir / f"{mode}.log"
    raw_csv = shape_dir / f"{mode}.csv"
    target = SCRIPT_DIR / "profile_cute_dsl_localized_mla_ltc_target.py"
    command = [
        str(args.ncu),
        "--force-overwrite",
        "--export",
        str(report_stem),
        "--metrics",
        METRIC,
        "--profile-from-start",
        "off",
        "--launch-count",
        "1",
        "--kernel-name-base",
        "demangled",
        str(args.python),
        "-P",
        str(target),
        "--mode",
        mode,
        "--batch",
        str(args.batch),
        "--seqlen-q",
        str(args.seqlen_q),
        "--seqlen-k",
        str(seqlen_k),
        "--device",
        str(args.device),
    ]
    environment = os.environ.copy()
    environment.setdefault("MAX_JOBS", "2")
    environment.setdefault("FLASHINFER_NVCC_THREADS", "2")
    started = time.monotonic()
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.monotonic() - started
    log.write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"NCU {mode} B={args.batch} Sk={seqlen_k} failed; inspect {log}\n"
            f"{result.stdout}"
        )
    if not report.is_file():
        raise RuntimeError(f"NCU did not write {report}")
    metadata = parse_target_metadata(result.stdout)
    raw_result = subprocess.run(
        [str(args.ncu), "--import", str(report), "--csv", "--page", "raw"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    raw_csv.write_text(raw_result.stdout)
    kernel, metric_value = parse_ncu_raw(raw_result.stdout)
    return {
        "mode": mode,
        "batch_size": args.batch,
        "seqlen_q": args.seqlen_q,
        "seqlen_k": seqlen_k,
        "metric": METRIC,
        "metric_value": metric_value,
        "kernel": kernel,
        "target_metadata": metadata,
        "wall_seconds": wall_seconds,
        "report": str(report.relative_to(report_dir)),
        "raw_csv": str(raw_csv.relative_to(report_dir)),
        "log": str(log.relative_to(report_dir)),
        "command": command,
    }


def main() -> None:
    args = parse_args()
    if args.batch < 2:
        raise ValueError("--batch must be >= 2")
    if any(value <= 0 or value % 64 for value in args.seqlen_ks):
        raise ValueError("all --seqlen-ks values must be positive multiples of 64")
    report_dir = args.report_dir.expanduser().resolve()
    output = report_dir / "post_ncu_ltc.json"
    document = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profiler": "Nsight Compute",
        "ncu_version": ncu_version(args.ncu),
        "metric": METRIC,
        "metric_description": "# of LTS requests from LTC Fabric",
        "batch_size": args.batch,
        "seqlen_q": args.seqlen_q,
        "seqlen_ks": args.seqlen_ks,
        "isolation": "one fresh process and one profiled launch per mode and shape",
        "results": [],
    }
    if args.resume and output.is_file():
        document = json.loads(output.read_text())
        if document["batch_size"] != args.batch:
            raise ValueError("resume file batch differs from command line")
        if document["seqlen_ks"] != args.seqlen_ks:
            raise ValueError("resume file seqlen axis differs from command line")
        if document.get("seqlen_q", 1) != args.seqlen_q:
            raise ValueError("resume file seqlen_q differs from command line")
        document["seqlen_q"] = args.seqlen_q
        document["status"] = "running"
        document.pop("error", None)
        document.pop("finished_at", None)
    completed = {(row["mode"], row["seqlen_k"]) for row in document.get("results", [])}
    try:
        for seqlen_k in args.seqlen_ks:
            for mode in ("standard", "localized"):
                if (mode, seqlen_k) in completed:
                    continue
                print(f"PROFILE {mode} B={args.batch} Sk={seqlen_k:,}", flush=True)
                row = profile_one(args, report_dir, mode, seqlen_k)
                document["results"].append(row)
                write_json_atomic(output, document)
                print(f"DONE {mode}: {row['metric_value']:,.0f} requests", flush=True)
    except BaseException as error:
        document["status"] = "failed"
        document["error"] = f"{type(error).__name__}: {error}"
        document["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(output, document)
        raise

    by_shape = {}
    for row in document["results"]:
        by_shape.setdefault(row["seqlen_k"], {})[row["mode"]] = row["metric_value"]
    comparisons = []
    for seqlen_k in args.seqlen_ks:
        standard = by_shape[seqlen_k]["standard"]
        localized = by_shape[seqlen_k]["localized"]
        comparisons.append(
            {
                "batch_size": args.batch,
                "seqlen_q": args.seqlen_q,
                "seqlen_k": seqlen_k,
                "standard_requests": standard,
                "localized_requests": localized,
                "localized_over_standard": localized / standard,
                "reduction_fraction": 1.0 - localized / standard,
            }
        )
    document["comparisons"] = comparisons
    document["status"] = "passed"
    document["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output, document)
    print(output)


if __name__ == "__main__":
    main()
