#!/usr/bin/env python3
"""Run Sq=128 dense timing, then separate LTC and memory profiles, sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
OLD = REPO / "reports/localized_mla_b200_74_74_20260905"
PREFILL_REFERENCE = (
    REPO / "reports/localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json"
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sample(root):
    mem = {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    cpu = Path("/proc/stat").read_text().splitlines()[0]
    rss_kib = 0
    for p in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in p.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
        except (OSError, ProcessLookupError):
            pass
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu,clocks.sm",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    value = {
        "time": now(),
        "logical_cpus": os.cpu_count(),
        "load": os.getloadavg(),
        "mem_available_kib": mem["MemAvailable"],
        "aggregate_process_rss_kib": rss_kib,
        "cpu_jiffies": cpu,
        "tmp_disk_free_bytes": shutil.disk_usage("/tmp").free,
        "gpu": gpu,
    }
    with (root / "resource_samples.jsonl").open("a") as handle:
        handle.write(json.dumps(value) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--existing-prefill-pid",
        type=int,
        help="Continue this already-running initial launch only.",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.existing_prefill_pid and (root / "prefill/timing.json").exists():
        raise FileExistsError("Use a fresh output root.")
    if (root / "ltc/matrix_summary.json").exists() or (
        root / "memory/matrix_summary.json"
    ).exists():
        raise FileExistsError(
            "Use per-experiment --resume commands to preserve completed reports."
        )
    environment = os.environ.copy()
    environment.update(
        MAX_JOBS="5",
        FLASHINFER_NVCC_THREADS="1",
        OMP_NUM_THREADS="5",
        PYTHONPATH=str(REPO),
    )
    prefill = [
        str(PYTHON),
        "benchmarks/bench_cute_dsl_localized_mla_prefill.py",
        "--device",
        "0",
        "--expected-partition-sm-counts",
        "74",
        "74",
        "--seqlen-qs",
        "128",
        "--batch-sizes",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "--seqlen-ks",
        "512",
        "1024",
        "2048",
        "4096",
        "8192",
        "16384",
        "32768",
        "65536",
        "131072",
        "262144",
        "524288",
        "1008576",
        "--no-capacity-point",
        "--paired-warmups",
        "20",
        "--timing-warmup-ms",
        "500",
        "--timing-repeat-ms",
        "1000",
        "--timing-blocks",
        "4",
        "--timing-min-samples",
        "20",
        "--output",
        str(root / "prefill/timing.json"),
    ]
    sources = [str(OLD / f"decode/sq{sq}/post_flops.json") for sq in (1, 4)] + [
        str(root / "prefill/timing.json")
    ]
    commands = {"prefill": prefill}
    for experiment in ("ltc", "memory"):
        commands[experiment] = [
            str(PYTHON),
            "benchmarks/profile_cute_dsl_localized_mla_separate.py",
            "--experiment",
            experiment,
            "--device",
            "0",
            "--expected-partition-sm-counts",
            "74",
            "74",
            "--timing-sources",
            *sources,
            "--output-root",
            str(root / experiment),
        ]
    commands["plot_prefill"] = [
        str(PYTHON),
        "benchmarks/plot_cute_dsl_localized_mla_prefill.py",
        "--timing",
        str(root / "prefill/timing.json"),
        "--output-dir",
        str(root / "prefill/figures"),
    ]
    write(root / "execution_commands.json", commands)
    scripts = [
        "benchmarks/bench_cute_dsl_localized_mla_prefill.py",
        "benchmarks/bench_cute_dsl_localized_mla.py",
        "benchmarks/localized_mla_prefill_benchmark.py",
        "benchmarks/profile_cute_dsl_localized_mla_separate.py",
        "benchmarks/profile_cute_dsl_localized_mla_ltc_target.py",
        "benchmarks/profile_cute_dsl_localized_mla_memory.py",
        "benchmarks/profile_cute_dsl_localized_mla_boundary.py",
    ]
    metadata = {
        "started_at": now(),
        "initial_resource_sample": sample(root),
        "scripts_sha256": {
            p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in scripts
        },
        "prefill_reference": {
            "path": str(PREFILL_REFERENCE),
            "sha256": hashlib.sha256(PREFILL_REFERENCE.read_bytes()).hexdigest(),
        },
        "environment": {
            k: environment[k]
            for k in (
                "MAX_JOBS",
                "FLASHINFER_NVCC_THREADS",
                "OMP_NUM_THREADS",
                "PYTHONPATH",
            )
        },
    }
    write(root / "environment.json", metadata)
    for stage, command in commands.items():
        directory = root / ("prefill" if stage == "plot_prefill" else stage)
        directory.mkdir(parents=True, exist_ok=True)
        status_path = root / (
            "prefill/timing.json"
            if stage == "prefill"
            else f"{stage}/matrix_summary.json"
        )
        print(f"{now()} START {stage}", flush=True)
        if stage == "prefill" and args.existing_prefill_pid:

            def active():
                return Path(f"/proc/{args.existing_prefill_pid}").exists()

            process = None
            log = None
        else:
            log = (
                directory / ("plot.log" if stage == "plot_prefill" else "run.log")
            ).open("w")
            process = subprocess.Popen(
                command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            active = lambda: process.poll() is None
        while active():
            resource = sample(root)
            if status_path.exists():
                doc = json.loads(status_path.read_text())
                count = (
                    len(doc.get("results", []))
                    if stage == "prefill"
                    else sum(c["status"] == "complete" for c in doc["configs"])
                )
                print(
                    f"{now()} {stage}: {count} complete; {resource['gpu']}", flush=True
                )
            time.sleep(30)
        if log:
            log.close()
        if process and process.returncode:
            raise RuntimeError(f"{stage} exited {process.returncode}; see {directory}")
        if stage != "plot_prefill":
            doc = json.loads(status_path.read_text())
            expected = "passed" if stage == "prefill" else "complete"
            if doc["status"] != expected:
                raise RuntimeError(f"{stage} incomplete: {doc['status']}")
        print(f"{now()} FINISHED {stage}", flush=True)
    metadata["finished_at"] = now()
    metadata["final_resource_sample"] = sample(root)
    write(root / "environment.json", metadata)
    print("All experiments collected.", flush=True)


if __name__ == "__main__":
    main()
