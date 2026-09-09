#!/usr/bin/env python3
"""Reproduce standard versus equal-work localized decode, with exact checks."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO))

import bench_cute_dsl_localized_mla as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("sq*/post_flops.json")):
        raise FileExistsError("Use a fresh output directory")

    # Reuse the existing resource sampler; this performs no GPU workload.
    spec = importlib.util.spec_from_file_location(
        "resource_monitor",
        REPO / "reports/localized_mla_b200_74_74_separate_20260905/run_experiments.py",
    )
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monitor.sample(root)
    stop = threading.Event()

    def sample_resources():
        while not stop.wait(30):
            monitor.sample(root)

    worker = threading.Thread(target=sample_resources, daemon=True)
    worker.start()
    original_case = bench.PreparedMLACase
    checks = {}

    class CheckedCase(original_case):
        def __init__(self, *case_args, **kwargs):
            super().__init__(*case_args, **kwargs)
            try:
                cache = self.localized_cache
                assert cache.work_p0 == cache.work_p1, "work is not 1:1"
                assert cache.owner_page_counts[0] == cache.owner_page_counts[1]
                result = self.check_exact()
                assert result["output_exact"] and result["lse_exact"], result
                checks[(self.batch_size, self.seqlen_k)] = result
            except BaseException:
                self.close()
                raise

    original_run_case = bench.run_case

    def checked_run_case(batch_size, seqlen_k, device, case_args):
        row = original_run_case(batch_size, seqlen_k, device, case_args)
        row["correctness"] = checks[(batch_size, seqlen_k)]
        return row

    bench.PreparedMLACase = CheckedCase
    bench.run_case = checked_run_case
    source_paths = [
        "flashinfer/cute_dsl/attention/experimental/localized_mla.py",
        "flashinfer/cute_dsl/attention/mla_decode.py",
        "flashinfer/cute_dsl/attention/scheduler/mla_persistent.py",
        "benchmarks/bench_cute_dsl_localized_mla.py",
        "benchmarks/localized_mla_benchmark.py",
    ]
    metadata = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "work_policy": "equal split-work counts; physical topology retained",
        "baseline": "unchanged standard modular MLA with ordinary cudaMalloc KV",
        "source_sha256": {
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "build_environment": {
            key: os.environ.get(key)
            for key in (
                "MAX_JOBS",
                "FLASHINFER_NVCC_THREADS",
                "NVCC_THREADS",
                "OMP_NUM_THREADS",
                "TORCH_CUDA_ARCH_LIST",
            )
        },
        "commands": [],
    }
    try:
        for sq in (1,):
            output = root / f"sq{sq}" / "post_flops.json"
            sys.argv = [
                "bench_cute_dsl_localized_mla.py",
                "--expected-partition-sm-counts",
                "70",
                "78",
                "--seqlen-q",
                str(sq),
                "--seqlen-ks",
                *[str(512 * 2**i) for i in range(12)],
                "--data-initialization",
                "random",
                "--timing-warmup-ms",
                "500",
                "--timing-repeat-ms",
                "1000",
                "--output",
                str(output),
            ]
            metadata["commands"].append(sys.argv.copy())
            monitor.write(root / "experiment.json", metadata)
            bench.main()
            document = json.loads(output.read_text())
            document["configuration"]["partition_work_policy"] = "equal-1:1"
            monitor.write(output, document)
    finally:
        stop.set()
        worker.join()
        monitor.sample(root)


if __name__ == "__main__":
    main()
