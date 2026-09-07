#!/usr/bin/env python3
"""Supplement B=64, Sk=32768 MLA profiles with separate LTC and sector L2 runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
import profile_cute_dsl_localized_mla_memory as memory

MONITOR = REPO / "reports/localized_mla_b200_74_74_separate_20260905/run_experiments.py"
spec = importlib.util.spec_from_file_location("resource_monitor", MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)

LTC = "lts__t_requests_srcunit_ltcfabric.sum"
GROUPS = {"ltc": (LTC,), "memory": memory.MEMORY_METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--expected-partition-sm-counts", type=int, nargs=2, default=[74, 74])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    overrides = dict(MAX_JOBS="5", FLASHINFER_NVCC_THREADS="1", OMP_NUM_THREADS="5", PYTHONPATH=str(REPO))
    env = os.environ.copy()
    env.update(overrides)
    ncu = memory.resolve_ncu(None)
    command_args = argparse.Namespace(python=Path(sys.executable), device=0, cache_control="all", clock_control="boost")
    sources = set((REPO / "flashinfer/cute_dsl/attention").rglob("*.py"))
    sources.update((REPO / "flashinfer/cute_dsl/attention/experimental").glob("*.cu"))
    sources.update([Path(__file__), Path(memory.__file__), memory.TARGET, MONITOR,
                    REPO / "benchmarks/localized_mla_benchmark.py",
                    REPO / "benchmarks/localized_mla_prefill_benchmark.py"])
    manifest = dict(
        status="running", started_at=memory.utc_now(), commit=memory.git_revision(),
        invocation=sys.argv, cwd=str(REPO), python=sys.version, platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in ("torch", "triton", "nvidia-cutlass-dsl", "apache-tvm-ffi", "numpy")},
        expected_partition_sm_counts=args.expected_partition_sm_counts,
        environment=overrides, ncu_version=memory.ncu_version(ncu), ncu=str(ncu),
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,uuid,name,power.limit", "--format=csv"], text=True),
        source_sha256={str(p.relative_to(REPO)): memory.file_sha256(p) for p in sorted(sources)},
        settings=dict(batch_size=64, seqlen_k=32768, heads=128, latent_dim=512, rope_dim=64,
                      page_size=64, dtype="bfloat16", split_kv=1, enable_pdl=False,
                      data_initialization="random", seed=42, warmups=3, profiled_launches=1,
                      cache_control="all", clock_control="boost", replay_mode="kernel",
                      trials=args.trials, groups=GROUPS, prefill="Sq=128 dense"),
        initial_resources=monitor.sample(root), profiles=[],
    )
    memory.write_json_atomic(root / "results.json", manifest)
    memory.validate_metrics(ncu, 0, (LTC, *memory.MEMORY_METRICS))
    try:
        for workload, sq in (("decode", 1), ("decode", 4), ("prefill", 128)):
            config = dict(workload=workload, batch_size=64, seqlen_q=sq, seqlen_k=32768, split_kv=1)
            for group, metrics in GROUPS.items():
                for trial in range(1, args.trials + 1):
                    modes = ["standard", "localized"] if trial % 2 else ["localized", "standard"]
                    for mode in modes:
                        directory = root / f"{workload}_sq{sq}" / group
                        directory.mkdir(parents=True, exist_ok=True)
                        stem = directory / f"{mode}_{trial:02d}"
                        cmd = memory.profile_command(command_args, ncu, config, mode, stem)
                        cmd[cmd.index("--metrics") + 1] = ",".join(metrics)
                        cmd += ["--workload", workload, "--expected-partition-sm-counts", *map(str, args.expected_partition_sm_counts)]
                        memory.write_json_atomic(stem.with_suffix(".command.json"), dict(argv=cmd, environment=overrides, cwd=str(REPO)))
                        print(f"START {workload}_sq{sq}/{group}/{stem.name}", flush=True)
                        started = time.monotonic()
                        with stem.with_suffix(".log").open("w") as log:
                            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                            while True:
                                try:
                                    status = proc.wait(timeout=15)
                                    break
                                except subprocess.TimeoutExpired:
                                    monitor.sample(root)
                        if status:
                            raise RuntimeError(f"NCU failed ({status}): {stem}.log")
                        metadata = memory.parse_target_metadata(stem.with_suffix(".log").read_text())
                        memory.validate_target(metadata, config, mode)
                        if metadata["workload"] != workload or metadata["device"] != "NVIDIA B200":
                            raise RuntimeError(f"unexpected target: {metadata}")
                        if mode == "localized" and metadata["partition_sm_counts"] != args.expected_partition_sm_counts:
                            raise RuntimeError(f"unexpected topology: {metadata}")
                        profile = memory.parse_ncu_report(ncu, stem.with_suffix(".ncu-rep"), stem.with_suffix(".csv"), metrics)
                        profile.update(config=config, group=group, mode=mode, trial=trial,
                                       elapsed_seconds=time.monotonic() - started, target_metadata=metadata,
                                       command_file=str(stem.with_suffix(".command.json")))
                        if group == "memory":
                            memory.derive_metrics(profile, allow_inconsistent_l2=True)
                        else:
                            if memory.metric_value(profile, LTC) < 0:
                                raise RuntimeError("negative LTC requests")
                        manifest["profiles"].append(profile)
                        memory.write_json_atomic(root / "results.json", manifest)
                        metric = LTC if group == "ltc" else memory.L2_HIT_RATE_METRIC
                        print(f"DONE {stem.name}: {metric}={memory.metric_value(profile, metric)} ({profile['ncu_replay_passes']} passes)", flush=True)
                        monitor.sample(root)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at"] = memory.utc_now()
        manifest["final_resources"] = monitor.sample(root)
        memory.write_json_atomic(root / "results.json", manifest)


if __name__ == "__main__":
    main()
