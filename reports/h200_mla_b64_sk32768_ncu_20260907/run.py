#!/usr/bin/env python3
"""Collect separate LTC and sector L2 reports for B64/Sk32768 on H200."""

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
TARGET = Path(__file__).with_name("target.py")
LTC = "lts__t_requests_srcunit_ltcfabric.sum"
GROUPS = {"ltc": (LTC,), "memory": memory.MEMORY_METRICS}


def validate_metadata(metadata, sq, mode):
    expected = dict(
        mode=mode,
        batch_size=64,
        seqlen_q=sq,
        seqlen_k=32768,
        device="NVIDIA H200",
        partition_sm_counts=[66, 66],
        audit=False,
        warmup_launches=3,
        profiled_launches=1,
        dtype="bfloat16",
        causal=False,
        heads=128,
        latent_dim=512,
        rope_dim=64,
        page_size=64,
        seed=42,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Invalid target {key}: {metadata.get(key)} != {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    overrides = dict(
        MAX_JOBS="4",
        FLASHINFER_NVCC_THREADS="1",
        OMP_NUM_THREADS="4",
        FLASHINFER_CUDA_ARCH_LIST="9.0a",
        PYTHONPATH=str(REPO),
    )
    env = os.environ.copy()
    env.update(overrides)
    ncu = memory.resolve_ncu(None)
    previous = json.loads(
        (REPO / "reports/h200_partition_stages34/source_manifest.json").read_text()
    )
    sources = {REPO / p for p in previous["files"]}
    sources.update([Path(__file__), TARGET, Path(memory.__file__), MONITOR])
    hashes = {str(p.relative_to(REPO)): memory.file_sha256(p) for p in sorted(sources)}
    for name, digest in previous["files"].items():
        if hashes[name] != digest:
            raise RuntimeError(f"Source differs from previous H200 experiment: {name}")
    manifest = dict(
        status="running",
        started_at=memory.utc_now(),
        commit=memory.git_revision(),
        invocation=sys.argv,
        cwd=str(REPO),
        python=sys.version,
        platform=platform.platform(),
        packages={
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "triton",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
                "numpy",
            )
        },
        environment=overrides,
        ncu_version=memory.ncu_version(ncu),
        ncu=str(ncu),
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,uuid,name,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        source_sha256=hashes,
        previous_h200_source_hashes_match=True,
        settings=dict(
            batch_size=64,
            seqlen_k=32768,
            seqlen_q=[1, 4, 128],
            modes=["baseline", "compact"],
            trials=args.trials,
            groups=GROUPS,
            boundary="one attention kernel; excludes merge, audit, eviction, setup",
            split="unchanged SM90 planner, same plan for baseline and compact",
            cache_control="all",
            clock_control="boost",
            replay_mode="kernel",
        ),
        initial_resources=monitor.sample(root),
        profiles=[],
    )
    memory.write_json_atomic(root / "results.json", manifest)
    try:
        memory.validate_metrics(ncu, 0, (LTC, *memory.MEMORY_METRICS))
        for sq in (1, 4, 128):
            workload = "prefill" if sq == 128 else "decode"
            for group, metrics in GROUPS.items():
                for trial in range(1, args.trials + 1):
                    modes = (
                        ["baseline", "compact"]
                        if trial % 2
                        else ["compact", "baseline"]
                    )
                    for mode in modes:
                        directory = root / f"{workload}_sq{sq}" / group
                        directory.mkdir(parents=True, exist_ok=True)
                        stem = directory / f"{mode}_{trial:02d}"
                        cmd = [
                            str(ncu),
                            "--export",
                            str(stem),
                            "--devices",
                            "0",
                            "--profile-from-start",
                            "off",
                            "--launch-count",
                            "1",
                            "--kernel-name-base",
                            "demangled",
                            "--cache-control",
                            "all",
                            "--clock-control",
                            "boost",
                            "--replay-mode",
                            "kernel",
                            "--metrics",
                            ",".join(metrics),
                            sys.executable,
                            "-P",
                            str(TARGET),
                            "--mode",
                            mode,
                            "--seqlen-q",
                            str(sq),
                        ]
                        memory.write_json_atomic(
                            stem.with_suffix(".command.json"),
                            dict(argv=cmd, environment=overrides, cwd=str(REPO)),
                        )
                        print(
                            f"START {workload}_sq{sq}/{group}/{stem.name}", flush=True
                        )
                        started = time.monotonic()
                        with stem.with_suffix(".log").open("w") as log:
                            proc = subprocess.Popen(
                                cmd,
                                cwd=REPO,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                            while True:
                                try:
                                    status = proc.wait(timeout=15)
                                    break
                                except subprocess.TimeoutExpired:
                                    monitor.sample(root)
                        if status:
                            raise RuntimeError(f"NCU failed ({status}): {stem}.log")
                        output = stem.with_suffix(".log").read_text()
                        metadata = memory.parse_target_metadata(output)
                        validate_metadata(metadata, sq, mode)
                        if (
                            "POST_PROFILE_CORRECTNESS bitwise equal output and LSE"
                            not in output
                        ):
                            raise RuntimeError("Post-profile correctness check missing")
                        profile = memory.parse_ncu_report(
                            ncu,
                            stem.with_suffix(".ncu-rep"),
                            stem.with_suffix(".csv"),
                            metrics,
                        )
                        kernel = profile["kernel_name"]
                        if (
                            "BatchMLA" not in kernel
                            or "Hopper" not in kernel
                            or "Merge" in kernel
                        ):
                            raise RuntimeError(f"Unexpected kernel: {kernel}")
                        profile.update(
                            config=dict(
                                workload=workload,
                                batch_size=64,
                                seqlen_q=sq,
                                seqlen_k=32768,
                            ),
                            group=group,
                            mode=mode,
                            trial=trial,
                            elapsed_seconds=time.monotonic() - started,
                            target_metadata=metadata,
                            command_file=str(stem.with_suffix(".command.json")),
                        )
                        if group == "memory":
                            memory.derive_metrics(profile, allow_inconsistent_l2=True)
                        elif memory.metric_value(profile, LTC) < 0:
                            raise RuntimeError("Negative LTC requests")
                        manifest["profiles"].append(profile)
                        memory.write_json_atomic(root / "results.json", manifest)
                        metric = LTC if group == "ltc" else memory.L2_HIT_RATE_METRIC
                        print(
                            f"DONE {stem.name}: {metric}={memory.metric_value(profile, metric)} ({profile['ncu_replay_passes']} passes)",
                            flush=True,
                        )
                        monitor.sample(root)
        manifest["source_hashes_unchanged"] = all(
            memory.file_sha256(REPO / p) == h for p, h in hashes.items()
        )
        if not manifest["source_hashes_unchanged"]:
            raise RuntimeError("Source files changed during profiling")
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
