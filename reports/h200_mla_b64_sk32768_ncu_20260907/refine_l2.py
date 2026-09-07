#!/usr/bin/env python3
"""Repeat L2 with four sector metrics only to reduce collection interference."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run import REPO, memory, monitor, validate_metadata

METRICS = (
    memory.L2_HIT_RATE_METRIC,
    memory.L2_TOTAL_SECTORS_METRIC,
    memory.L2_HIT_SECTORS_METRIC,
    memory.L2_MISS_SECTORS_METRIC,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sq", type=int, nargs="+", default=[4, 128])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    original = json.loads((args.source / "results.json").read_text())
    assert original["status"] == "complete"
    assert args.trials > 0 and set(args.sq) <= {1, 4, 128}
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    hashes = dict(original["source_sha256"])
    hashes[str(Path(__file__).resolve().relative_to(REPO))] = memory.file_sha256(
        Path(__file__)
    )
    assert all(memory.file_sha256(REPO / p) == h for p, h in hashes.items())
    document = dict(
        status="running",
        started_at=memory.utc_now(),
        source=str(args.source.resolve()),
        invocation=sys.argv,
        source_sha256=hashes,
        metrics=METRICS,
        profiles=[],
        reason="Reduce metric set after inconsistent sector counters in full memory collection",
    )
    monitor.sample(root)
    try:
        for sq in args.sq:
            for trial in range(1, args.trials + 1):
                modes = (
                    ("baseline", "compact") if trial % 2 else ("compact", "baseline")
                )
                for mode in modes:
                    previous = next(
                        p
                        for p in original["profiles"]
                        if p["config"]["seqlen_q"] == sq and p["mode"] == mode
                    )
                    command = json.loads(Path(previous["command_file"]).read_text())
                    stem = root / f"sq{sq}_{mode}_{trial:02d}"
                    argv = command["argv"]
                    argv[argv.index("--export") + 1] = str(stem)
                    argv[argv.index("--metrics") + 1] = ",".join(METRICS)
                    memory.write_json_atomic(stem.with_suffix(".command.json"), command)
                    import os

                    env = os.environ.copy()
                    env.update(command["environment"])
                    print(f"START {stem.name}", flush=True)
                    with stem.with_suffix(".log").open("w") as log:
                        proc = subprocess.Popen(
                            argv,
                            cwd=command["cwd"],
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
                    assert status == 0, stem
                    output = stem.with_suffix(".log").read_text()
                    metadata = memory.parse_target_metadata(output)
                    validate_metadata(metadata, sq, mode)
                    assert (
                        "POST_PROFILE_CORRECTNESS bitwise equal output and LSE"
                        in output
                    )
                    for field in ("plan", "scale", "audits", "compact_physical_span"):
                        assert metadata[field] == previous["target_metadata"][field]
                    profile = memory.parse_ncu_report(
                        Path(original["ncu"]),
                        stem.with_suffix(".ncu-rep"),
                        stem.with_suffix(".csv"),
                        METRICS,
                    )
                    assert profile["kernel_name"] == previous["kernel_name"]
                    rate, total, hit, miss = [
                        memory.metric_value(profile, m) for m in METRICS
                    ]
                    assert total > 0 and hit >= 0 and miss >= 0
                    delta = (hit + miss) / total - 1
                    profile.update(
                        sq=sq,
                        mode=mode,
                        trial=trial,
                        target_metadata=metadata,
                        command_file=str(stem.with_suffix(".command.json")),
                        l2=dict(
                            hit_rate_pct=rate,
                            total_sectors=total,
                            lookup_hit_sectors=hit,
                            lookup_miss_sectors=miss,
                            counter_sum_relative_error=delta,
                            hit_rate_usable=0 <= rate <= 100 and abs(delta) <= 0.05,
                        ),
                    )
                    document["profiles"].append(profile)
                    memory.write_json_atomic(root / "results.json", document)
                    monitor.sample(root)
                    print(
                        f"DONE {stem.name}: rate={rate}, counter_error={delta:+.4%}, passes={profile['ncu_replay_passes']}",
                        flush=True,
                    )
        assert all(memory.file_sha256(REPO / p) == h for p, h in hashes.items())
        document["status"] = "complete"
    except BaseException as error:
        document["status"] = "failed"
        document["error"] = repr(error)
        raise
    finally:
        document["finished_at"] = memory.utc_now()
        memory.write_json_atomic(root / "results.json", document)


if __name__ == "__main__":
    main()
