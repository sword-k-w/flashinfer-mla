#!/usr/bin/env python3
"""Check isolated NCU target outputs against the existing paired benchmark setup."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from localized_mla_benchmark import PreparedMLACase
from localized_mla_prefill_benchmark import PreparedMLAPrefillCase
from profile_cute_dsl_localized_mla_memory import (
    REPO_ROOT,
    TARGET,
    parse_target_metadata,
    write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    results = []
    for workload, sq in (("decode", 1), ("decode", 4), ("prefill", 128)):
        with tempfile.TemporaryDirectory(prefix="mla-target-validation-") as temporary:
            outputs = {}
            metadata = {}
            for mode in ("standard", "localized"):
                path = Path(temporary) / f"{mode}.pt"
                command = [
                    sys.executable,
                    "-P",
                    str(TARGET),
                    "--workload",
                    workload,
                    "--mode",
                    mode,
                    "--batch",
                    "2",
                    "--seqlen-q",
                    str(sq),
                    "--seqlen-k",
                    "512",
                    "--data-initialization",
                    "random",
                    "--cold-l2",
                    "--expected-partition-sm-counts",
                    "74",
                    "74",
                    "--output-tensors",
                    str(path),
                ]
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log = args.output_root / f"{workload}_sq{sq}_{mode}.log"
                log.write_text(completed.stdout)
                if completed.returncode:
                    raise RuntimeError(f"target validation failed: {log}")
                metadata[mode] = parse_target_metadata(completed.stdout)
                outputs[mode] = torch.load(path, weights_only=True)
            device = torch.device("cuda", 0)
            case = (
                PreparedMLAPrefillCase(2, sq, 512, device=device)
                if workload == "prefill"
                else PreparedMLACase(
                    2, 512, device=device, seq_len_q=sq, initialize_for_correctness=True
                )
            )
            with case:
                case.standard_call()
                torch.cuda.synchronize()
                expected = {
                    "output": case.standard_out.cpu(),
                    "lse": case.standard_lse.cpu(),
                }
                checks = {
                    mode: {
                        key: torch.equal(value, expected[key])
                        for key, value in output.items()
                    }
                    for mode, output in outputs.items()
                }
                if not all(all(check.values()) for check in checks.values()):
                    raise AssertionError(
                        f"{workload} Sq={sq} differs from benchmark: {checks}"
                    )
            torch.cuda.empty_cache()
            results.append(
                {
                    "workload": workload,
                    "seqlen_q": sq,
                    "batch_size": 2,
                    "seqlen_k": 512,
                    "bitwise_equal_to_benchmark": checks,
                    "metadata": metadata,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    write_json_atomic(
        args.output_root / "correctness.json", {"status": "passed", "results": results}
    )


if __name__ == "__main__":
    main()
