#!/usr/bin/env python3
"""Run the existing H200 MLA checks with an isolated, MLA-enabled FA3 build."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa3-repo", type=Path, default=Path("/workspace/vllm-fa"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = args.fa3_repo.resolve()
    python = repo / ".venv/bin/python"
    extension = here / "fa3_lib"
    if not list(extension.glob("flash_attn_3_cuda*.so")):
        raise RuntimeError("Build the focused FA3 extension first; see README.md")
    env = os.environ.copy()
    env.update(MAX_JOBS="4", NVCC_THREADS="1", OMP_NUM_THREADS="4")
    # Existing runners prepend hopper/ to sys.path. Preload this separate build
    # so their imports cannot silently select the older in-place extension.
    launcher = (
        "import sys, runpy; import torch; "
        "sys.path.insert(0, sys.argv.pop(1)); "
        "import flash_attn_3_cuda; "
        "print('FA3_EXTENSION', flash_attn_3_cuda.__file__, flush=True); "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    cases = [
        ("fa3_baseline", "mla_baseline/run_mla_baseline.py", 2, 1024),
        (
            "fa3_owner_split_compact",
            "mla_partition_aware/run_partition_aware_mla.py",
            2,
            1024,
        ),
        # The upstream reference repeats KV across all 128 query heads. Keep
        # this case small enough for its FP32 copies plus the 64-GiB arena.
        (
            "fa3_owner_split_compact_b16_sk4096",
            "mla_partition_aware/run_partition_aware_mla.py",
            16,
            4096,
        ),
    ]
    records = []
    for name, script, batch, seqlen in cases:
        argv = [
            str(python),
            "-c",
            launcher,
            str(extension),
            str(repo / "hopper/exp_h200" / script),
            "--device",
            "0",
            "--batch-size",
            str(batch),
            "--seqlen-k",
            str(seqlen),
            "--output",
            str(output / f"{name}.json"),
        ]
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        records.append({"name": name, "argv": argv, "returncode": result.returncode})
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
