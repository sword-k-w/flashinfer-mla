"""Check actual launch structure with a CUDA activity trace (outside timing)."""

import json
from pathlib import Path

import torch
from benchmarks.sm90_mla_separate_merge import make_uniform_case


def main():
    case = make_uniform_case(2, 1, 1024)
    case.original()
    case.run()
    torch.cuda.synchronize()
    result = {}
    for name, fn in (
        ("fused", case.original),
        ("attention", case.attention),
        ("merge", case.merge),
        ("full", case.run),
    ):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            fn()
            torch.cuda.synchronize()
        path = Path(__file__).with_name(f"separate_merge_{name}_trace.json")
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        launches = [
            e["name"]
            for e in events
            if e.get("cat") == "cuda_runtime" and "Launch" in e["name"]
        ]
        assert len(kernels) == (2 if name == "full" else 1), (name, kernels)
        if name == "fused":
            assert any("Cooperative" in x for x in launches), launches
        else:
            assert not any("Cooperative" in x for x in launches), launches
        if name in ("attention", "full"):
            assert "HopperMergeKernel" not in kernels[0]
            assert ", false>(" in kernels[0]
        if name == "merge":
            assert "HopperMergeKernel" in kernels[0]
        if name == "full":
            assert "HopperMergeKernel" in kernels[1]
        result[name] = {
            "kernel_count": len(kernels),
            "kernel_names": kernels,
            "launch_apis": launches,
        }
        print(name, result[name], flush=True)
    Path(__file__).with_name("separate_merge_launches.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
