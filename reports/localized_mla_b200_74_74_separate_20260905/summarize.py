#!/usr/bin/env python3
"""Verify and summarize the separately collected B200 74/74 experiments offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LTC = "lts__t_requests_srcunit_ltcfabric.sum"
MEMORY = {
    "gpu__time_duration.avg",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def geomean(values):
    return math.exp(statistics.mean(map(math.log, values)))


def verify_profile(profile, expected):
    assert set(profile["metrics"]) == expected
    command = profile["command"]
    assert set(command[command.index("--metrics") + 1].split(",")) == expected
    for flag, value in [
        ("--cache-control", "all"),
        ("--clock-control", "boost"),
        ("--replay-mode", "kernel"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--cold-l2" in command
    assert command[command.index("--data-initialization") + 1] == "random"
    assert command[command.index("--seed") + 1] == "42"
    metadata = profile["target_metadata"]
    assert metadata["device"] == "NVIDIA B200" and metadata["profiled_launches"] == 1
    if metadata["mode"] == "localized":
        assert metadata["partition_sm_counts"] == [74, 74]
    assert Path(profile["report"]).is_file()
    with Path(profile["raw_csv"]).open() as handle:
        rows = list(csv.reader(handle))
    matches = []
    for i, row in enumerate(rows):
        if row and row[0] == "ID" and expected.issubset(row):
            for value in rows[i + 2 :]:
                if value and value[0].isdigit():
                    matches.append(dict(zip(row, value, strict=True)))
    assert len(matches) == 1
    raw = matches[0]
    if expected == MEMORY:
        assert LTC not in raw
    elif expected == {LTC}:
        assert not MEMORY.intersection(raw)
    for metric in expected:
        assert (
            float(raw[metric].replace(",", "")) == profile["metrics"][metric]["value"]
        )
    assert int(float(raw["profiler__replayer_passes"])) == profile["ncu_replay_passes"]


def plot(root, experiment, configs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for workload, sq in sorted({(c["workload"], c["seqlen_q"]) for c in configs}):
        group = [c for c in configs if (c["workload"], c["seqlen_q"]) == (workload, sq)]
        panels = (
            [("ltc", "LTC fabric requests", "Requests", 1)]
            if experiment == "ltc"
            else [
                ("l2", "L2 hit rate (flagged readings omitted)", "%", 1),
                ("hbm_bytes", "HBM read + write bytes", "GiB", 2**30),
                ("hbm_bandwidth", "HBM achieved bandwidth", "GB/s", 1),
            ]
        )
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(7 * len(panels), 4.8),
            squeeze=False,
            constrained_layout=True,
        )
        x = np.arange(len(group))
        for axis, (key, title, unit, scale) in zip(axes.flat, panels, strict=True):
            for offset, mode, color in [
                (-0.19, "standard", "#5978a5"),
                (0.19, "localized", "#db8748"),
            ]:
                values = []
                for index, config in enumerate(group):
                    p = config["profiles"][mode]
                    if key == "ltc":
                        value = p["ltc_fabric_requests"]
                    elif key == "l2":
                        value = (
                            p["l2"]["hit_rate_pct"]
                            if p["l2"]["hit_rate_usable"]
                            else float("nan")
                        )
                        if not p["l2"]["hit_rate_usable"]:
                            axis.text(
                                index + offset,
                                2,
                                "N/A",
                                ha="center",
                                rotation=90,
                                fontsize=7,
                            )
                    else:
                        value = p["hbm"][
                            "total_bytes" if key == "hbm_bytes" else "total_gb_s"
                        ]
                    values.append(value / scale)
                axis.bar(x + offset, values, width=0.36, label=mode, color=color)
            if key == "ltc":
                axis.set_yscale("log")
            axis.set_xlim(-0.6, len(group) - 0.4)
            axis.set_xticks(
                x,
                [f"B={c['batch_size']}\nSk={c['seqlen_k']:,}" for c in group],
                fontsize=8,
            )
            axis.set_title(title)
            axis.set_ylabel(unit)
            axis.legend()
        fig.suptitle(
            f"B200 74/74 SM: MLA {workload} Sq={sq}, independent {experiment} experiment"
        )
        out = root / experiment / "figures" / f"{workload}_sq{sq}.png"
        out.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    timing = json.loads((root / "prefill/timing.json").read_text())
    old = json.loads(
        (
            REPO
            / "reports/localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json"
        ).read_text()
    )
    assert timing["status"] == "passed" and len(timing["results"]) == 72
    for key in (
        "timing_policy",
        "comparison",
        "configuration",
        "timing",
        "batch_sizes",
        "seqlen_qs",
        "seqlen_ks",
    ):
        assert timing[key] == old[key], key
    shapes = [
        (r["batch_size"], r["seqlen_q"], r["seqlen_k"]) for r in timing["results"]
    ]
    assert len(set(shapes)) == 72
    assert all(r["resident_partition_clusters"] == [37, 37] for r in timing["results"])
    assert all(
        r["correctness"]["output_exact"] and r["correctness"]["lse_exact"]
        for r in timing["results"]
    )
    for row in timing["results"]:
        assert len(row["blocks"]) == 4
        assert [b["order"] for b in row["blocks"]] == timing["timing"][
            "balanced_block_order"
        ]
        assert all(
            b[mode]["sample_count"] >= 20
            for b in row["blocks"]
            for mode in ("standard", "localized")
        )
    spreads = [
        max(b["paired_speedup"] for b in r["blocks"])
        / min(b["paired_speedup"] for b in r["blocks"])
        - 1
        for r in timing["results"]
    ]
    stats = {
        "points": 72,
        "started_at": timing["started_at"],
        "finished_at": timing["finished_at"],
        "speedup_geomean": geomean([r["speedup"] for r in timing["results"]]),
        "speedup_median": statistics.median(r["speedup"] for r in timing["results"]),
        "localized_wins": sum(r["speedup"] > 1 for r in timing["results"]),
        "speedup_min": min(r["speedup"] for r in timing["results"]),
        "speedup_max": max(r["speedup"] for r in timing["results"]),
        "max_block_speedup_spread_pct": 100 * max(spreads),
        "points_with_block_spread_above_10pct": sum(s > 0.1 for s in spreads),
        "old_74_74_speedup_geomean": geomean([r["speedup"] for r in old["results"]]),
    }
    groups, verified = {}, []
    for experiment, expected in [("ltc", {LTC}), ("memory", MEMORY)]:
        d = json.loads((root / experiment / "matrix_summary.json").read_text())
        assert d["status"] == "complete" and len(d["configs"]) == 18
        assert set(d["settings"]["metrics"]) == expected
        passes = []
        for source in d["settings"]["sources"]:
            assert (
                hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
                == source["sha256"]
            )
        for c in d["configs"]:
            assert c["status"] == "complete"
            assert set(c["profiles"]) == {"standard", "localized"}
            for p in c["profiles"].values():
                verify_profile(p, expected)
                verified.append(p["report"])
                passes.append(p["ncu_replay_passes"])
        plot(root, experiment, d["configs"])
        groups[experiment] = {
            "pairs": 18,
            "reports": 36,
            "metrics": sorted(expected),
            "replay_pass_counts": sorted(set(passes)),
            "quality_warnings": d["metric_quality_warnings"],
            "configs": d["configs"],
        }
    assert set(verified[:36]).isdisjoint(verified[36:])
    a = [
        (c["workload"], c["seqlen_q"], c["batch_size"], c["seqlen_k"])
        for c in groups["ltc"]["configs"]
    ]
    b = [
        (c["workload"], c["seqlen_q"], c["batch_size"], c["seqlen_k"])
        for c in groups["memory"]["configs"]
    ]
    assert a == b
    lines = [
        "# B200 74/74 SM：独立重测结果",
        "",
        "**本轮重新测量 prefill Sq=128 dense 性能，并分别采集 LTC 与 L2/memory。**",
        "",
        f"Prefill 性能：72 点，几何平均加速 **{stats['speedup_geomean']:.4f}×**，localized 更快 {stats['localized_wins']}/72 点；最大 block 加速比波动 {stats['max_block_speedup_spread_pct']:.2f}%。",
        f"旧同拓扑参考的几何平均加速为 {stats['old_74_74_speedup_geomean']:.4f}×，仅用于对照，本轮表格全部来自新数据。",
        "",
        f"LTC：36 份独立单指标报告，replay pass 数 {groups['ltc']['replay_pass_counts']}。",
        f"L2/memory：36 份独立十指标报告，replay pass 数 {groups['memory']['replay_pass_counts']}。",
        f"L2 质量检查标记 {len(groups['memory']['quality_warnings'])}/36 条 mode/shape 记录；采集完成不代表这些记录是有效命中率。",
        "",
        "L2 异常定义、两轮对照及单指标后续测量见 [完整记录](L2_HIT_RATE_NOTES.md)。",
        "",
        "## Prefill 性能（新测）",
        "",
        "| B | Sk | standard ms | localized ms | 加速 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in timing["results"]:
        lines.append(
            f"| {r['batch_size']} | {r['seqlen_k']} | {r['standard_ms']:.6f} | {r['localized_ms']:.6f} | {r['speedup']:.4f}× |"
        )
    lines += [
        "",
        "## LTC 独立采集",
        "",
        "| Workload | B | Sk | standard requests | localized requests | 减少 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in groups["ltc"]["configs"]:
        p = c["profiles"]
        lines.append(
            f"| {c['workload']} Sq={c['seqlen_q']} | {c['batch_size']} | {c['seqlen_k']} | {p['standard']['ltc_fabric_requests']:.0f} | {p['localized']['ltc_fabric_requests']:.0f} | {c['comparison']['ltc_reduction_pct']:.2f}% |"
        )
    lines += [
        "",
        "## L2/memory 独立采集",
        "",
        "† 表示原始命中率存在质量警告，不参与有效 L2 命中率差异解释。HBM 比率均为 localized/standard。",
        "",
        "| Workload | B | Sk | standard L2 % | localized L2 % | HBM bytes 比 | HBM bandwidth 比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in groups["memory"]["configs"]:
        p = c["profiles"]
        rates = [
            f"{p[m]['l2']['hit_rate_pct']:.2f}"
            + ("" if p[m]["l2"]["hit_rate_usable"] else "†")
            for m in ("standard", "localized")
        ]
        lines.append(
            f"| {c['workload']} Sq={c['seqlen_q']} | {c['batch_size']} | {c['seqlen_k']} | {rates[0]} | {rates[1]} | {c['comparison']['localized_to_standard_hbm_total_bytes_ratio']:.4f} | {c['comparison']['localized_to_standard_hbm_bandwidth_ratio']:.4f} |"
        )
    lines += [
        "",
        "复现、脚本位置及参数见 [README](README.md)。原始值在各组 `metrics.csv` 和 `profiles/`。",
        "NCU duration 只作为诊断数据；独立性能结果来自 `prefill/timing.json`。",
        "",
    ]
    (root / "results.md").write_text("\n".join(lines))
    write(
        root / "summary.json",
        {
            "prefill_performance": stats,
            "experiments": {
                k: {a: b for a, b in v.items() if a != "configs"}
                for k, v in groups.items()
            },
        },
    )
    environment = json.loads((root / "environment.json").read_text())
    changed_scripts = [
        p
        for p, h in environment["scripts_sha256"].items()
        if hashlib.sha256((REPO / p).read_bytes()).hexdigest() != h
    ]
    kernel_manifest = root / "software_and_kernel_sources.json"
    changed_kernel_sources = None
    if kernel_manifest.exists():
        kernel_sources = json.loads(kernel_manifest.read_text())[
            "kernel_sources_sha256"
        ]
        changed_kernel_sources = [
            p
            for p, h in kernel_sources.items()
            if hashlib.sha256((REPO / p).read_bytes()).hexdigest() != h
        ]
    write(
        root / "verification.json",
        {
            "status": "passed",
            "prefill_protocol_matches_old": True,
            "prefill_topologies_74_74": 72,
            "prefill_output_and_lse_bitwise_equal_pairs": 72,
            "all_prefill_blocks_have_at_least_20_samples_per_mode": True,
            "distinct_main_reports": len(set(verified)),
            "all_metric_values_match_raw_csv": True,
            "same_boundary_shapes_in_both_experiments": True,
            "ltc_and_memory_metrics_separate": True,
            "changed_scripts_since_initial_snapshot": changed_scripts,
            "changed_kernel_sources_since_snapshot": changed_kernel_sources,
            "caveat": "Raw-data integrity checks do not establish accuracy of anomalous L2 metrics.",
        },
    )
    print(json.dumps(stats, indent=2))
    print("L2 flagged records:", len(groups["memory"]["quality_warnings"]))


if __name__ == "__main__":
    main()
