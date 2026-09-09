#!/usr/bin/env python3
"""Validate and summarize the equal-work localized MLA timing matrices."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
from plot_cute_dsl_localized_mla import plot_timing_matrix


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def describe(rows):
    ratios = [r["speedup"] for r in rows]
    spreads = [
        max(r[key]) / min(r[key]) - 1
        for r in rows
        for key in ("standard_block_medians_ms", "localized_block_medians_ms")
    ]
    return {
        "count": len(rows),
        "wins": sum(x > 1 for x in ratios),
        "ties": sum(x == 1 for x in ratios),
        "geomean_speedup": geomean(ratios),
        "median_speedup": statistics.median(ratios),
        "min_speedup": min(ratios),
        "max_speedup": max(ratios),
        "max_block_latency_spread": max(spreads),
        "geomean_latency_increase_pct": (1 / geomean(ratios) - 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent)
    root = parser.parse_args().output_root.resolve()
    color_reference = (
        REPO / "reports/localized_mla_b200_74_74_20260905/decode/sq1/post_flops.json"
    )
    reference = json.loads(color_reference.read_text())
    color_span = max(
        0.10, max(abs(row["speedup"] - 1.0) for row in reference["results"])
    )
    summary = {}
    text = [
        "# B200 asymmetry-oblivious MLA decode experiment",
        "",
        "All 72 Sq=1 points passed exact output/LSE checks. Equal work exposes "
        "a substantial regression at B=4/8: for Sk >= 65536, localized latency "
        "is geometrically 30.9%/33.0% higher than standard. These batches "
        "assign 36 tasks to each partition, but P0 has only 35 clusters, "
        "requiring a second persistent wave while P1's 39 clusters need one. "
        "The other batches assign 32/32 tasks and retain long-sequence gains. "
        "This explanation follows the scheduler geometry; no per-partition "
        "timeline profiling was performed.",
        "",
        "Whole-matrix geometric-mean speedup is 0.9776x (2.29% higher latency); "
        "44 wins, 3 ties, 25 losses. Worst point: B=4, Sk=262144, "
        "0.265312 ms standard vs 0.367600 ms localized (0.7217x, 38.55% "
        "higher latency). Maximum per-mode block spread is 3.95%.",
        "",
        "Standard baseline is unchanged. Only localized decode's host-side work "
        "cut changes: `work_p0 = (batch_size * split_kv) // 2`. The device "
        "scheduler consumes this cut; its physical SM map, cluster ranks and "
        "partition-local stride still reflect the real 70/78 SM topology. "
        "KV pages follow the same cut so all scheduled reads remain local.",
        "",
        "BF16 random input (seed 42), H=128, latent/RoPE=512/64, page=64, "
        "PDL off. Sq=1, B=2/4/8/16/32/64, Sk=512 through 1048576. "
        "Each point verifies equal task/page counts and bitwise output/LSE "
        "equality before timing. Allocation, initialization, scatter and "
        "correctness checks are outside timing. Maximum paired KV is 144 GiB.",
        "",
        "Timing reuses the existing cold-L2 Triton CUDA-event benchmark: 20 "
        "paired warmups, four AB/BA/BA/AB blocks, 500 ms warmup and 1000 ms "
        "measurement per mode per block, at least 20 samples. Latency is the "
        "median of four block medians. Speedup = standard / localized; below "
        "1 means the equal-work localized kernel is slower. GPU clocks are "
        "not locked. This experiment compares against standard, not against "
        "a freshly measured SM-proportional localized implementation.",
        "",
        "| Sq | Region | Localized wins | Geomean speedup | Range |",
        "| --- | --- | --- | --- | --- |",
    ]
    all_rows = []
    documents = {}
    for sq in (1,):
        document = json.loads((root / f"sq{sq}/post_flops.json").read_text())
        assert document["status"] == "passed"
        rows = document["results"]
        expected = {
            (b, sk) for b in document["batch_sizes"] for sk in document["seqlen_ks"]
        }
        assert len(rows) == len(expected) == 72
        assert {(r["batch_size"], r["seqlen_k"]) for r in rows} == expected
        for row in rows:
            assert row["resident_partition_clusters"] == [35, 39]
            assert row["owner_work_counts"][0] == row["owner_work_counts"][1]
            assert row["owner_page_counts"][0] == row["owner_page_counts"][1]
            assert row["correctness"]["output_exact"]
            assert row["correctness"]["lse_exact"]
            assert len(row["blocks"]) == 4
            assert math.isclose(
                row["speedup"], row["standard_ms"] / row["localized_ms"]
            )
            assert all(
                block[mode]["sample_count"] >= 20
                for block in row["blocks"]
                for mode in ("standard", "localized")
            )
        plot_timing_matrix(document, root / f"sq{sq}/figures", color_span=color_span)
        regions = {
            "all": rows,
            "Sk >= 16384": [r for r in rows if r["seqlen_k"] >= 16384],
            "Sk >= 65536": [r for r in rows if r["seqlen_k"] >= 65536],
        }
        for batch in document["batch_sizes"]:
            regions[f"B={batch}, Sk >= 65536"] = [
                r for r in rows if r["batch_size"] == batch and r["seqlen_k"] >= 65536
            ]
        summary[f"sq{sq}"] = {
            name: describe(region) for name, region in regions.items()
        }
        for name, stats in summary[f"sq{sq}"].items():
            text.append(
                f"| {sq} | {name} | {stats['wins']}/{stats['count']} | "
                f"{stats['geomean_speedup']:.4f}x | "
                f"{stats['min_speedup']:.4f}–{stats['max_speedup']:.4f}x |"
            )
        all_rows.extend(rows)
        documents[sq] = document

    text += [
        "",
        "## Geometry",
        "",
        "| Sq | B | split_kv | Tasks P0/P1 | Tiles P0/P1 | Waves P0/P1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for sq, document in documents.items():
        for row in document["results"][:6]:
            waves = [
                math.ceil(t / c)
                for t, c in zip(
                    row["owner_tile_counts"],
                    row["resident_partition_clusters"],
                    strict=True,
                )
            ]
            text.append(
                f"| {sq} | {row['batch_size']} | {row['split_kv']} | "
                f"{row['owner_work_counts']} | {row['owner_tile_counts']} | {waves} |"
            )

    for sq, document in documents.items():
        text += [
            "",
            f"## Sq={sq} speedup matrix",
            "",
            "Colors match localized_mla_b200_74_74_20260905 decode Sq=1: "
            f"RdYlGn, centered at 1.0, range {1 - color_span:.6f}–"
            f"{1 + color_span:.6f}x. Values outside this range use the endpoint "
            "colors; cell labels retain the actual measurements.",
            "",
            "![Performance comparison](sq"
            + str(sq)
            + "/figures/performance_comparison.png)",
            "",
            "| Sk / B | 2 | 4 | 8 | 16 | 32 | 64 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        by_shape = {(r["batch_size"], r["seqlen_k"]): r for r in document["results"]}
        for sk in document["seqlen_ks"]:
            values = [
                f"{by_shape[(b, sk)]['speedup']:.4f}" for b in document["batch_sizes"]
            ]
            text.append(f"| {sk} | " + " | ".join(values) + " |")
    fields = [
        "seqlen_q",
        "batch_size",
        "seqlen_k",
        "split_kv",
        "standard_ms",
        "localized_ms",
        "speedup",
        "owner_work_counts",
        "owner_tile_counts",
        "resident_partition_clusters",
        "correctness",
    ]
    with (root / "timing.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(all_rows)
    samples = [
        json.loads(line)
        for line in (root / "resource_samples.jsonl").read_text().splitlines()
    ]
    summary["resources"] = {
        "sample_count": len(samples),
        "min_available_host_gib": min(s["mem_available_kib"] for s in samples) / 2**20,
        "max_aggregate_rss_gib": max(s["aggregate_process_rss_kib"] for s in samples)
        / 2**20,
        "min_tmp_free_gib": min(s["tmp_disk_free_bytes"] for s in samples) / 2**30,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 NVCC_THREADS=1 OMP_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=10.0a \\",
        "  .venv/bin/python reports/asymmetry_oblivious_b200_20260909/run_matrix.py --output-root /path/to/fresh-output",
        ".venv/bin/python reports/asymmetry_oblivious_b200_20260909/summarize.py --output-root /path/to/fresh-output",
        "```",
        "",
        "Raw per-block timing and correctness results: `sq1/post_flops.json`. "
        "Source hashes and arguments: `experiment.json`. All latencies: `timing.csv`. "
        "Resource samples: `resource_samples.jsonl`. Existing correctness suite: "
        "56 passed (`pytest.log`).",
        "",
        "Sq=4 was interrupted at the user's request after Sq=1 completed. "
        "Its partial record is retained in `interrupted_sq4/` and excluded "
        "from all tables and figures. The reproduction runner now runs Sq=1 only.",
    ]
    (root / "README.md").write_text("\n".join(text) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
