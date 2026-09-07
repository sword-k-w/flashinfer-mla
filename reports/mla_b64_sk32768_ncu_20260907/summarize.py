#!/usr/bin/env python3
"""Verify saved NCU CSVs and generate the supplement tables without GPU work."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
import profile_cute_dsl_localized_mla_memory as memory

LTC = "lts__t_requests_srcunit_ltcfabric.sum"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    doc = json.loads((root / "results.json").read_text())
    assert doc["status"] == "complete"
    trials = doc["settings"]["trials"]
    assert len(doc["profiles"]) == 3 * 2 * 2 * trials
    rows, files, keys = [], {}, set()
    for p in doc["profiles"]:
        config = p["config"]
        key = (config["workload"], config["seqlen_q"], p["group"], p["mode"], p["trial"])
        assert key not in keys
        keys.add(key)
        raw_path = Path(p["raw_csv"])
        raw = list(csv.reader(raw_path.read_text().splitlines()))
        i = next(i for i, r in enumerate(raw) if r and r[0] == "ID")
        header, units = raw[i], raw[i + 1]
        data = [r for r in raw[i + 2:] if r and r[0].strip()]
        assert len(data) == 1 and len(data[0]) == len(header)
        values = dict(zip(header, data[0], strict=True))
        unit_map = dict(zip(header, units, strict=True))
        assert values["Kernel Name"] == p["kernel_name"]
        assert int(values["profiler__replayer_passes"]) == p["ncu_replay_passes"]
        for name, record in p["metrics"].items():
            assert memory.parse_number(values[name], name) == record["value"]
            assert unit_map[name] == record["unit"]
        command = json.loads(Path(p["command_file"]).read_text())
        argv = command["argv"]
        assert argv[argv.index("--metrics") + 1].split(",") == doc["settings"]["groups"][p["group"]]
        for flag, expected in (("--cache-control", "all"), ("--clock-control", "boost"),
                               ("--replay-mode", "kernel"), ("--profile-from-start", "off"),
                               ("--launch-count", "1"), ("--warmups", "3"), ("--seed", "42"),
                               ("--batch", "64"), ("--seqlen-k", "32768")):
            assert argv[argv.index(flag) + 1] == expected
        assert "--cold-l2" in argv
        metadata = memory.parse_target_metadata(raw_path.with_suffix(".log").read_text())
        assert metadata == p["target_metadata"]
        memory.validate_target(metadata, config, p["mode"])
        assert metadata["workload"] == config["workload"]
        if p["mode"] == "localized":
            assert metadata["partition_sm_counts"] == doc["expected_partition_sm_counts"]
        if p["group"] == "memory":
            old_l2 = p["l2"].copy()
            memory.derive_metrics(p, allow_inconsistent_l2=True)
            assert old_l2 == p["l2"]
        for path in (raw_path, raw_path.with_suffix(".log"), Path(p["report"]), Path(p["command_file"])):
            assert path.stat().st_size > 0
            files[str(path.relative_to(root))] = memory.file_sha256(path)
        rows.append(dict(workload=config["workload"], sq=config["seqlen_q"], group=p["group"],
                         mode=p["mode"], trial=p["trial"], replay_passes=p["ncu_replay_passes"],
                         **{name: value["value"] for name, value in p["metrics"].items()},
                         l2_hit_rate_usable=p.get("l2", {}).get("hit_rate_usable"),
                         l2_counter_sum_relative_error=p.get("l2", {}).get("counter_sum_relative_error"),
                         raw_csv=str(raw_path.relative_to(root))))
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (root / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    lines = ["# B=64、Sk=32768：NCU 补测结果", "",
             "Baseline = standard；实验组 = localized。Prefill 为 Sq=128 dense。",
             f"每项 {trials} 次独立进程采集；LTC 与 memory 分开运行。LTC 单位为 requests，不换算为 bytes。", "",
             f"| 配置 | baseline LTC 中位数 [min, max] | 实验组 LTC 中位数 [min, max] | LTC 减少 | baseline L2 sector hit rate {trials} 次 (%) | 实验组 L2 {trials} 次 (%) |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    warnings = []
    for workload, sq in (("decode", 1), ("decode", 4), ("prefill", 128)):
        item = dict(workload=workload, sq=sq, modes={})
        for mode in ("standard", "localized"):
            selected = [p for p in doc["profiles"] if p["config"]["workload"] == workload and p["config"]["seqlen_q"] == sq and p["mode"] == mode]
            ltc = [memory.metric_value(p, LTC) for p in selected if p["group"] == "ltc"]
            l2 = [p for p in selected if p["group"] == "memory"]
            assert len(ltc) == len(l2) == trials
            rates = [p["l2"]["hit_rate_pct"] for p in l2]
            item["modes"][mode] = dict(ltc_requests=ltc, ltc_median=statistics.median(ltc),
                l2_hit_rate_pct=rates, l2_median=statistics.median(rates),
                l2_spread_pp=max(rates)-min(rates), l2_usable=[p["l2"]["hit_rate_usable"] for p in l2],
                l2_replay_passes=[p["ncu_replay_passes"] for p in l2])
            for p in l2:
                if not p["l2"]["hit_rate_usable"]:
                    warnings.append(dict(workload=workload, sq=sq, mode=mode, trial=p["trial"],
                                         l2=p["l2"], raw_csv=str(Path(p["raw_csv"]).relative_to(root))))
        a, b = (item["modes"][m] for m in ("standard", "localized"))
        reduction = 100 * (1 - b["ltc_median"] / a["ltc_median"])
        item["ltc_reduction_pct"] = reduction
        def traffic(m):
            return f"{m['ltc_median']:,.0f} [{min(m['ltc_requests']):,.0f}, {max(m['ltc_requests']):,.0f}]"
        def rates(m):
            return ", ".join(f"{v:.2f}" + ("†" if not valid else "") for v, valid in zip(m["l2_hit_rate_pct"], m["l2_usable"], strict=True))
        lines.append(f"| {workload} Sq={sq} | {traffic(a)} | {traffic(b)} | {reduction:.2f}% | {rates(a)} | {rates(b)} |")
        summary.append(item)
    lines += ["", "L2 为 NCU 原始 `lts__t_sector_hit_rate.pct`。† 表示超出 [0,100]% 或 hit+miss 与 total 偏差超过 5%；不截断、不修复、不据此计算有效命中率提升。5% 沿用项目 QA 阈值，并非 NVIDIA 精度规范。", "",
              "## Sector 原始计数", "", "δ = 100 × ((hit+miss)/total − 1)。", "",
              "| 原始 CSV | hit rate (%) | total sectors | hit sectors | miss sectors | δ (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in doc["profiles"]:
        if p["group"] != "memory":
            continue
        path = str(Path(p["raw_csv"]).relative_to(root))
        l2 = p["l2"]
        lines.append(f"| [{path}]({path}) | {l2['hit_rate_pct']:.2f} | {l2['total_sectors']:.0f} | {l2['lookup_hit_sectors']:.0f} | {l2['lookup_miss_sectors']:.0f} | {100*l2['counter_sum_relative_error']:+.2f} |")
    lines += ["", "NCU duration 仅用于诊断，不作为无 profiler 时的性能测量。完整 requested metrics、软件版本、源码 SHA-256、逐次 argv/env 见 results.json 和 *.command.json。", ""]
    (root / "results.md").write_text("\n".join(lines))
    memory.write_json_atomic(root / "summary.json", dict(results=summary, l2_quality_warnings=warnings))
    source_changes = [p for p, digest in doc["source_sha256"].items() if memory.file_sha256(REPO / p) != digest]
    samples = [json.loads(s) for s in (root / "resource_samples.jsonl").read_text().splitlines()]
    busy = []
    for a, b in zip(samples, samples[1:]):
        x, y = ([int(v) for v in s["cpu_jiffies"].split()[1:9]] for s in (a, b))
        delta = [j-i for i, j in zip(x, y, strict=True)]
        if sum(delta) > 0:
            busy.append(100 * (1 - (delta[3]+delta[4]) / sum(delta)))
    memory.write_json_atomic(root / "verification.json", dict(status="passed", profiles_verified=len(rows),
        l2_quality_warning_count=len(warnings), source_changes_since_start=source_changes,
        files_sha256=files, resources=dict(samples=len(samples),
            min_available_memory_gib=min(s["mem_available_kib"] for s in samples)/2**20,
            peak_aggregate_rss_gib=max(s["aggregate_process_rss_kib"] for s in samples)/2**20,
            min_tmp_free_gib=min(s["tmp_disk_free_bytes"] for s in samples)/2**30,
            peak_cpu_busy_pct=max(busy, default=0), peak_load_1m=max(s["load"][0] for s in samples))))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
