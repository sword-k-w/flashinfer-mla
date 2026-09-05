#!/usr/bin/env python3
"""Regenerate this experiment's result tables from completed JSON artifacts."""

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def timing_summary(path):
    data = json.loads(path.read_text())
    assert data["status"] == "passed" and len(data["results"]) == 72
    assert data["environment"]["device"] == "NVIDIA B200"
    rows = data["results"]
    assert all(row["resident_partition_clusters"] == [37, 37] for row in rows)
    spreads = []
    for row in rows:
        ratios = [
            standard / localized
            for standard, localized in zip(
                row["standard_block_medians_ms"],
                row["localized_block_medians_ms"],
                strict=True,
            )
        ]
        spreads.append(100 * (max(ratios) / min(ratios) - 1))
    capacity = max(rows, key=lambda row: (row["batch_size"], row["seqlen_k"]))
    return {
        "source": str(path),
        "started_at": data["started_at"],
        "finished_at": data["finished_at"],
        "count": len(rows),
        "wins": sum(row["speedup"] > 1 for row in rows),
        "geomean": geometric_mean([row["speedup"] for row in rows]),
        "median": statistics.median(row["speedup"] for row in rows),
        "min": min(row["speedup"] for row in rows),
        "max": max(row["speedup"] for row in rows),
        "long_sk_geomean": geometric_mean(
            [row["speedup"] for row in rows if row["seqlen_k"] >= 65536]
        ),
        "b2_geomean": geometric_mean(
            [row["speedup"] for row in rows if row["batch_size"] == 2]
        ),
        "max_paired_block_spread_pct": max(spreads),
        "cells_with_paired_block_spread_above_10_pct": sum(v > 10 for v in spreads),
        "min_free_hbm_gib": min(row["free_hbm_bytes_after_alloc"] for row in rows)
        / 2**30,
        "capacity": {
            key: capacity[key]
            for key in (
                "batch_size",
                "seqlen_k",
                "standard_ms",
                "localized_ms",
                "speedup",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    root = parser.parse_args().output_root.resolve()
    timings = {
        f"decode_sq{sq}": timing_summary(root / f"decode/sq{sq}/post_flops.json")
        for sq in (1, 4)
    }
    timings["prefill_sq128_reused"] = timing_summary(
        ROOT.parent
        / "localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json"
    )
    counters = json.loads((root / "ncu_boundary/matrix_summary.json").read_text())
    assert counters["status"] == "complete" and len(counters["configs"]) == 18
    validation = json.loads((root / "validation/correctness.json").read_text())
    assert validation["status"] == "passed"
    passes = set()
    for config in counters["configs"]:
        assert config["status"] == "complete"
        for mode, profile in config["profiles"].items():
            assert len(profile["metrics"]) == 11
            assert Path(profile["report"]).is_file()
            assert Path(profile["raw_csv"]).is_file()
            if mode == "localized":
                assert profile["target_metadata"]["partition_sm_counts"] == [74, 74]
            passes.add(profile["ncu_replay_passes"])
    summary = {
        "status": "complete",
        "partition_sm_counts": [74, 74],
        "timings": timings,
        "counter_config_count": 18,
        "ncu_report_count": 36,
        "ncu_replay_passes": sorted(passes),
        "counter_started_at": counters["created_at"],
        "counter_finished_at": counters["finished_at"],
        "validation_status": validation["status"],
        "metric_quality_warnings": counters.get("metric_quality_warnings", []),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# B200 74/74 SM MLA 结果汇总",
        "",
        "全部新增性能/指标实验完成。Prefill Sq=128 性能复用 2026-09-04 的同拓扑 dense 结果。",
        "",
        f"L2 有 {len(counters.get('metric_quality_warnings', []))} 个 mode/shape 读数被标记为不可直接解读；原值保留，见下文。",
        "",
        "L2 异常定义、两轮对照及单指标后续测量见 [完整记录](../localized_mla_b200_74_74_separate_20260905/L2_HIT_RATE_NOTES.md)。",
        "",
        "## 独立性能测量",
        "",
        "速度比为 standard / localized；大于 1 表示 localized 更快。",
        "",
        "| Workload | 点数 | 加速点 | 几何平均 | Sk≥65,536 几何平均 | 最大 block 加速比波动 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in timings.items():
        lines.append(
            f"| {name} | {data['count']} | {data['wins']} | {data['geomean']:.4f}× | {data['long_sk_geomean']:.4f}× | {data['max_paired_block_spread_pct']:.2f}% |"
        )
    lines += [
        "",
        "| Workload | 最大 B/Sk | Standard ms | Localized ms | 速度比 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, data in timings.items():
        row = data["capacity"]
        lines.append(
            f"| {name} | {row['batch_size']}/{row['seqlen_k']:,} | {row['standard_ms']:.4f} | {row['localized_ms']:.4f} | {row['speedup']:.4f}× |"
        )
    lines += [
        "",
        "## NCU 边界点测量",
        "",
        "LTC 列为请求数；L2 为 sector hit rate；HBM 字节比为 localized / standard。",
        "",
        "| Workload / Sq | B | Sk | LTC standard → localized | LTC 减少 | L2 standard → localized | HBM 字节比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in counters["configs"]:
        standard = config["profiles"]["standard"]
        localized = config["profiles"]["localized"]
        comparison = config["comparison"]
        l2_labels = [
            f"{profile['l2']['hit_rate_pct']:.2f}%"
            + ("†" if not profile["l2"]["hit_rate_usable"] else "")
            for profile in (standard, localized)
        ]
        lines.append(
            f"| {config['workload']} / {config['seqlen_q']} | {config['batch_size']} | {config['seqlen_k']:,} | "
            f"{standard['ltc_fabric_requests']:,.0f} → {localized['ltc_fabric_requests']:,.0f} | "
            f"{comparison['ltc_reduction_pct']:.2f}% | "
            f"{l2_labels[0]} → {l2_labels[1]} | "
            f"{comparison['localized_to_standard_hbm_total_bytes_ratio']:.4f}× |"
        )
    lines += [
        "",
        "† 为不可直接解读的 NCU 原始 L2 百分比：超出 [0,100]%，或 "
        "abs((hit+miss)/total−1)>5%。5% 是本实验的质量检查阈值，不是 NVIDIA 精度保证。"
        "异常值未裁剪，hit/(hit+miss) 也不被当作修复值。"
        "详见 [复测记录](ncu_boundary/diagnostics/README.md)。",
        "",
        "完整读写字节数、带宽、利用率、L2 sectors、指标单位和 NCU duration 见 "
        "[指标 CSV](ncu_boundary/boundary_metrics.csv) 与 [原始汇总](ncu_boundary/matrix_summary.json)。",
        "",
        "NCU duration 是 replay/受控时钟下的诊断样本，不能替代上面的独立性能测量。"
        "这里只有 18 个边界点，不是完整硬件指标矩阵。",
        "",
        "## 验证与复现",
        "",
        "- 两组 decode 性能各 72 点，所有分区 cluster 计数均为 [37,37]。",
        "- 18 对 NCU 配置全部完成，36 份报告均包含 11 项指标；localized 元数据全部为 [74,74] SM。",
        f"- NCU 实际 replay pass 数：{sorted(passes)}。",
        "- 三种 workload 的 standard/localized profiling target output/LSE 均与原 benchmark setup 逐位一致，见 [正确性结果](validation/correctness.json)。",
        "- [复现方法及脚本位置](README.md)；[完整复现脚本](reproduce.sh)；[执行命令](execution_commands.json)。",
        "- 在仓库根目录运行 `python reports/localized_mla_b200_74_74_20260905/summarize.py` 可离线重建本表。",
        "",
    ]
    (root / "results.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
