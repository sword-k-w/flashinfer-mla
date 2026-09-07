"""Validate and plot the complete rectangular H200 matrix (CPU only)."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

MODES = ("baseline", "schedule_only", "compact")
RATIOS = ("compact_vs_baseline", "schedule_vs_baseline", "compact_vs_schedule")
LABELS = {
    "compact_vs_baseline": "A / C: baseline vs. compact",
    "schedule_vs_baseline": "A / B: baseline vs. owner schedule",
    "compact_vs_schedule": "B / C: ordinary vs. compact KV",
}


def geomean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def aggregate(rows):
    return {
        key: {
            "geomean": geomean([r["speedup"][key] for r in rows]),
            "min": min(r["speedup"][key] for r in rows),
            "max": max(r["speedup"][key] for r in rows),
            "above_1_05": sum(r["speedup"][key] > 1.05 for r in rows),
            "below_0_95": sum(r["speedup"][key] < 0.95 for r in rows),
            "count": len(rows),
        }
        for key in RATIOS
    }


def validate(doc, root):
    assert doc["status"] == "passed", "Matrix is not complete"
    axes = (doc["seqlen_qs"], doc["batch_sizes"], doc["seqlen_ks"])
    expected = set(itertools.product(*axes))
    rows = doc["results"]
    actual = {(r["sq"], r["batch"], r["sk"]) for r in rows}
    assert actual == expected and len(rows) == len(expected), "Not a rectangle"
    assert doc["completed_configurations"] == len(expected)
    for path, digest in doc["source_hashes"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    for row in rows:
        assert row["status"] == "passed"
        assert len(row["blocks"]) == doc["timing"]["blocks"]
        assert row["l2_eviction_bytes"] == 256 * 2**20
        for block in row["blocks"]:
            for mode in MODES:
                assert block[mode]["sample_count"] >= doc["timing"]["min_samples"]
                assert 0 < block[mode]["min_ms"] <= block[mode]["median_ms"]
        for audit in row["audits"].values():
            assert audit["sms"] == 132
            assert all(
                audit[key] == 0
                for key in ("duplicate_tasks", "missing_tasks", "wrong_partition_tasks")
            )
        t = row["median_ms"]
        for key, a, b in zip(
            RATIOS,
            ("baseline", "baseline", "schedule_only"),
            ("compact", "schedule_only", "compact"),
            strict=True,
        ):
            assert math.isclose(row["speedup"][key], t[a] / t[b], rel_tol=1e-12)
    return rows


def plot(doc, directory, ratios, name):
    sqs, batches, sks = doc["seqlen_qs"], doc["batch_sizes"], doc["seqlen_ks"]
    lookup = {(r["sq"], r["batch"], r["sk"]): r for r in doc["results"]}
    fig, axes = plt.subplots(
        len(sqs),
        len(ratios),
        figsize=(15 * len(ratios), 3.6 * len(sqs)),
        squeeze=False,
        layout="constrained",
    )
    vals = [r["speedup"][k] for r in doc["results"] for k in ratios]
    span = max(0.15, min(0.75, max(abs(v - 1) for v in vals)))
    norm = TwoSlopeNorm(vmin=1 - span, vcenter=1, vmax=1 + span)
    for i, sq in enumerate(sqs):
        for j, key in enumerate(ratios):
            ax = axes[i, j]
            matrix = np.array(
                [[lookup[sq, b, sk]["speedup"][key] for sk in sks] for b in batches]
            )
            im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
            for y, x in itertools.product(range(len(batches)), range(len(sks))):
                v = matrix[y, x]
                ax.text(
                    x,
                    y,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(v - 1) > span * 0.8 else "black",
                )
            ax.set_xticks(range(len(sks)), [f"{k:,}" for k in sks], fontsize=9)
            ax.set_yticks(range(len(batches)), batches)
            ax.set_xlabel("KV length (tokens)")
            ax.set_ylabel("Batch size")
            ax.set_title(f"Sq = {sq} | {LABELS[key]}")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.7, label="Speedup (>1 is faster)"
    )
    fig.suptitle(
        "H200 SM90 MLA | attention kernel only | cold L2 | BF16, H=128", fontsize=16
    )
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path(__file__).with_name("matrix.json")
    )
    args = parser.parse_args()
    doc = json.loads(args.input.read_text())
    root = Path(__file__).resolve().parents[2]
    rows = validate(doc, root)
    out = args.input.parent
    fields = ["sq", "batch", "sk", "kv_bytes", "compact_physical_span"]
    fields += [f"{m}_ms" for m in MODES] + list(RATIOS)
    with (out / "matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            record = {k: r[k] for k in fields[:5]}
            record.update({f"{m}_ms": r["median_ms"][m] for m in MODES})
            record.update(r["speedup"])
            writer.writerow(record)
    summary = dict(
        configurations=len(rows),
        matrix_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        all=aggregate(rows),
        by_sq={
            str(sq): aggregate([r for r in rows if r["sq"] == sq])
            for sq in doc["seqlen_qs"]
        },
        large_kv_by_sq={
            str(sq): aggregate([r for r in rows if r["sq"] == sq and r["sk"] >= 131072])
            for sq in doc["seqlen_qs"]
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(doc, out, ("compact_vs_baseline",), "speedup")
    plot(doc, out, ("schedule_vs_baseline", "compact_vs_schedule"), "ablation")
    lines = [
        "# H200 SM90 MLA：完整矩阵测量结果",
        "",
        f"**{len(rows)} / {len(rows)}** 个配置通过。最大 KV 长度统一为 **{max(doc['seqlen_ks']):,}**；各 Sq 均为同一张 batch × KV 矩阵。",
        "",
        "A=原始调度/普通 KV，B=owner-local 调度/普通 KV，C=owner-local 调度/compact KV。均使用独立 merge，只计 attention kernel。加速比大于 1 表示后者更快。",
        "",
        f"全矩阵等权几何平均：A/C **{summary['all']['compact_vs_baseline']['geomean']:.3f}×**，A/B **{summary['all']['schedule_vs_baseline']['geomean']:.3f}×**，B/C **{summary['all']['compact_vs_schedule']['geomean']:.3f}×**。",
        "",
        "| Sq | 点数 | A/C 几何平均 | A/B 几何平均 | B/C 几何平均 | A/C 范围 | A/C >1.05 | A/C <0.95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sq, group in summary["by_sq"].items():
        c, s, a = [group[k] for k in RATIOS]
        lines.append(
            f"| {sq} | {c['count']} | {c['geomean']:.3f}× | {s['geomean']:.3f}× | {a['geomean']:.3f}× | {c['min']:.3f}–{c['max']:.3f}× | {c['above_1_05']} | {c['below_0_95']} |"
        )
    lines += [
        "",
        "各配置等权重；几何平均不是某种实际请求分布的端到端收益。±5% 仅用于描述点数，不是统计置信界。JSON 保留每个 block 的样本数及 p20/p80。",
        "",
        "![完整矩阵](speedup.png)",
        "",
        "## 同一最大 KV 长度的边界列",
        "",
        "| Sq | B | A (ms) | B (ms) | C (ms) | A/C |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["sk"] == max(doc["seqlen_ks"]):
            t = r["median_ms"]
            lines.append(
                f"| {r['sq']} | {r['batch']} | {t['baseline']:.6f} | {t['schedule_only']:.6f} | {t['compact']:.6f} | {r['speedup']['compact_vs_baseline']:.3f}× |"
            )
    lines += ["", "## 结果解释", ""]
    if all(
        group["compact_vs_baseline"]["geomean"] < 1
        for group in summary["by_sq"].values()
    ):
        lines.append(
            "本轮已实现并验证 compact KV 的真实加载，但三个 Sq 子矩阵的总体几何平均均低于 1，未取得整体加速。个别配置的调度改善没有形成普遍的 compact 收益。"
        )
        lines.append("")
    if summary["all"]["compact_vs_schedule"]["geomean"] < 1:
        lines.append(
            "固定 owner-local 调度后，compact 相对普通 KV 的全矩阵几何平均也低于 1。这是存储布局和地址计算变化合在一起的净结果；本轮没有采集硬件计数器，不能据此确定某条指令或某一级内存是退化原因。"
        )
        lines.append("")
    for sq in doc["seqlen_qs"]:
        group = [r for r in rows if r["sq"] == sq]
        split = sum(r["plan"]["split_work_count"] > 0 for r in group)
        lines.append(
            f"- Sq={sq}：{split} 个配置包含 split work，{len(group) - split} 个配置只有 direct work。三条路径使用相同原始 plan。"
        )
    lines += [
        "",
        "[调度/存储消融图](ablation.png) · [全部时间 CSV](matrix.csv) · [原始 JSON](matrix.json) · [汇总 JSON](summary.json) · [实现及测量口径](README.md)",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
