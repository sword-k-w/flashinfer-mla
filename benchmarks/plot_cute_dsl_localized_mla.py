#!/usr/bin/env python3
"""Plot the localized MLA timing matrix and NCU LTC-fabric results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = (
    SCRIPT_DIR.parent
    / "reports"
    / "localized_mla_capacity_matrix"
    / "iter_000_evaluation"
)
UNDERFILL_COLOR = "#79C7E3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing", type=Path, default=DEFAULT_REPORT_DIR / "post_flops.json"
    )
    parser.add_argument(
        "--ncu", type=Path, default=DEFAULT_REPORT_DIR / "post_ncu_ltc.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "figures"
    )
    return parser.parse_args()


def load_completed(path: Path) -> dict:
    document = json.loads(path.read_text())
    if document.get("status") != "passed" or not document.get("results"):
        raise ValueError(f"{path} is not a completed result document")
    return document


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def save_figure(fig, path: Path, *, tight_rect=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_timing_matrix(document: dict, output_dir: Path) -> Path:
    rows = document["results"]
    batches = document["batch_sizes"]
    seqlen_ks = document["seqlen_ks"]
    by_shape = {(row["seqlen_k"], row["batch_size"]): row for row in rows}
    speedups = np.full((len(seqlen_ks), len(batches)), np.nan)
    standard_ms = np.full_like(speedups, np.nan)
    localized_ms = np.full_like(speedups, np.nan)
    underfilled = np.zeros_like(speedups, dtype=bool)
    active_clusters = np.zeros_like(speedups, dtype=int)
    seqlen_q = document["configuration"]["seqlen_q"]
    for row_index, seqlen_k in enumerate(seqlen_ks):
        for column_index, batch in enumerate(batches):
            row = by_shape[(seqlen_k, batch)]
            speedups[row_index, column_index] = row["speedup"]
            standard_ms[row_index, column_index] = row["standard_ms"]
            localized_ms[row_index, column_index] = row["localized_ms"]
            fraction = row["localized_theoretical_active_cluster_fraction"]
            underfilled[row_index, column_index] = fraction < 0.8
            active_clusters[row_index, column_index] = row[
                "localized_theoretical_active_clusters"
            ]

    figure_height = max(8.0, 0.72 * len(seqlen_ks) + 3.2)
    fig, axis = plt.subplots(figsize=(13.5, figure_height))
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#E8E8E8")
    finite = speedups[np.isfinite(speedups)]
    span = max(0.10, float(np.max(np.abs(finite - 1.0))))
    image = axis.imshow(
        speedups,
        cmap=color_map,
        norm=TwoSlopeNorm(vmin=1.0 - span, vcenter=1.0, vmax=1.0 + span),
        aspect="auto",
    )
    axis.set_xticks(range(len(batches)), [str(batch) for batch in batches])
    axis.set_yticks(range(len(seqlen_ks)), [f"{seqlen_k:,}" for seqlen_k in seqlen_ks])
    axis.set_xlabel("Batch size")
    axis.set_ylabel("seqlen_k")
    fig.suptitle(
        "B300 FlashInfer Modular MLA: Standard / Localized Speedup",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_title(
        f"decode, Sq={seqlen_q}, BF16, H=128, D=512+64, page=64; "
        "dashed light-blue = localized first-wave active clusters < 80%",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    total_clusters = sum(rows[0]["resident_partition_clusters"])
    for row_index in range(len(seqlen_ks)):
        for column_index in range(len(batches)):
            speedup = speedups[row_index, column_index]
            if underfilled[row_index, column_index]:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=UNDERFILL_COLOR,
                        linestyle="--",
                        linewidth=2.2,
                    )
                )
            label = (
                f"{speedup:.3f}x\n"
                f"{standard_ms[row_index, column_index]:.3f}/"
                f"{localized_ms[row_index, column_index]:.3f} ms"
            )
            if underfilled[row_index, column_index]:
                label += (
                    f"\nlocal {active_clusters[row_index, column_index]}/"
                    f"{total_clusters} clusters"
                )
            text_color = "white" if abs(speedup - 1.0) >= 0.65 * span else "black"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=6.6,
                color=text_color,
            )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Speedup = standard / localized (>1 means localized is faster)")
    axis.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor=UNDERFILL_COLOR,
                linestyle="--",
                linewidth=2.2,
                label="Source-derived localized first-wave underfill",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=9,
    )
    path = output_dir / "performance_comparison.png"
    save_figure(fig, path, tight_rect=(0, 0.04, 1, 0.96))
    print(
        f"timing cells={len(rows)}, geomean={geometric_mean(finite.tolist()):.6f}x, "
        f"min={finite.min():.6f}x, max={finite.max():.6f}x"
    )
    return path


def plot_ltc(document: dict, output_dir: Path) -> tuple[Path, Path]:
    comparisons = document["comparisons"]
    # Sq=1 reports produced before multi-token support did not record this
    # top-level field.  Keep those reports re-plottable.
    seqlen_q = document.get("seqlen_q", comparisons[0].get("seqlen_q", 1))
    fig, axes = plt.subplots(
        1, len(comparisons), figsize=(5 * len(comparisons), 5), squeeze=False
    )
    fig.suptitle(
        "B300 FlashInfer MLA LTC Fabric Traffic: Standard vs Localized "
        f"(B={document['batch_size']}, Sq={seqlen_q})",
        fontsize=14,
        fontweight="bold",
    )
    width = 0.35
    for axis_index, row in enumerate(comparisons):
        axis = axes[0, axis_index]
        standard = row["standard_requests"]
        localized = row["localized_requests"]
        position = np.array([0.0])
        axis.bar(
            position - width / 2,
            [standard / 1e6],
            width,
            label="Standard",
            color="#d62728",
            alpha=0.85,
        )
        axis.bar(
            position + width / 2,
            [localized / 1e6],
            width,
            label="Localized",
            color="#2ca02c",
            alpha=0.85,
        )
        axis.set_xlabel("Batch size")
        axis.set_ylabel("LTC Fabric Requests (M)")
        axis.set_title(f"seqlen_k = {row['seqlen_k']:,}")
        axis.set_xticks(position, [str(row["batch_size"])])
        axis.set_ylim(bottom=0)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=9)
        reduction = 100.0 * row["reduction_fraction"]
        reduction_label = (
            f"{reduction:.2f}%\nreduction"
            if reduction >= 99.9
            else f"{reduction:.1f}%\nreduction"
        )
        axis.annotate(
            reduction_label,
            xy=(width / 2, localized / 1e6),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2ca02c",
            fontweight="bold",
        )
    requests_path = output_dir / "ltc_fabric_requests.png"
    save_figure(fig, requests_path, tight_rect=(0, 0, 1, 0.90))

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        [row["seqlen_k"] for row in comparisons],
        [100.0 * row["reduction_fraction"] for row in comparisons],
        marker="o",
        linewidth=2,
        color="#e45756",
    )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("seqlen_k")
    axis.set_ylabel("LTC-fabric request reduction (%)")
    axis.set_title(
        "Localized MLA fabric-traffic reduction "
        f"(NCU, B={document['batch_size']}, Sq={seqlen_q})"
    )
    axis.grid(True, which="both", alpha=0.25)
    reduction_path = output_dir / "ltc_fabric_reduction.png"
    save_figure(fig, reduction_path)
    return requests_path, reduction_path


def main() -> None:
    args = parse_args()
    timing = load_completed(args.timing)
    ncu = load_completed(args.ncu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_timing_matrix(timing, args.output_dir)
    plot_ltc(ncu, args.output_dir)


if __name__ == "__main__":
    main()
