#!/usr/bin/env python3
"""Plot partition-localized monolithic MLA prefill timing results."""

from __future__ import annotations

import argparse
import copy
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
    / "localized_mla_prefill_matrix"
    / "iter_000_evaluation"
)
UNDERFILL_COLOR = "#24A8D8"
WAVE_IMBALANCE_COLOR = "#7B2CBF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing",
        type=Path,
        nargs="+",
        default=[DEFAULT_REPORT_DIR / "timing.json"],
        help="One or more compatible completed timing documents to combine.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "figures"
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help="Optionally save the combined timing document as JSON.",
    )
    return parser.parse_args()


def load_completed(path: Path) -> dict:
    document = json.loads(path.read_text())
    if document.get("status") != "passed" or not document.get("results"):
        raise ValueError(f"{path} is not a completed result document")
    return document


def combine_documents(documents: list[dict], sources: list[Path]) -> dict:
    combined = copy.deepcopy(documents[0])
    combined["source_documents"] = [str(path.resolve()) for path in sources]
    seen_shapes = {
        (row["batch_size"], row["seqlen_q"], row["seqlen_k"])
        for row in combined["results"]
    }
    for document in documents[1:]:
        if document["batch_sizes"] != combined["batch_sizes"]:
            raise ValueError("timing documents have different batch axes")
        if document["seqlen_ks"] != combined["seqlen_ks"]:
            raise ValueError("timing documents have different seqlen_k axes")
        for key in ("num_heads", "latent_dim", "rope_dim", "page_size", "dtype"):
            if document["configuration"][key] != combined["configuration"][key]:
                raise ValueError(f"timing documents differ in configuration {key}")
        for seqlen_q in document["seqlen_qs"]:
            if seqlen_q not in combined["seqlen_qs"]:
                combined["seqlen_qs"].append(seqlen_q)
        for row in document["results"]:
            shape = (row["batch_size"], row["seqlen_q"], row["seqlen_k"])
            if shape in seen_shapes:
                raise ValueError(f"duplicate timing shape {shape}")
            seen_shapes.add(shape)
            combined["results"].append(row)
    combined["seqlen_qs"].sort()
    combined["results"].sort(
        key=lambda row: (row["seqlen_q"], row["seqlen_k"], row["batch_size"])
    )
    combined["status"] = "passed"
    return combined


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def save_figure(fig, path: Path, *, tight_rect=None, prearranged: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not prearranged:
        if tight_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_rect)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return path


def plot_speedup_heatmaps(document: dict, output_dir: Path) -> Path:
    batches = document["batch_sizes"]
    seqlen_qs = document["seqlen_qs"]
    seqlen_ks = document["seqlen_ks"]
    rows = document["results"]
    by_shape = {
        (row["seqlen_q"], row["seqlen_k"], row["batch_size"]): row
        for row in rows
    }
    all_speedups = np.array([row["speedup"] for row in rows])
    span = max(0.05, float(np.max(np.abs(all_speedups - 1.0))))
    norm = TwoSlopeNorm(vmin=1.0 - span, vcenter=1.0, vmax=1.0 + span)
    color_map = plt.get_cmap("RdYlGn").copy()

    columns = 2
    panel_rows = math.ceil(len(seqlen_qs) / columns)
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(15.5, max(7.0, panel_rows * (0.72 * len(seqlen_ks) + 2.0))),
        squeeze=False,
    )
    image = None
    for panel_index, seqlen_q in enumerate(seqlen_qs):
        axis = axes.flat[panel_index]
        speedups = np.array(
            [
                [by_shape[(seqlen_q, seqlen_k, batch)]["speedup"] for batch in batches]
                for seqlen_k in seqlen_ks
            ]
        )
        image = axis.imshow(
            speedups, cmap=color_map, norm=norm, aspect="auto"
        )
        axis.set_title(f"Sq={seqlen_q}", fontweight="bold")
        axis.set_xticks(range(len(batches)), [str(value) for value in batches])
        axis.set_yticks(
            range(len(seqlen_ks)), [f"{value:,}" for value in seqlen_ks]
        )
        axis.set_xlabel("Batch size")
        axis.set_ylabel("seqlen_k")
        for row_index, seqlen_k in enumerate(seqlen_ks):
            for column_index, batch in enumerate(batches):
                row = by_shape[(seqlen_q, seqlen_k, batch)]
                fraction = row["localized_theoretical_active_cluster_fraction"]
                waves = row["owner_wave_counts"]
                if fraction < 0.8:
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.48, row_index - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor=UNDERFILL_COLOR,
                            linestyle="--",
                            linewidth=1.8,
                        )
                    )
                if waves[0] != waves[1]:
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.42, row_index - 0.42),
                            0.84,
                            0.84,
                            fill=False,
                            edgecolor=WAVE_IMBALANCE_COLOR,
                            linestyle=":",
                            linewidth=1.8,
                        )
                    )
                speedup = row["speedup"]
                text_color = "white" if abs(speedup - 1.0) >= 0.7 * span else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{speedup:.3f}x\n{row['standard_ms']:.3f}/{row['localized_ms']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=text_color,
                )
    for panel_index in range(len(seqlen_qs), axes.size):
        axes.flat[panel_index].axis("off")

    device = document["environment"]["device"]
    fig.suptitle(
        f"{device} Monolithic MLA Prefill: Standard / Localized Speedup\n"
        "BF16, H=128, D=512+64, split_kv=1; cell text = speedup and standard/localized ms",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.08, right=0.88, bottom=0.11, top=0.86, hspace=0.40, wspace=0.25
    )
    colorbar_axis = fig.add_axes((0.91, 0.15, 0.015, 0.68))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Speedup (>1 means localized is faster)")
    fig.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor=UNDERFILL_COLOR,
                linestyle="--",
                linewidth=1.8,
                label="localized active clusters < 80%",
            ),
            Patch(
                facecolor="none",
                edgecolor=WAVE_IMBALANCE_COLOR,
                linestyle=":",
                linewidth=1.8,
                label="P0/P1 wave-count mismatch",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    path = output_dir / "performance_speedup_matrix.png"
    save_figure(fig, path, prearranged=True)
    print(
        f"cells={len(rows)}, geomean={geometric_mean(all_speedups.tolist()):.6f}x, "
        f"min={all_speedups.min():.6f}x, max={all_speedups.max():.6f}x"
    )
    return path


def plot_speedup_trends(document: dict, output_dir: Path) -> Path:
    batches = document["batch_sizes"]
    seqlen_qs = document["seqlen_qs"]
    rows = document["results"]
    by_shape = {
        (row["seqlen_q"], row["seqlen_k"], row["batch_size"]): row
        for row in rows
    }
    columns = 2
    panel_rows = math.ceil(len(seqlen_qs) / columns)
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(14.5, 4.2 * panel_rows),
        squeeze=False,
        sharex=True,
    )
    for panel_index, seqlen_q in enumerate(seqlen_qs):
        axis = axes.flat[panel_index]
        for batch in batches:
            shape_rows = sorted(
                (
                    row
                    for key, row in by_shape.items()
                    if key[0] == seqlen_q and key[2] == batch
                ),
                key=lambda row: row["seqlen_k"],
            )
            axis.plot(
                [row["seqlen_k"] for row in shape_rows],
                [row["speedup"] for row in shape_rows],
                marker="o",
                linewidth=1.6,
                markersize=4,
                label=f"B={batch}",
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_title(f"Sq={seqlen_q}", fontweight="bold")
        axis.set_xlabel("seqlen_k")
        axis.set_ylabel("standard / localized")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    for panel_index in range(len(seqlen_qs), axes.size):
        axes.flat[panel_index].axis("off")
    fig.suptitle(
        f"{document['environment']['device']} MLA Prefill Localized Speedup Trends",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(
        fig,
        output_dir / "performance_speedup_trends.png",
        tight_rect=(0, 0, 1, 0.96),
    )


def main() -> None:
    args = parse_args()
    documents = [load_completed(path) for path in args.timing]
    document = combine_documents(documents, args.timing)
    if args.combined_output is not None:
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        args.combined_output.write_text(json.dumps(document, indent=2) + "\n")
        print(args.combined_output)
    plot_speedup_heatmaps(document, args.output_dir)
    plot_speedup_trends(document, args.output_dir)


if __name__ == "__main__":
    main()
