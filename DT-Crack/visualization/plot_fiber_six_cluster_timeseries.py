"""Plot six-cluster liquid and sand time series from the real fiber text file.

The source file stores stage-level balance fields and cluster-level liquid/sand
fields in one row per time step.  This script keeps those two granularities
separate: the four panels are all cluster-level curves, while the stage-level
balance fields are reported in the metadata and console output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "DT-Crack") not in sys.path:
    sys.path.insert(0, str(ROOT / "DT-Crack"))

from data_fusion.frac_monitor_text_adapter import load_frac_monitor_text


DEFAULT_INPUT = ROOT / "Data" / "3Dfrac" / "光纤本井监测08.txt"
DEFAULT_OUTPUT = ROOT / "outputs" / "app" / "fiber_six_cluster"


def _configure_fonts() -> None:
    """Prefer installed Chinese fonts but keep the plot runnable everywhere."""

    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def _write_csv(path: Path, table) -> None:
    columns = [
        "elapsed_s",
        "elapsed_min",
        "cluster_id",
        "cluster_name",
        "liquid_volume_m3",
        "sand_mass_t",
        "cumulative_liquid_volume_m3",
        "cumulative_sand_mass_t",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in table.sort_values(["step", "cluster_id"]).itertuples(index=False):
            writer.writerow(
                {
                    "elapsed_s": int(row.step) + 1,
                    "elapsed_min": (int(row.step) + 1) / 60.0,
                    "cluster_id": int(row.cluster_id),
                    "cluster_name": f"簇{int(row.cluster_id) + 1}",
                    "liquid_volume_m3": float(row.liquid_volume_m3),
                    "sand_mass_t": float(row.sand_mass_t),
                    "cumulative_liquid_volume_m3": float(
                        row.cumulative_liquid_volume_m3
                    ),
                    "cumulative_sand_mass_t": float(row.cumulative_sand_mass_t),
                }
            )


def _plot(table, output_path: Path, title: str) -> None:
    _configure_fonts()
    clusters = sorted(int(value) for value in table["cluster_id"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0.02, 0.62, len(clusters)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    panels = [
        ("cumulative_liquid_volume_m3", "累计进液量 (m³)", "六簇累计进液量"),
        ("cumulative_sand_mass_t", "累计进砂量 (t)", "六簇累计进砂量"),
        ("liquid_volume_m3", "瞬时进液量 (m³)", "六簇瞬时进液量"),
        ("sand_mass_t", "瞬时进砂量 (t)", "六簇瞬时进砂量"),
    ]

    for ax, (column, ylabel, panel_title) in zip(axes.flat, panels):
        for color, cluster_id in zip(colors, clusters):
            part = table[table["cluster_id"] == cluster_id].sort_values("step")
            x = (part["step"].to_numpy(dtype=float) + 1.0) / 60.0
            y = part[column].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.35,
                label=f"簇{cluster_id + 1}",
            )
        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(ncol=3, fontsize=9, frameon=False, loc="best")

    axes[1, 0].set_xlabel("相对时间 (min)")
    axes[1, 1].set_xlabel("相对时间 (min)")
    fig.text(
        0.5,
        0.015,
        "数据源：光纤本井监测08.txt，第08段；簇级曲线来自分簇液量/砂量。"
        "裂缝均衡程度是阶段级字段，未作为簇级曲线重复绘制。",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tables = load_frac_monitor_text(args.input)
    table = tables.stage_info.copy()
    clusters = sorted(int(value) for value in table["cluster_id"].unique())
    if len(clusters) != 6:
        raise ValueError(f"expected six clusters, found {clusters}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fiber_six_cluster_timeseries.png"
    csv_path = output_dir / "fiber_six_cluster_timeseries.csv"
    summary_path = output_dir / "summary.json"

    _write_csv(csv_path, table)
    _plot(
        table,
        png_path,
        f"第{table['stage'].iloc[0]}段六簇光纤分簇液砂时序",
    )

    summary = {
        "source": str(args.input.resolve()),
        "well_name": str(table["well_name"].iloc[0]),
        "stage": str(table["stage"].iloc[0]),
        "cluster_ids_zero_based": clusters,
        "cluster_names": [f"簇{cluster_id + 1}" for cluster_id in clusters],
        "rows": int(len(table)),
        "points_per_cluster": {
            str(cluster_id + 1): int((table["cluster_id"] == cluster_id).sum())
            for cluster_id in clusters
        },
        "elapsed_seconds": {
            "start": 1,
            "end": int(table["step"].max()) + 1,
        },
        "stage_level_fields": [
            "balance_degree",
            "cumulative_balance_degree",
        ],
        "cluster_level_fields": [
            "liquid_volume_m3",
            "sand_mass_t",
            "cumulative_liquid_volume_m3",
            "cumulative_sand_mass_t",
        ],
        "outputs": {
            "figure": str(png_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
