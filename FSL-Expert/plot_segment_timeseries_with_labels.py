from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frac_gnn.data import discover_segment_frames, sort_segment_frames  # noqa: E402


DEFAULT_COLUMNS = ["SGBY", "SB", "PL", "YTND", "LJSL", "ZDCLYL", "ZDJ", "LJYL", "BZJD"]
DEFAULT_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#8CD17D",
    "#B6992D",
]


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", str(value))
    value = re.sub(r"\s+", "_", value.strip())
    return value[:120] or "segment"


def normalize_label(value: object, normal_label: str) -> str:
    if pd.isna(value):
        return normal_label
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return normal_label
    return text


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            data[column] = pd.to_numeric(frame[column], errors="coerce")
    return data


def minmax_normalize(data: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=data.index)
    for column in data.columns:
        series = data[column].astype(float).interpolate(limit_direction="both")
        low = float(series.min(skipna=True)) if series.notna().any() else 0.0
        high = float(series.max(skipna=True)) if series.notna().any() else 1.0
        if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
            normalized[column] = 0.5
        else:
            normalized[column] = (series - low) / (high - low)
    return normalized


def build_label_color_map(labels: list[str], normal_label: str) -> dict[str, str]:
    ordered = [normal_label] + sorted(label for label in set(labels) if label != normal_label)
    return {label: DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index, label in enumerate(ordered)}


def plot_label_strip(ax, labels: list[str], label_colors: dict[str, str]) -> None:
    start = 0
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        if label != current:
            ax.axvspan(start, index, color=label_colors[current], alpha=0.85)
            start = index
            current = label
    ax.axvspan(start, len(labels), color=label_colors[current], alpha=0.85)
    ax.set_xlim(0, len(labels))
    ax.set_yticks([])
    ax.set_ylabel("工况")
    ax.set_title("WORKING_TYPE 工况标签条")


def annotate_label_changes(ax, labels: list[str]) -> None:
    last = None
    for index, label in enumerate(labels):
        if label != last:
            ax.text(index, 1.02, label, rotation=45, ha="left", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())
            last = label


def plot_segment(
    segment_id: str,
    frame: pd.DataFrame,
    columns: list[str],
    label_column: str,
    normal_label: str,
    output_path: Path,
) -> dict:
    data = numeric_frame(frame, columns)
    if data.empty:
        raise ValueError(f"No numeric columns found for segment {segment_id}")
    labels = [normalize_label(value, normal_label) for value in frame[label_column].tolist()]
    label_colors = build_label_color_map(labels, normal_label)
    normalized = minmax_normalize(data)
    x = np.arange(len(frame))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [0.7, 4.3]},
        sharex=True,
    )
    plot_label_strip(axes[0], labels, label_colors)
    annotate_label_changes(axes[0], labels)

    for column in normalized.columns:
        axes[1].plot(x, normalized[column], label=column, linewidth=1.3, alpha=0.9)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("归一化数值")
    axes[1].set_xlabel("段内时间顺序采样点")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=5, fontsize=8, loc="upper right")

    label_counts = pd.Series(labels).value_counts().to_dict()
    title = f"Segment {segment_id} | samples={len(frame)} | labels={', '.join(label_counts.keys())}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    stats = {
        "segment_id": segment_id,
        "sample_count": int(len(frame)),
        "plot_path": str(output_path),
        "label_counts": label_counts,
        "columns": list(data.columns),
    }
    for column in data.columns:
        stats[f"{column}_min"] = float(data[column].min(skipna=True)) if data[column].notna().any() else None
        stats[f"{column}_max"] = float(data[column].max(skipna=True)) if data[column].notna().any() else None
        stats[f"{column}_mean"] = float(data[column].mean(skipna=True)) if data[column].notna().any() else None
    return stats


def write_html_index(rows: list[dict], output_path: Path) -> None:
    items = []
    for row in rows:
        rel = Path(row["plot_path"]).name
        labels = ", ".join(f"{html.escape(str(k))}:{v}" for k, v in row["label_counts"].items())
        items.append(
            f"<section><h2>{html.escape(row['segment_id'])}</h2>"
            f"<p>samples={row['sample_count']} | labels={labels}</p>"
            f"<img src='{html.escape(rel)}' style='max-width:100%; border:1px solid #ccc;'></section>"
        )
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Segment Time Series</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}section{margin-bottom:36px;}</style>"
        "</head><body><h1>压裂分段时序与工况标签可视化</h1>"
        + "\n".join(items)
        + "</body></html>",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot each fracturing segment time series with WORKING_TYPE labels.")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Data" / "raw_frac"))
    parser.add_argument("--reference-header-path", default=str(PROJECT_ROOT / "Data" / "raw_frac" / "FDBH26.xlsx"))
    parser.add_argument(
        "--exclude-name-patterns",
        nargs="*",
        default=[
            "WITHfiltered",
            "便签",
            "FDBH1.1",
            "FDBH1.xlsx",
            "FDBH16",
            "FDBH18",
            "FDBH2",
            "FDBH22",
            "FDBH3",
            "FDBH7",
            "FDBH8",
        ],
    )
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Reports" / "segment_timeseries_working_type"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = discover_segment_frames(
        data_path=args.data_path,
        segment_column=args.segment_column,
        label_column=args.label_column,
        reference_header_path=args.reference_header_path,
        exclude_name_patterns=args.exclude_name_patterns,
    )
    segments = sort_segment_frames(segments, args.time_column)
    rows = []
    for index, (segment_id, frame) in enumerate(segments.items(), start=1):
        if args.max_segments and index > args.max_segments:
            break
        if args.label_column not in frame.columns:
            continue
        plot_path = output_dir / f"{index:03d}_{sanitize_filename(segment_id)}.png"
        try:
            rows.append(plot_segment(segment_id, frame, args.columns, args.label_column, args.normal_label, plot_path))
        except Exception as exc:
            rows.append(
                {
                    "segment_id": segment_id,
                    "sample_count": int(len(frame)),
                    "plot_path": None,
                    "label_counts": {},
                    "columns": [],
                    "error": str(exc),
                }
            )

    summary = {
        "output_dir": str(output_dir),
        "segment_count": len(rows),
        "columns": args.columns,
        "rows": rows,
    }
    (output_dir / "segment_plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "segment_plot_summary.csv", index=False, encoding="utf-8-sig")
    write_html_index([row for row in rows if row.get("plot_path")], output_dir / "index.html")
    print(json.dumps({"output_dir": str(output_dir), "segment_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
