from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from frac_gnn.data import discover_segment_frames, normalize_labels_in_frames, save_json, sort_segment_frames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze WORKING_TYPE transition probabilities.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=None)
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--output-dir", default="runs/working_type_transitions")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_frames = discover_segment_frames(
        data_path=args.data_path,
        segment_column=args.segment_column,
        label_column=args.label_column,
        reference_header_path=args.reference_header_path,
        exclude_name_patterns=args.exclude_name_patterns,
    )
    segment_frames = sort_segment_frames(segment_frames, args.time_column)
    segment_frames = normalize_labels_in_frames(segment_frames, args.label_column, args.normal_label)

    transitions: list[dict[str, str]] = []
    state_counts: dict[str, int] = {}
    segment_label_rows: list[dict[str, object]] = []

    for segment_id, frame in segment_frames.items():
        labels = frame[args.label_column].astype(str).tolist()
        counts = pd.Series(labels).value_counts().to_dict()
        segment_label_rows.append(
            {
                "segment_id": segment_id,
                "sample_count": len(labels),
                "labels": " | ".join(sorted(counts.keys())),
                "label_counts": " | ".join(f"{label}:{count}" for label, count in sorted(counts.items())),
            }
        )
        for label, count in counts.items():
            state_counts[label] = state_counts.get(label, 0) + int(count)
        for previous, current in zip(labels[:-1], labels[1:]):
            transitions.append(
                {
                    "segment_id": segment_id,
                    "from_label": previous,
                    "to_label": current,
                }
            )

    if not transitions:
        raise ValueError("No adjacent label transitions were found.")

    transition_df = pd.DataFrame(transitions)
    count_matrix = pd.crosstab(transition_df["from_label"], transition_df["to_label"])
    probability_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).fillna(0.0)

    top_rows = []
    for from_label, row in probability_matrix.iterrows():
        counts = count_matrix.loc[from_label]
        for to_label, probability in row.sort_values(ascending=False).items():
            top_rows.append(
                {
                    "from_label": from_label,
                    "to_label": to_label,
                    "transition_count": int(counts[to_label]),
                    "transition_probability": float(probability),
                }
            )
    top_df = pd.DataFrame(top_rows).sort_values(
        ["from_label", "transition_probability", "transition_count"],
        ascending=[True, False, False],
    )

    count_path = output_dir / "working_type_transition_counts.csv"
    probability_path = output_dir / "working_type_transition_probabilities.csv"
    top_path = output_dir / "working_type_transition_ranked.csv"
    segment_path = output_dir / "segment_label_summary.csv"
    summary_path = output_dir / "transition_summary.json"

    count_matrix.to_csv(count_path, encoding="utf-8-sig")
    probability_matrix.to_csv(probability_path, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_label_rows).to_csv(segment_path, index=False, encoding="utf-8-sig")

    summary = {
        "segment_count": len(segment_frames),
        "transition_count": len(transitions),
        "labels": sorted(state_counts.keys()),
        "state_counts": dict(sorted(state_counts.items())),
        "outputs": {
            "transition_counts": str(count_path),
            "transition_probabilities": str(probability_path),
            "transition_ranked": str(top_path),
            "segment_label_summary": str(segment_path),
        },
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
