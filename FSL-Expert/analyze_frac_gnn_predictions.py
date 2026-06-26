from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets
from frac_gnn.model import TemporalSegmentGNN


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze frac GNN predictions.")
    parser.add_argument("--run-root", required=True, help="Run directory that contains config.json.")
    parser.add_argument("--window-size", type=int, default=1, help="Window size to analyze.")
    return parser


def load_run_config(run_root: Path) -> SimpleNamespace:
    config = load_json(run_root / "config.json")
    return SimpleNamespace(**config)


def load_model(run_root: Path, window_size: int, input_dim: int, output_dim: int, args) -> TemporalSegmentGNN:
    model = TemporalSegmentGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)
    state_dict = torch.load(
        run_root / f"window_{window_size}" / "model.pt",
        map_location=args.device,
        weights_only=False,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_loader(model, graphs, device: str) -> tuple[list[int], list[int], list[str]]:
    loader = DataLoader(graphs, batch_size=512, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    segment_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(batch.y.cpu().numpy().tolist())
            segment_ids.extend(batch.segment_id)
    return y_true, y_pred, segment_ids


def plot_confusion_matrix(run_root: Path, window_size: int, labels: list[str], matrix: np.ndarray) -> Path:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Test Confusion Matrix - Window {window_size}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    output_path = run_root / f"window_{window_size}_test_confusion_matrix.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def summarize_prediction_bias(
    labels: list[str], y_true: list[int], y_pred: list[int], normal_label: str
) -> dict:
    normal_index = labels.index(normal_label)
    true_normal = sum(1 for value in y_true if value == normal_index)
    pred_normal = sum(1 for value in y_pred if value == normal_index)
    abnormal_true = len(y_true) - true_normal
    abnormal_pred = len(y_pred) - pred_normal

    abnormal_hits = sum(
        1
        for truth, pred in zip(y_true, y_pred)
        if truth != normal_index and pred != normal_index
    )
    abnormal_as_normal = sum(
        1
        for truth, pred in zip(y_true, y_pred)
        if truth != normal_index and pred == normal_index
    )

    return {
        "test_sample_count": len(y_true),
        "true_normal_count": true_normal,
        "true_normal_ratio": true_normal / len(y_true) if y_true else 0.0,
        "pred_normal_count": pred_normal,
        "pred_normal_ratio": pred_normal / len(y_pred) if y_pred else 0.0,
        "true_abnormal_count": abnormal_true,
        "pred_abnormal_count": abnormal_pred,
        "abnormal_predicted_as_abnormal": abnormal_hits,
        "abnormal_predicted_as_normal": abnormal_as_normal,
        "abnormal_non_normal_recall": abnormal_hits / abnormal_true if abnormal_true else 0.0,
    }


def build_segment_distribution(base_prepared, normal_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for segment_id, frame in base_prepared.segment_frames.items():
        labels = frame[base_prepared.label_column].astype(str)
        normal_count = int((labels == normal_label).sum())
        total_count = int(len(frame))
        abnormal_count = total_count - normal_count
        rows.append(
            {
                "segment_id": segment_id,
                "split": next(
                    split_name
                    for split_name, segment_ids in base_prepared.split_manifest.items()
                    if segment_id in segment_ids
                ),
                "total_count": total_count,
                "normal_count": normal_count,
                "abnormal_count": abnormal_count,
                "normal_ratio": normal_count / total_count if total_count else 0.0,
                "abnormal_ratio": abnormal_count / total_count if total_count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "segment_id"]).reset_index(drop=True)


def build_segment_prediction_distribution(
    segment_ids: list[str],
    labels: list[str],
    y_true: list[int],
    y_pred: list[int],
    normal_label: str,
) -> pd.DataFrame:
    normal_index = labels.index(normal_label)
    rows: dict[str, dict] = {}
    for segment_id, truth, pred in zip(segment_ids, y_true, y_pred):
        row = rows.setdefault(
            segment_id,
            {
                "segment_id": segment_id,
                "test_window_count": 0,
                "true_normal_windows": 0,
                "true_abnormal_windows": 0,
                "pred_normal_windows": 0,
                "pred_abnormal_windows": 0,
            },
        )
        row["test_window_count"] += 1
        if truth == normal_index:
            row["true_normal_windows"] += 1
        else:
            row["true_abnormal_windows"] += 1
        if pred == normal_index:
            row["pred_normal_windows"] += 1
        else:
            row["pred_abnormal_windows"] += 1

    frame = pd.DataFrame(rows.values())
    if frame.empty:
        return frame
    frame["pred_normal_ratio"] = frame["pred_normal_windows"] / frame["test_window_count"]
    frame["true_normal_ratio"] = frame["true_normal_windows"] / frame["test_window_count"]
    return frame.sort_values("segment_id").reset_index(drop=True)


def main() -> None:
    parsed = build_arg_parser().parse_args()
    run_root = Path(parsed.run_root)
    args = load_run_config(run_root)
    window_size = parsed.window_size

    base_prepared = prepare_base_datasets(
        data_path=args.data_path,
        label_column=args.label_column,
        segment_column=args.segment_column,
        time_column=args.time_column,
        include_columns=args.include_columns,
        exclude_columns=args.exclude_columns,
        reference_header_path=args.reference_header_path,
        exclude_name_patterns=args.exclude_name_patterns,
        normal_label=args.normal_label,
        binary_normal_abnormal=getattr(args, "binary_normal_abnormal", False),
        abnormal_label=getattr(args, "abnormal_label", "异常"),
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        transfer_ratio=args.transfer_ratio,
    )
    prepared = build_prepared_for_window(base_prepared, window_size)
    model = load_model(
        run_root,
        window_size,
        input_dim=len(prepared.feature_columns),
        output_dim=len(prepared.classes),
        args=args,
    )

    y_true, y_pred, segment_ids = predict_loader(model, prepared.test_graphs, args.device)
    labels = prepared.classes
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    matrix_path = plot_confusion_matrix(run_root, window_size, labels, matrix)

    prediction_bias = summarize_prediction_bias(labels, y_true, y_pred, args.normal_label)
    segment_distribution = build_segment_distribution(base_prepared, args.normal_label)
    segment_distribution_path = run_root / "segment_normal_abnormal_distribution.csv"
    segment_distribution.to_csv(segment_distribution_path, index=False, encoding="utf-8-sig")

    test_segment_pred = build_segment_prediction_distribution(
        segment_ids, labels, y_true, y_pred, args.normal_label
    )
    test_segment_pred_path = run_root / f"window_{window_size}_test_segment_prediction_distribution.csv"
    test_segment_pred.to_csv(test_segment_pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "window_size": window_size,
        "labels": labels,
        "normal_label": args.normal_label,
        "prediction_bias": prediction_bias,
        "confusion_matrix_path": str(matrix_path),
        "segment_distribution_path": str(segment_distribution_path),
        "test_segment_prediction_distribution_path": str(test_segment_pred_path),
    }
    summary_path = run_root / f"window_{window_size}_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
