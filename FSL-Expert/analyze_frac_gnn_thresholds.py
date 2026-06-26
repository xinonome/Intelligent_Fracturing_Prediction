from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets
from frac_gnn.model import TemporalSegmentGNN


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Threshold and PR analysis for frac GNN.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--window-size", type=int, default=6)
    return parser


def load_run_config(run_root: Path) -> SimpleNamespace:
    return SimpleNamespace(**load_json(run_root / "config.json"))


def load_model(run_root: Path, window_size: int, input_dim: int, output_dim: int, args):
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


def collect_probs(model, graphs, device: str) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(graphs, batch_size=512, shuffle=False)
    probs: list[float] = []
    y_true: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            score = torch.softmax(logits, dim=-1)[:, 0]
            probs.extend(score.cpu().numpy().tolist())
            y_true.extend(batch.y.cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(probs)


def plot_pr_curve(run_root: Path, window_size: int, y_true: np.ndarray, y_prob: np.ndarray) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=0)
    ap = average_precision_score((y_true == 0).astype(int), y_prob)
    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall (异常)")
    plt.ylabel("Precision (异常)")
    plt.title(f"PR Curve - Window {window_size}")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path = run_root / f"window_{window_size}_pr_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    y_binary = (y_true == 0).astype(int)
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision_abnormal": float(precision_score(y_binary, y_pred, zero_division=0)),
                "recall_abnormal": float(recall_score(y_binary, y_pred, zero_division=0)),
                "f1_abnormal": float(f1_score(y_binary, y_pred, zero_division=0)),
                "pred_abnormal_count": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_sweep(run_root: Path, window_size: int, sweep: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 5))
    plt.plot(sweep["threshold"], sweep["precision_abnormal"], marker="o", label="Precision")
    plt.plot(sweep["threshold"], sweep["recall_abnormal"], marker="o", label="Recall")
    plt.plot(sweep["threshold"], sweep["f1_abnormal"], marker="o", label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Sweep - Window {window_size}")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path = run_root / f"window_{window_size}_threshold_sweep.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


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
    model = load_model(run_root, window_size, len(prepared.feature_columns), len(prepared.classes), args)
    y_true, y_prob = collect_probs(model, prepared.test_graphs, args.device)

    pr_curve_path = plot_pr_curve(run_root, window_size, y_true, y_prob)
    sweep = threshold_sweep(y_true, y_prob)
    sweep_path = run_root / f"window_{window_size}_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    sweep_plot_path = plot_threshold_sweep(run_root, window_size, sweep)

    best_row = sweep.sort_values(["f1_abnormal", "recall_abnormal"], ascending=False).iloc[0].to_dict()
    summary = {
        "window_size": window_size,
        "abnormal_positive_label_index": 0,
        "average_precision_abnormal": float(average_precision_score((y_true == 0).astype(int), y_prob)),
        "roc_auc_abnormal": float(roc_auc_score((y_true == 0).astype(int), y_prob)),
        "best_threshold_by_f1": best_row,
        "pr_curve_path": str(pr_curve_path),
        "threshold_sweep_csv": str(sweep_path),
        "threshold_sweep_plot": str(sweep_plot_path),
    }
    output_path = run_root / f"window_{window_size}_threshold_analysis.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
