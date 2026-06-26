from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot frac GNN training results.")
    parser.add_argument("--run-root", required=True, help="Path to a frac_gnn run directory.")
    return parser


def plot_window_comparison(run_root: Path, summary: dict) -> Path:
    window_results = summary["window_results"]
    windows = sorted(int(key) for key in window_results.keys())
    val_f1 = [window_results[str(window)]["val_macro_f1"] for window in windows]
    test_f1 = [window_results[str(window)]["test_macro_f1"] for window in windows]
    transfer_f1 = [window_results[str(window)]["transfer_macro_f1"] for window in windows]

    plt.figure(figsize=(8, 5))
    plt.plot(windows, val_f1, marker="o", label="Val Macro F1")
    plt.plot(windows, test_f1, marker="o", label="Test Macro F1")
    plt.plot(windows, transfer_f1, marker="o", label="Transfer Macro F1")
    plt.xticks(windows)
    plt.xlabel("Window Size")
    plt.ylabel("Macro F1")
    plt.title("Frac GNN Window Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path = run_root / "window_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_window_metrics_overview(run_root: Path, summary: dict) -> Path:
    windows = sorted(int(key) for key in summary["window_results"].keys())
    test_accuracy = []
    test_macro_f1 = []
    val_accuracy = []
    val_macro_f1 = []

    for window in windows:
        metrics = load_json(run_root / f"window_{window}" / "metrics.json")
        test_accuracy.append(metrics["metrics"]["test"]["accuracy"])
        test_macro_f1.append(metrics["metrics"]["test"]["macro_f1"])
        val_accuracy.append(metrics["metrics"]["val"]["accuracy"])
        val_macro_f1.append(metrics["metrics"]["val"]["macro_f1"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(windows, test_accuracy, marker="o", linewidth=2, label="Test Accuracy")
    axes[0].plot(windows, val_accuracy, marker="o", linewidth=2, label="Val Accuracy")
    axes[0].set_title("Accuracy by Window Size")
    axes[0].set_xlabel("Window Size")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(windows)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(windows, test_macro_f1, marker="o", linewidth=2, label="Test Macro F1")
    axes[1].plot(windows, val_macro_f1, marker="o", linewidth=2, label="Val Macro F1")
    axes[1].set_title("Macro F1 by Window Size")
    axes[1].set_xlabel("Window Size")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xticks(windows)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    output_path = run_root / "window_metrics_overview.png"
    fig.suptitle("Frac GNN Metrics Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_loss_curves_grid(run_root: Path, summary: dict) -> Path:
    windows = sorted(int(key) for key in summary["window_results"].keys())
    cols = 3
    rows = int(np.ceil(len(windows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows), squeeze=False)

    for idx, window in enumerate(windows):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        metrics = load_json(run_root / f"window_{window}" / "metrics.json")
        history = metrics.get("history", [])
        epochs = [entry["epoch"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        val_loss = [entry["val_loss"] for entry in history]

        ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
        ax.plot(epochs, val_loss, marker="o", linewidth=2, label="Val Loss")
        ax.set_title(f"Window {window}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        if epochs:
            ax.set_xticks(epochs)
        ax.legend()

    total_axes = rows * cols
    for idx in range(len(windows), total_axes):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis("off")

    output_path = run_root / "loss_curves_grid.png"
    fig.suptitle("Frac GNN Train/Val Loss Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_test_class_report(run_root: Path, metrics: dict) -> Path:
    report = metrics["reports"]["test"]
    class_rows = []
    for key, value in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(value, dict):
            continue
        class_rows.append((key, value["support"], value["f1-score"]))

    class_rows.sort(key=lambda item: item[0])
    labels = [row[0] for row in class_rows]
    supports = [row[1] for row in class_rows]
    f1_scores = [row[2] for row in class_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(labels, supports, color="#4C78A8")
    axes[0].set_title("Test Class Support")
    axes[0].set_xlabel("Encoded Class")
    axes[0].set_ylabel("Support")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, f1_scores, color="#F58518")
    axes[1].set_title("Test Class F1")
    axes[1].set_xlabel("Encoded Class")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    output_path = run_root / "window_1_test_class_report.png"
    fig.suptitle("Frac GNN Test Report (Window 1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    summary = load_json(run_root / "summary.json")
    metrics = load_json(run_root / "window_1" / "metrics.json")

    outputs = {
        "window_comparison": str(plot_window_comparison(run_root, summary)),
        "window_metrics_overview": str(plot_window_metrics_overview(run_root, summary)),
        "loss_curves_grid": str(plot_loss_curves_grid(run_root, summary)),
        "window_1_test_class_report": str(plot_test_class_report(run_root, metrics)),
    }
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
