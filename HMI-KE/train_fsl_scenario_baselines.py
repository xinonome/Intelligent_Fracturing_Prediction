from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline import build_dataset, discover_segment_frames, estimate_sample_interval_seconds
from simulator.fsl_scenario_library import SCENARIO_CLASSES, annotate_real_scenarios


def configure_fonts() -> None:
    for path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def find_group_split(labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    classes = set(np.unique(labels))
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(labels, labels, groups))
        score = len(classes & set(labels[train_idx])) + len(classes & set(labels[test_idx]))
        if best is None or score > best[0]:
            best = (score, train_idx, test_idx)
        if score == 2 * len(classes):
            break
    assert best is not None
    return best[1], best[2]


def cap_by_class(indices: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    subset_labels = labels[indices]
    for label in np.unique(subset_labels):
        candidates = indices[subset_labels == label]
        quota = max(1, int(round(max_samples * len(candidates) / len(indices))))
        selected.extend(rng.choice(candidates, size=min(quota, len(candidates)), replace=False).tolist())
    if len(selected) > max_samples:
        selected = rng.choice(np.asarray(selected), size=max_samples, replace=False).tolist()
    return np.asarray(sorted(selected), dtype=int)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = ["future_60s_mean_PL", "future_60s_mean_SB"]
    result: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        result[name] = {
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))),
            "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
        }
    return result


def plot_confusion(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测场景"); ax.set_ylabel("真实场景"); ax.set_title("真实工况场景分类：测试集混淆矩阵")
    fig.colorbar(image, ax=ax, shrink=0.82); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_metrics(metrics: dict, path: Path) -> None:
    labels = list(metrics["per_class"])
    values = [metrics["per_class"][label]["f1"] for label in labels]
    recalls = [metrics["per_class"][label]["recall"] for label in labels]
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, values, width, label="F1", color="#147d78")
    ax.bar(x + width / 2, recalls, width, label="Recall", color="#e0a11a")
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("指标")
    ax.set_title("各真实场景识别能力"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train real-scenario classification and 60-second action baselines.")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Data" / "raw_frac"))
    parser.add_argument("--reference-header-path", default=str(PROJECT_ROOT / "Data" / "raw_frac" / "WITHfiltered2ASSELECTDISTINCTJTHJDFROMhagHAGMARKPOINTWHEREWORKIN_202511211131.xlsx"))
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=60000)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "fsl_scenario_models"))
    args = parser.parse_args()
    configure_fonts()

    frames = discover_segment_frames(args.data_path, args.reference_header_path, "FDBH", "SGSJ", ["SGBY", "PL", "SB", "WORKING_TYPE"], ["WITHfiltered"], args.max_files, args.max_rows_per_file)
    interval = estimate_sample_interval_seconds(frames, "SGSJ", 10.0)
    bundle = build_dataset(frames, ["SGBY", "PL", "SB"], ["PL", "SB"], "SGSJ", max(2, int(round(300 / interval))), max(1, int(round(60 / interval))), "WORKING_TYPE")
    meta = annotate_real_scenarios(bundle.meta)
    labels = meta["real_scenario_class"].to_numpy(dtype=str)
    groups = meta["segment_id"].to_numpy(dtype=str)
    train_idx, test_idx = find_group_split(labels, groups, args.seed)
    train_idx = cap_by_class(train_idx, labels, args.max_train_samples, args.seed)
    active_labels = [label for label in SCENARIO_CLASSES if label in set(labels[test_idx]) or label in set(labels[train_idx])]

    classifier = ExtraTreesClassifier(n_estimators=args.n_estimators, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=args.seed)
    classifier.fit(bundle.x[train_idx], labels[train_idx])
    predicted_labels = classifier.predict(bundle.x[test_idx])
    class_metrics = classification_metrics(labels[test_idx], predicted_labels, active_labels)

    regressor = ExtraTreesRegressor(n_estimators=args.n_estimators, min_samples_leaf=2, n_jobs=-1, random_state=args.seed)
    regressor.fit(bundle.x[train_idx], bundle.y[train_idx])
    predicted_actions = regressor.predict(bundle.x[test_idx])
    action_metrics = regression_metrics(bundle.y[test_idx], predicted_actions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.run_dir) / timestamp; output.mkdir(parents=True, exist_ok=True)
    with (output / "scenario_classifier.pkl").open("wb") as handle:
        pickle.dump(classifier, handle)
    with (output / "action_model.pkl").open("wb") as handle:
        pickle.dump(regressor, handle)
    matrix = confusion_matrix(labels[test_idx], predicted_labels, labels=active_labels)
    plot_confusion(matrix, active_labels, output / "scenario_confusion_matrix.png")
    plot_metrics(class_metrics, output / "scenario_per_class_metrics.png")
    predictions = pd.DataFrame({"real_scenario": labels[test_idx], "predicted_scenario": predicted_labels, "true_future_flow": bundle.y[test_idx, 0], "predicted_future_flow": predicted_actions[:, 0], "true_future_sand": bundle.y[test_idx, 1], "predicted_future_sand": predicted_actions[:, 1]})
    predictions.to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "task": "FSL real scenario recognition and future-60-second action imitation baselines",
        "scientific_role": "Supervised warm start for the Gymnasium PPO/SAC/HRL policy; not the final reinforcement-learning policy.",
        "state": "previous 300 seconds of SGBY/PL/SB",
        "targets": ["future 60-second scenario", "future 60-second mean PL", "future 60-second mean SB"],
        "split": "grouped by segment, 75% train / 25% test",
        "train_samples": int(len(train_idx)), "test_samples": int(len(test_idx)),
        "train_segments": int(len(set(groups[train_idx]))), "test_segments": int(len(set(groups[test_idx]))),
        "scenario_classifier": class_metrics,
        "action_model": action_metrics,
        "classes": active_labels,
        "confusion_matrix": matrix.tolist(),
        "outputs": {"scenario_classifier": str(output / "scenario_classifier.pkl"), "action_model": str(output / "action_model.pkl"), "predictions": str(output / "test_predictions.csv"), "confusion_matrix": str(output / "scenario_confusion_matrix.png"), "per_class_metrics": str(output / "scenario_per_class_metrics.png")},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(output), "macro_f1": class_metrics["macro_f1"], "accuracy": class_metrics["accuracy"], "action_metrics": action_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
