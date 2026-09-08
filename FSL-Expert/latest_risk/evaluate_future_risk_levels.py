from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from frac_gnn.future_risk_data import indices_for_manifest
from frac_gnn.future_risk_gnn import KnowledgeTemporalRiskGNN
from frac_gnn.sand_risk_data import build_sand_risk_dataset


LABEL_NAMES = ["绿色", "黄色", "红色"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Predict the expert-reviewed green/yellow/red risk level one sample ahead."
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--enhanced-file", required=True)
    result.add_argument("--reference-risk-run", required=True)
    result.add_argument("--output-dir", default="runs/future_risk_levels")
    result.add_argument("--seed", type=int, default=42)
    return result


def build_intervals(enhanced: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]]:
    warnings = pd.read_excel(enhanced, sheet_name="砂堵预警")
    warnings["start"] = pd.to_datetime(warnings["开始时间"], errors="coerce")
    warnings["end"] = pd.to_datetime(warnings["结束时间"], errors="coerce")
    warnings["segment"] = warnings["段号"].astype(str).str.strip()
    warnings["level"] = warnings["预警类型"].map({"砂堵迹象": 1, "砂堵风险": 2})
    warnings = warnings.dropna(subset=["start", "end", "level"])
    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
    for row in warnings.itertuples(index=False):
        result.setdefault(row.segment, []).append((row.start, row.end, int(row.level)))
    return result


def label_at(segment: str, timestamp: pd.Timestamp, intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]]) -> int:
    level = 0
    for start, end, candidate in intervals.get(str(segment).strip(), []):
        if start <= timestamp <= end:
            level = max(level, candidate)
    return level


def gnn_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    report = classification_report(
        truth,
        prediction,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    binary_truth = (truth > 0).astype(int)
    binary_prediction = (prediction > 0).astype(int)
    binary_report = classification_report(
        binary_truth,
        binary_prediction,
        labels=[0, 1],
        target_names=["绿色", "黄+红"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "three_level_accuracy": float(report["accuracy"]),
        "three_level_macro_f1": float(report["macro avg"]["f1-score"]),
        "three_level_report": report,
        "three_level_confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "green_vs_yellow_red_accuracy": float(binary_report["accuracy"]),
        "yellow_red_precision": float(binary_report["黄+红"]["precision"]),
        "yellow_red_recall": float(binary_report["黄+红"]["recall"]),
        "yellow_red_f1": float(binary_report["黄+红"]["f1-score"]),
        "green_vs_yellow_red_confusion_matrix": confusion_matrix(binary_truth, binary_prediction, labels=[0, 1]).tolist(),
    }


def train_knowledge_gnn(
    windows: np.ndarray,
    rule_scores: np.ndarray,
    current_one_hot: np.ndarray,
    target: np.ndarray,
    indices: dict[str, np.ndarray],
    seed: int,
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_rows = windows[indices["train"]].reshape(-1, windows.shape[-1])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(train_rows))
    normalized = scaler.transform(
        imputer.transform(windows.reshape(-1, windows.shape[-1]))
    ).reshape(windows.shape).astype(np.float32)

    def loader(name: str, shuffle: bool) -> DataLoader:
        rows = indices[name]
        data = TensorDataset(
            torch.from_numpy(normalized[rows]).float(),
            torch.from_numpy(rule_scores[rows]).float(),
            torch.from_numpy(current_one_hot[rows]).float(),
            torch.from_numpy(target[rows]).long(),
        )
        return DataLoader(data, batch_size=512, shuffle=shuffle)

    loaders = {name: loader(name, name == "train") for name in indices}
    model = KnowledgeTemporalRiskGNN(
        base_feature_dim=normalized.shape[-1],
        window_size=normalized.shape[1],
        hidden_dim=48,
        dropout=0.2,
        use_rule_nodes=True,
        context_dim=3,
    )
    model.classifier[-1] = nn.Linear(48, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(target[indices["train"]], minlength=3).astype(float)
    weights = np.sqrt(counts.max() / np.maximum(counts, 1.0))
    weights = np.minimum(weights, 5.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    prior_strength = 3.0

    def predict(data_loader: DataLoader) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_windows, batch_rules, batch_current, _ in data_loader:
                logits = model(batch_windows, batch_rules, batch_current)
                logits = logits + prior_strength * batch_current
                outputs.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(outputs)

    best_state = None
    best_score = -1.0
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, 31):
        model.train()
        losses: list[float] = []
        for batch_windows, batch_rules, batch_current, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_windows, batch_rules, batch_current)
            logits = logits + prior_strength * batch_current
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_prediction = predict(loaders["val"])
        score = float(
            classification_report(
                target[indices["val"]],
                validation_prediction,
                labels=[0, 1, 2],
                output_dict=True,
                zero_division=0,
            )["macro avg"]["f1-score"]
        )
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": score})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = {name: predict(data_loader) for name, data_loader in loaders.items()}
    return model, predictions, {
        "history": history,
        "best_validation_macro_f1": best_score,
        "class_weights": weights.tolist(),
        "prior_strength": prior_strength,
        "architecture": "6 time nodes + 3 rule nodes + current rule-state prior; 2-layer GAT, hidden 48",
    }


def main() -> None:
    args = parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    enhanced = data_dir / args.enhanced_file
    reference_run = Path(args.reference_risk_run).resolve()
    reference_metrics = json.loads((reference_run / "metrics.json").read_text(encoding="utf-8"))
    manifest = reference_metrics["split_manifest"]
    dataset = build_sand_risk_dataset(
        data_dir,
        result_glob=enhanced.name,
        horizons_seconds=[60],
        window_size=int(reference_metrics["window_size"]),
        stride=int(reference_metrics["stride"]),
    )
    indices = indices_for_manifest(dataset.segment_groups, manifest)
    intervals = build_intervals(enhanced)

    metadata = dataset.metadata.copy()
    metadata["window_end"] = pd.to_datetime(metadata["window_end"], errors="coerce")
    # Raw data are sampled at about 10 seconds. The target is the reviewed risk
    # level at the next sample, not the current level.
    metadata["target_time"] = metadata["window_end"] + pd.Timedelta(seconds=10)
    current_level = np.asarray(
        [label_at(row.segment, row.window_end, intervals) for row in metadata.itertuples(index=False)],
        dtype=int,
    )
    target = np.asarray(
        [label_at(row.segment, row.target_time, intervals) for row in metadata.itertuples(index=False)],
        dtype=int,
    )
    # Current rule state is available at inference time and is embedded as expert
    # knowledge. It does not use the future target. One-hot encoding keeps the
    # three operational levels explicit for the model.
    current_one_hot = np.eye(3, dtype=np.float32)[current_level]
    model_features = np.column_stack([dataset.features, current_one_hot]).astype(np.float32)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        model_features[indices["train"]],
        target[indices["train"]],
        eval_set=[(model_features[indices["val"]], target[indices["val"]])],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(period=0)],
    )

    result: dict[str, object] = {}
    for split, rows in indices.items():
        prediction = model.predict(model_features[rows]).astype(int)
        truth = target[rows]
        persistence_prediction = current_level[rows]
        report = classification_report(
            truth,
            prediction,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )
        binary_truth = (truth > 0).astype(int)
        binary_prediction = (prediction > 0).astype(int)
        binary_report = classification_report(
            binary_truth,
            binary_prediction,
            labels=[0, 1],
            target_names=["绿色", "黄+红"],
            output_dict=True,
            zero_division=0,
        )
        result[split] = {
            "sample_count": int(len(rows)),
            "label_counts": {LABEL_NAMES[level]: int((truth == level).sum()) for level in range(3)},
            "three_level_accuracy": float(report["accuracy"]),
            "three_level_macro_f1": float(report["macro avg"]["f1-score"]),
            "three_level_report": report,
            "three_level_confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
            "green_vs_yellow_red_accuracy": float(binary_report["accuracy"]),
            "yellow_red_precision": float(binary_report["黄+红"]["precision"]),
            "yellow_red_recall": float(binary_report["黄+红"]["recall"]),
            "yellow_red_f1": float(binary_report["黄+红"]["f1-score"]),
            "green_vs_yellow_red_confusion_matrix": confusion_matrix(binary_truth, binary_prediction, labels=[0, 1]).tolist(),
            "current_rule_state_persistence_accuracy": float((persistence_prediction == truth).mean()),
        }

    gnn, gnn_predictions, gnn_training = train_knowledge_gnn(
        dataset.windows,
        dataset.rule_scores,
        current_one_hot,
        target,
        indices,
        args.seed,
    )
    gnn_result = {
        split: {
            "sample_count": int(len(rows)),
            "label_counts": {LABEL_NAMES[level]: int((target[rows] == level).sum()) for level in range(3)},
            **gnn_metrics(target[rows], gnn_predictions[split]),
        }
        for split, rows in indices.items()
    }

    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "past 60 seconds predict the expert-reviewed green/yellow/red risk level 10 seconds ahead",
        "well": "JH_焦页5-Z6HF",
        "label_definition": {
            "green": "no reviewed warning interval is active at the target time",
            "yellow": "a reviewed sand-warning interval is active at the target time",
            "red": "a reviewed sand-risk interval is active at the target time",
            "execution_zone": "yellow plus red",
        },
        "expert_knowledge_input": "current green/yellow/red rule state at time t, one-hot encoded; no future label is used",
        "split_manifest": manifest,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "lightgbm_baseline": result,
        "knowledge_gnn": gnn_result,
        "gnn_training": gnn_training,
        "limitation": "This predicts the next-sample risk state. It is easier and different from predicting a new event start within the next 60 seconds.",
    }
    torch.save(gnn.state_dict(), run_dir / "knowledge_risk_level_gnn.pt")
    (run_dir / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), "lightgbm_test": result["test"], "knowledge_gnn_test": gnn_result["test"], "gnn_training": gnn_training}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
