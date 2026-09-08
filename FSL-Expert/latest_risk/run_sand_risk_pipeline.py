from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_future_risk_boundaries import boundary_metrics, choose_boundaries
from frac_gnn.future_risk_data import indices_for_manifest, split_groups
from frac_gnn.future_risk_gnn import KnowledgeTemporalRiskGNN
from frac_gnn.sand_risk_data import build_sand_risk_dataset
from train_future_risk_baseline import choose_threshold, metric_dict


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Weakly supervised future sand-risk prediction with knowledge GNN."
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--result-glob", default="*_小条统计_增强版.xlsx")
    result.add_argument("--output-dir", default="runs/sand_risk_pipeline")
    result.add_argument("--horizons-seconds", nargs="*", type=int, default=[60, 180, 300])
    result.add_argument("--window-size", type=int, default=6)
    result.add_argument("--stride", type=int, default=3)
    result.add_argument("--boundary-validation-target", type=float, default=0.97)
    result.add_argument("--minimum-validation-red-count", type=int, default=10)
    result.add_argument("--minimum-validation-red-recall", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--split-unit", choices=["well", "segment"], default="well")
    result.add_argument("--epochs", type=int, default=12)
    result.add_argument("--patience", type=int, default=4)
    result.add_argument("--batch-size", type=int, default=1024)
    result.add_argument("--hidden-dim", type=int, default=48)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--skip-gnn", action="store_true")
    result.add_argument(
        "--model-mode",
        choices=["gnn_main", "ensemble", "tree_main"],
        default="gnn_main",
        help="主预测器：knowledge GNN、融合模型或LightGBM",
    )
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prediction_frame(
    metadata: pd.DataFrame,
    index: np.ndarray,
    targets: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    frame = metadata.iloc[index].reset_index(drop=True).copy()
    frame["target_binary"] = targets[index]
    frame["risk_probability"] = probability
    return frame


def choose_sand_boundaries(
    frame: pd.DataFrame,
    validation_target: float,
    minimum_red_count: int,
    minimum_red_recall: float,
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for low in np.linspace(0.02, 0.48, 47):
        for high in np.linspace(0.52, 0.98, 47):
            candidate = boundary_metrics(frame, float(low), float(high))
            if candidate["decided_accuracy"] < validation_target:
                continue
            if candidate["red_count"] < minimum_red_count:
                continue
            if candidate["red_recall_over_all_positive"] < minimum_red_recall:
                continue
            if best is None or (
                candidate["coverage"],
                candidate["red_recall_over_all_positive"],
                candidate["decided_accuracy"],
            ) > (
                best["coverage"],
                best["red_recall_over_all_positive"],
                best["decided_accuracy"],
            ):
                best = candidate
    return best


def selective_result(
    frame: pd.DataFrame,
    validation_target: float,
    minimum_red_count: int,
    minimum_red_recall: float,
) -> dict[str, object]:
    selected = choose_sand_boundaries(
        frame, validation_target, minimum_red_count, minimum_red_recall
    )
    if selected is None:
        return {"status": "no_feasible_boundaries"}
    return selected


def train_tree(
    features: np.ndarray,
    targets: np.ndarray,
    indices: dict[str, np.ndarray],
    seed: int,
) -> tuple[LGBMClassifier, dict[str, np.ndarray]]:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        features[indices["train"]],
        targets[indices["train"]],
        eval_set=[(features[indices["val"]], targets[indices["val"]])],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(60, verbose=False), log_evaluation(period=0)],
    )
    probabilities = {
        name: model.predict_proba(features[index])[:, 1]
        for name, index in indices.items()
    }
    return model, probabilities


def make_loader(
    windows: np.ndarray,
    rules: np.ndarray,
    context: np.ndarray,
    teacher: np.ndarray,
    targets: np.ndarray,
    index: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(windows[index]).float(),
            torch.from_numpy(rules[index]).float(),
            torch.from_numpy(context[index]).float(),
            torch.from_numpy(teacher[index]).float(),
            torch.from_numpy(targets[index]).float(),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def gnn_predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for windows, rules, context, _, _ in loader:
        logits = model(windows.to(device), rules.to(device), context.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_gnn(
    context_features: np.ndarray,
    windows: np.ndarray,
    rules: np.ndarray,
    targets: np.ndarray,
    teacher_probabilities: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    *,
    hidden_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, object]]:
    train_rows = windows[indices["train"]].reshape(-1, windows.shape[-1])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(train_rows))
    flat = windows.reshape(-1, windows.shape[-1])
    normalized = scaler.transform(imputer.transform(flat)).reshape(windows.shape).astype(np.float32)
    context_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    context_scaler = StandardScaler()
    context_scaler.fit(
        context_imputer.fit_transform(context_features[indices["train"]])
    )
    normalized_context = context_scaler.transform(
        context_imputer.transform(context_features)
    ).astype(np.float32)
    teacher = np.zeros(len(targets), dtype=np.float32)
    for name, index in indices.items():
        teacher[index] = teacher_probabilities[name].astype(np.float32)
    loaders = {
        name: make_loader(
            normalized,
            rules,
            normalized_context,
            teacher,
            targets,
            index,
            batch_size,
            shuffle=name == "train",
        )
        for name, index in indices.items()
    }
    model = KnowledgeTemporalRiskGNN(
        base_feature_dim=normalized.shape[-1],
        window_size=normalized.shape[1],
        hidden_dim=hidden_dim,
        dropout=0.2,
        use_rule_nodes=True,
        context_dim=normalized_context.shape[-1],
    ).to(device)
    train_targets = targets[indices["train"]]
    positive = max(int(train_targets.sum()), 1)
    negative = max(int(len(train_targets) - positive), 1)
    pos_weight = min(negative / positive, 10.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_score = -1.0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_windows, batch_rules, batch_context, teacher_probability, labels in loaders["train"]:
            batch_windows = batch_windows.to(device)
            batch_rules = batch_rules.to(device)
            batch_context = batch_context.to(device)
            teacher_probability = teacher_probability.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_windows, batch_rules, batch_context)
            classification = criterion(logits, labels)
            distillation = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, teacher_probability
            )
            confidence = batch_rules[:, 3]
            consistency = (
                (confidence * torch.nn.functional.softplus(-logits)).sum()
                / confidence.sum().clamp_min(1.0)
            )
            loss = classification + 0.35 * distillation + 0.05 * consistency
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_probability = gnn_predict(model, loaders["val"], device)
        threshold = choose_threshold(targets[indices["val"]], val_probability, "macro_f1")
        score = float(
            metric_dict(targets[indices["val"]], val_probability, threshold)["macro_f1"]
        )
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": score})
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    probabilities = {
        name: gnn_predict(model, loader, device) for name, loader in loaders.items()
    }
    preprocessing = {
        "window_imputer": imputer,
        "window_scaler": scaler,
        "context_imputer": context_imputer,
        "context_scaler": context_scaler,
        "history": history,
        "distillation_weight": 0.35,
    }
    return model, probabilities, preprocessing


def choose_ensemble(
    target: np.ndarray,
    tree: np.ndarray,
    gnn: np.ndarray,
    validation_target: float,
    minimum_red_count: int,
    minimum_red_recall: float,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for alpha in np.linspace(0.0, 1.0, 21):
        probability = alpha * tree + (1.0 - alpha) * gnn
        frame = pd.DataFrame({"target_binary": target, "risk_probability": probability})
        boundary = choose_sand_boundaries(
            frame, validation_target, minimum_red_count, minimum_red_recall
        )
        if boundary is None:
            continue
        candidate = {"alpha_tree": float(alpha), "alpha_gnn": float(1.0 - alpha), **boundary}
        if best is None or (
            candidate["coverage"],
            candidate["red_recall_over_all_positive"],
        ) > (
            best["coverage"],
            best["red_recall_over_all_positive"],
        ):
            best = candidate
    if best is None:
        return {"status": "no_feasible_ensemble"}
    return best


def event_metrics(
    prediction: pd.DataFrame,
    events: pd.DataFrame,
    low: float,
    high: float,
    horizon_seconds: int,
) -> dict[str, object]:
    if events.empty:
        return {"event_count": 0, "hit_count": 0, "event_recall": 0.0}
    frame = prediction.copy()
    frame["window_end_dt"] = pd.to_datetime(frame["window_end"], errors="coerce")
    red = frame.loc[frame["risk_probability"] >= high]
    test_groups = set(zip(frame["well"].astype(str), frame["segment"].astype(str)))
    relevant = events.loc[
        [
            (str(well), str(segment)) in test_groups
            for well, segment in zip(events["well"], events["segment"])
        ]
    ].copy()
    hits = 0
    lead_seconds: list[float] = []
    for event in relevant.itertuples():
        start = pd.Timestamp(event.event_start)
        lower = start - pd.Timedelta(seconds=horizon_seconds)
        candidates = red.loc[
            red["well"].astype(str).eq(str(event.well))
            & red["segment"].astype(str).eq(str(event.segment))
            & red["window_end_dt"].between(lower, start, inclusive="left")
        ]
        if not candidates.empty:
            hits += 1
            first = candidates["window_end_dt"].min()
            lead_seconds.append(float((start - first).total_seconds()))
    decided = (frame["risk_probability"] <= low) | (frame["risk_probability"] >= high)
    return {
        "event_count": int(len(relevant)),
        "hit_count": int(hits),
        "event_recall": float(hits / len(relevant)) if len(relevant) else 0.0,
        "median_lead_seconds": float(np.median(lead_seconds)) if lead_seconds else None,
        "red_prediction_count": int((frame["risk_probability"] >= high).sum()),
        "decided_count": int(decided.sum()),
    }


def main() -> None:
    args = parser().parse_args()
    set_seed(args.seed)
    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_sand_risk_dataset(
        args.data_dir,
        horizons_seconds=args.horizons_seconds,
        window_size=args.window_size,
        stride=args.stride,
        result_glob=args.result_glob,
    )
    split_values = dataset.groups if args.split_unit == "well" else dataset.segment_groups
    manifest = split_groups(split_values, seed=args.seed)
    indices = indices_for_manifest(split_values, manifest)

    horizon_results: dict[str, object] = {}
    tree_models: dict[int, LGBMClassifier] = {}
    tree_probabilities: dict[int, dict[str, np.ndarray]] = {}
    horizon_rankings: list[tuple[float, float, int]] = []
    for horizon in sorted(dataset.targets):
        target = dataset.targets[horizon]
        model, probabilities = train_tree(dataset.features, target, indices, args.seed)
        tree_models[horizon] = model
        tree_probabilities[horizon] = probabilities
        val_frame = prediction_frame(
            dataset.metadata, indices["val"], target, probabilities["val"]
        )
        test_frame = prediction_frame(
            dataset.metadata, indices["test"], target, probabilities["test"]
        )
        boundary = selective_result(
            val_frame,
            args.boundary_validation_target,
            args.minimum_validation_red_count,
            args.minimum_validation_red_recall,
        )
        if boundary.get("status"):
            test_boundary = boundary
            coverage = 0.0
            red_recall = 0.0
        else:
            test_boundary = boundary_metrics(
                test_frame, float(boundary["low_threshold"]), float(boundary["high_threshold"])
            )
            coverage = float(boundary["coverage"])
            red_recall = float(boundary["red_recall_over_all_positive"])
        threshold = choose_threshold(
            target[indices["val"]], probabilities["val"], "macro_f1"
        )
        horizon_results[str(horizon)] = {
            "positive_counts": {
                name: int(target[index].sum()) for name, index in indices.items()
            },
            "threshold": threshold,
            "validation": metric_dict(target[indices["val"]], probabilities["val"], threshold),
            "test": metric_dict(target[indices["test"]], probabilities["test"], threshold),
            "validation_boundary": boundary,
            "test_boundary": test_boundary,
        }
        horizon_rankings.append((coverage, red_recall, horizon))

    selected_horizon = max(horizon_rankings)[2]
    selected_target = dataset.targets[selected_horizon]
    selected_tree = tree_probabilities[selected_horizon]
    final_probabilities = selected_tree
    model_summary: dict[str, object] = {"selected_model": "lightgbm"}

    if not args.skip_gnn and args.model_mode != "tree_main":
        gnn, gnn_probabilities, preprocessing = train_gnn(
            dataset.features,
            dataset.windows,
            dataset.rule_scores,
            selected_target,
            selected_tree,
            indices,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
        )
        if args.model_mode == "gnn_main":
            final_probabilities = gnn_probabilities
            model_summary = {
                "selected_model": "knowledge_gnn",
                "lightgbm_role": "baseline_only",
                "gnn_validation_macro_f1": metric_dict(
                    selected_target[indices["val"]],
                    gnn_probabilities["val"],
                    choose_threshold(
                        selected_target[indices["val"]],
                        gnn_probabilities["val"],
                        "macro_f1",
                    ),
                ),
            }
        else:
            ensemble = choose_ensemble(
                selected_target[indices["val"]],
                selected_tree["val"],
                gnn_probabilities["val"],
                args.boundary_validation_target,
                args.minimum_validation_red_count,
                args.minimum_validation_red_recall,
            )
            if not ensemble.get("status"):
                alpha = float(ensemble["alpha_tree"])
                final_probabilities = {
                    name: alpha * selected_tree[name] + (1.0 - alpha) * gnn_probabilities[name]
                    for name in indices
                }
                model_summary = {
                    "selected_model": "lightgbm_knowledge_gnn_ensemble",
                    "ensemble_validation": ensemble,
                }
        torch.save(gnn.state_dict(), run_dir / "knowledge_gnn.pt")
        joblib.dump(preprocessing, run_dir / "gnn_preprocessing.joblib")

    validation_frame = prediction_frame(
        dataset.metadata,
        indices["val"],
        selected_target,
        final_probabilities["val"],
    )
    test_frame = prediction_frame(
        dataset.metadata,
        indices["test"],
        selected_target,
        final_probabilities["test"],
    )
    final_boundary = selective_result(
        validation_frame,
        args.boundary_validation_target,
        args.minimum_validation_red_count,
        args.minimum_validation_red_recall,
    )
    if final_boundary.get("status"):
        test_boundary = final_boundary
        events = {"status": "boundaries_unavailable"}
    else:
        low = float(final_boundary["low_threshold"])
        high = float(final_boundary["high_threshold"])
        test_boundary = boundary_metrics(test_frame, low, high)
        events = event_metrics(
            test_frame, dataset.event_table, low, high, selected_horizon
        )
        for frame in (validation_frame, test_frame):
            frame["risk_level"] = np.where(
                frame["risk_probability"] <= low,
                "绿色",
                np.where(frame["risk_probability"] >= high, "红色", "黄色"),
            )

    validation_frame.to_csv(
        run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    test_frame.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(tree_models[selected_horizon], run_dir / "lightgbm.joblib")

    result = {
        "task": "past 60 seconds predict future rule-defined sand-risk candidate event",
        "label_type": "weak supervision from pressure-rise sand-warning rules",
        "excluded_rule": "pressure-drop candidate events are excluded pending expert confirmation",
        "data_dir": str(Path(args.data_dir).resolve()),
        "source_file_count": dataset.source_file_count,
        "well_count": int(len(np.unique(dataset.groups))),
        "segment_count": int(len(np.unique(dataset.segment_groups))),
        "sample_count": int(len(dataset.features)),
        "event_count": int(len(dataset.event_table)),
        "window_size": args.window_size,
        "stride": args.stride,
        "split_unit": args.split_unit,
        "horizons_seconds": sorted(dataset.targets),
        "boundary_validation_target": args.boundary_validation_target,
        "minimum_validation_red_count": args.minimum_validation_red_count,
        "minimum_validation_red_recall": args.minimum_validation_red_recall,
        "selected_horizon_seconds": selected_horizon,
        "split_manifest": manifest,
        "split_sample_counts": {
            name: int(len(index)) for name, index in indices.items()
        },
        "lightgbm_horizon_results": horizon_results,
        "model_summary": model_summary,
        "model_mode": args.model_mode,
        "final_validation_boundary": final_boundary,
        "final_test_boundary": test_boundary,
        "final_test_event_metrics": events,
        "limitations": [
            "The current JY workbooks contain no independently reviewed sand-plug event labels.",
            "Accuracy measures prediction of rule-generated candidate events, not confirmed field sand plugs.",
            "Final acceptance requires replacing or auditing weak labels with expert-reviewed events.",
        ],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "selected_horizon_seconds": selected_horizon,
                "model": model_summary,
                "test_boundary": test_boundary,
                "test_event_metrics": events,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
