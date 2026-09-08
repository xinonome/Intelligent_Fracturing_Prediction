from __future__ import annotations

"""Minimal cross-well transfer learning for the reviewed green/yellow/red task."""

import argparse
import json
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_future_risk_levels import build_intervals, gnn_metrics, label_at
from frac_gnn.future_risk_gnn import KnowledgeTemporalRiskGNN
from frac_gnn.sand_risk_data import build_sand_risk_dataset


def causal_rule_scores(windows: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Build rule-node activations from each historical window only.

    Unlike the reviewed interval labels, these scores do not read the enhanced
    warning sheet or any future endpoint. The three channels are pressure rise,
    pressure drop, and simultaneous rate/sand stability.
    """
    names = list(feature_names)
    pressure_idx = names.index("SGBY")
    rate_idx = names.index("PL")
    sand_idx = names.index("SB")
    pressure = windows[:, :, pressure_idx].astype(np.float32)
    rate = windows[:, :, rate_idx].astype(np.float32)
    sand = windows[:, :, sand_idx].astype(np.float32)
    pressure_delta = np.nan_to_num(pressure[:, -1] - pressure[:, 0], nan=0.0)
    pressure_rise = np.clip(np.maximum(pressure_delta, 0.0) / 5.0, 0.0, 1.0)
    pressure_drop = np.clip(np.maximum(-pressure_delta, 0.0) / 5.0, 0.0, 1.0)
    rate_std = np.nanstd(rate, axis=1)
    sand_std = np.nanstd(sand, axis=1)
    rate_stability = np.clip(1.0 - np.nan_to_num(rate_std, nan=1.0) / 0.2, 0.0, 1.0)
    sand_stability = np.clip(1.0 - np.nan_to_num(sand_std, nan=1.0) / 1.0, 0.0, 1.0)
    stability = rate_stability * sand_stability
    return np.column_stack([pressure_rise, pressure_drop, stability]).astype(np.float32)


def causal_current_levels(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Emit the live green/yellow/red rule state at each window end.

    Every decision uses only the historical window ending at the current sample
    and earlier windows from the same construction segment. The completed
    reviewed-warning intervals are deliberately not read here.
    """
    names = list(feature_names)
    pressure_idx = names.index("SGBY")
    rate_idx = names.index("PL")
    sand_idx = names.index("SB")
    ordered = metadata.copy()
    ordered["sample_row"] = np.arange(len(ordered))
    ordered["window_end"] = pd.to_datetime(ordered["window_end"], errors="coerce")
    ordered["window_start"] = pd.to_datetime(ordered["window_start"], errors="coerce")
    ordered = ordered.sort_values(["group", "window_end", "sample_row"], kind="stable")
    levels = np.zeros(len(ordered), dtype=np.int64)
    previous: dict[str, tuple[float, float] | None] = {}

    for row in ordered.itertuples(index=False):
        sample_index = int(row.sample_row)
        group = str(row.group)
        window = windows[sample_index]
        pressure = window[:, pressure_idx].astype(float)
        rate = window[:, rate_idx].astype(float)
        sand = window[:, sand_idx].astype(float)
        prior = previous.get(group)

        if np.isnan(pressure).sum() > len(pressure) * 0.3:
            previous[group] = None
            continue
        if not np.isfinite(pressure[0]) or not np.isfinite(pressure[-1]):
            previous[group] = None
            continue

        rate_start, rate_end = rate[0], rate[-1]
        rate_rising = (
            np.isfinite(rate_start)
            and np.isfinite(rate_end)
            and rate_end > rate_start + 0.2
            and np.nanmax(rate) - np.nanmin(rate) > 0.3
        )
        rate_std = np.nanstd(rate)
        sand_std = np.nanstd(sand)
        if rate_rising or rate_std > 0.2 or sand_std > 1.0:
            previous[group] = None
            continue

        differences = np.diff(pressure)
        if not np.isfinite(differences).all():
            previous[group] = None
            continue
        pressure_change = float(pressure[-1] - pressure[0])
        duration_seconds = (row.window_end - row.window_start).total_seconds()
        rising = bool(np.all(differences >= 0) and pressure_change > 0.5)
        falling = bool(np.all(differences <= 0) and pressure_change < -5.0)

        if falling:
            levels[sample_index] = 2
            previous[group] = None
            continue

        if rising:
            peak = float(np.nanmax(pressure))
            spread = float(np.nanstd(pressure))
            peak_increasing = prior is not None and peak > prior[0]
            spread_increasing = prior is not None and spread > prior[1]
            slope_per_minute = pressure_change / (duration_seconds / 60.0) if duration_seconds > 0 else 0.0
            if peak_increasing and spread_increasing and (duration_seconds >= 60.0 or slope_per_minute > 5.0):
                levels[sample_index] = 1
            previous[group] = (peak, spread)
            continue

        previous[group] = None

    return levels


LABEL_NAMES = ["绿色", "黄色", "红色"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Simple cross-well transfer learning for reviewed green/yellow/red risk levels."
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--target-well", required=True)
    result.add_argument(
        "--source-wells",
        default="",
        help="Optional comma-separated source-well list. If omitted, every non-target well is used.",
    )
    result.add_argument("--result-glob", default="*_小条统计_增强版.xlsx")
    result.add_argument("--support-ratio", type=float, default=0.30)
    result.add_argument("--window-size", type=int, default=6)
    result.add_argument("--stride", type=int, default=3)
    result.add_argument("--hidden-dim", type=int, default=48)
    result.add_argument("--pretrain-epochs", type=int, default=5)
    result.add_argument("--finetune-epochs", type=int, default=3)
    result.add_argument(
        "--pretrained-checkpoint",
        default="",
        help="Optional existing source-model checkpoint. If supplied, skip source re-training and transfer from it.",
    )
    result.add_argument(
        "--causal-input",
        action="store_true",
        help="Strict no-leakage mode: do not use the offline current risk label or its prior; use only historical windows and causal rule scores.",
    )
    result.add_argument(
        "--causal-rule-state",
        action="store_true",
        help="Strict no-leakage rule-guided mode: feed the live rule state calculated from the historical window, never the completed reviewed interval.",
    )
    result.add_argument("--batch-size", type=int, default=2048)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--finetune-learning-rate", type=float, default=2e-4)
    result.add_argument("--prior-strength", type=float, default=3.0)
    result.add_argument("--class-weight-cap", type=float, default=5.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--output-dir", default="runs/future_risk_levels_transfer")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def well_from_result(path: Path) -> str:
    suffix = "_小条统计_增强版.xlsx"
    name = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    return name.removesuffix("_更新后").removesuffix("_split")


def split_target_segments(
    segment_groups: np.ndarray,
    target_mask: np.ndarray,
    target_level: np.ndarray,
    support_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    ids = np.unique(segment_groups[target_mask].astype(str))
    if len(ids) < 2:
        raise ValueError("Target well must have at least two independent segments")
    count = max(1, min(len(ids) - 1, int(round(len(ids) * np.clip(support_ratio, 0.1, 0.9)))))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids).tolist()
    support_ids: list[str] = []
    # Keep at least one reviewed red segment in the support set whenever the
    # target well contains one. This makes the fine-tuning task meaningful.
    for level in (2, 1):
        candidates = [
            segment
            for segment in shuffled
            if segment not in support_ids
            and np.any(
                target_level[target_mask & (segment_groups.astype(str) == segment)] == level
            )
        ]
        if candidates and len(support_ids) < count:
            support_ids.append(str(candidates[0]))
    for segment in shuffled:
        if len(support_ids) >= count:
            break
        if segment not in support_ids:
            support_ids.append(str(segment))
    query_ids = sorted(str(segment) for segment in ids if str(segment) not in support_ids)
    support_ids = sorted(support_ids)
    support = target_mask & np.isin(segment_groups.astype(str), support_ids)
    query = target_mask & np.isin(segment_groups.astype(str), query_ids)
    return support, query, support_ids, query_ids


def make_labels(
    metadata: pd.DataFrame,
    data_dir: Path,
    result_glob: str,
) -> tuple[np.ndarray, np.ndarray]:
    interval_map: dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]]] = {}
    for path in sorted(data_dir.glob(result_glob)):
        try:
            interval_map[well_from_result(path)] = build_intervals(path)
        except (ValueError, FileNotFoundError):
            continue
    window_end = pd.to_datetime(metadata["window_end"], errors="coerce")
    target_time = window_end + pd.Timedelta(seconds=10)
    current: list[int] = []
    target: list[int] = []
    for row, current_timestamp, target_timestamp in zip(
        metadata.itertuples(index=False), window_end, target_time
    ):
        intervals = interval_map.get(str(row.well), {})
        current.append(label_at(str(row.segment), current_timestamp, intervals))
        target.append(label_at(str(row.segment), target_timestamp, intervals))
    return np.asarray(current, dtype=np.int64), np.asarray(target, dtype=np.int64)


def normalize_windows(windows: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    windows = np.where(np.isfinite(windows), windows, np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    source_rows = windows[source_mask].reshape(-1, windows.shape[-1])
    scaler.fit(imputer.fit_transform(source_rows))
    return scaler.transform(imputer.transform(windows.reshape(-1, windows.shape[-1]))).reshape(windows.shape).astype(np.float32)


def loader(
    windows: np.ndarray,
    rules: np.ndarray,
    current_one_hot: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    data = TensorDataset(
        torch.from_numpy(windows[mask]).float(),
        torch.from_numpy(rules[mask]).float(),
        torch.from_numpy(current_one_hot[mask]).float(),
        torch.from_numpy(targets[mask]).long(),
    )
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train_epochs(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    prior_strength: float,
    epochs: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for windows, rules, current, targets in data_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows.to(device), rules.to(device), current.to(device))
            logits = logits + prior_strength * current.to(device)
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0})
    return history


@torch.no_grad()
def predict(model: nn.Module, data_loader: DataLoader, device: str, prior_strength: float) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for windows, rules, current, _ in data_loader:
        logits = model(windows.to(device), rules.to(device), current.to(device))
        logits = logits + prior_strength * current.to(device)
        result.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.int64)


def main() -> None:
    args = parser().parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_sand_risk_dataset(
        data_dir,
        horizons_seconds=[60],
        window_size=args.window_size,
        stride=args.stride,
        result_glob=args.result_glob,
    )
    current_level, target_level = make_labels(dataset.metadata, data_dir, args.result_glob)
    if args.causal_input and args.causal_rule_state:
        raise ValueError("Use only one of --causal-input or --causal-rule-state")
    groups = dataset.groups.astype(str)
    target_mask = groups == str(args.target_well)
    if not target_mask.any():
        raise ValueError(f"Target well not found: {args.target_well}")
    if args.source_wells.strip():
        requested_sources = [item.strip() for item in args.source_wells.split(",") if item.strip()]
        available_wells = set(groups.tolist())
        missing_sources = sorted(set(requested_sources) - available_wells)
        if missing_sources:
            raise ValueError(f"Source wells not found in dataset: {missing_sources}")
        source_mask = np.isin(groups, requested_sources)
        overlap = source_mask & target_mask
        if overlap.any():
            raise ValueError("Source wells and target well overlap")
    else:
        source_mask = ~target_mask
    if not source_mask.any():
        raise ValueError("No source samples available")
    support_mask, query_mask, support_segments, query_segments = split_target_segments(
        dataset.segment_groups.astype(str), target_mask, target_level, args.support_ratio, args.seed
    )
    normalized_windows = normalize_windows(dataset.windows, source_mask)
    uses_causal_rules = args.causal_input or args.causal_rule_state
    rules = causal_rule_scores(dataset.windows, dataset.base_feature_names) if uses_causal_rules else np.asarray(dataset.rule_scores, dtype=np.float32)
    current_one_hot = np.eye(3, dtype=np.float32)[current_level]
    effective_current_one_hot = current_one_hot
    effective_prior_strength = float(args.prior_strength)
    input_current_level = current_level
    if args.causal_input:
        if args.pretrained_checkpoint.strip():
            raise ValueError(
                "--causal-input cannot reuse a checkpoint trained with the offline current-risk state; retrain the source model causally"
            )
        effective_current_one_hot = np.zeros_like(current_one_hot)
        effective_prior_strength = 0.0
        input_current_level = np.zeros_like(current_level)
    elif args.causal_rule_state:
        if args.pretrained_checkpoint.strip():
            raise ValueError(
                "--causal-rule-state cannot reuse a checkpoint trained with the offline current-risk state; retrain the source model with live rules"
            )
        input_current_level = causal_current_levels(dataset.windows, dataset.metadata, dataset.base_feature_names)
        effective_current_one_hot = np.eye(3, dtype=np.float32)[input_current_level]

    source_loader = loader(normalized_windows, rules, effective_current_one_hot, target_level, source_mask, args.batch_size, True)
    source_eval = loader(normalized_windows, rules, effective_current_one_hot, target_level, source_mask, args.batch_size, False)
    support_loader = loader(normalized_windows, rules, effective_current_one_hot, target_level, support_mask, args.batch_size, True)
    support_eval = loader(normalized_windows, rules, effective_current_one_hot, target_level, support_mask, args.batch_size, False)
    query_eval = loader(normalized_windows, rules, effective_current_one_hot, target_level, query_mask, args.batch_size, False)

    model = KnowledgeTemporalRiskGNN(
        base_feature_dim=normalized_windows.shape[-1],
        window_size=normalized_windows.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=0.2,
        use_rule_nodes=True,
        context_dim=3,
    ).to(args.device)
    model.classifier[-1] = nn.Linear(args.hidden_dim, 3)
    counts = np.bincount(target_level[source_mask], minlength=3).astype(float)
    weights = np.sqrt(counts.max() / np.maximum(counts, 1.0))
    weights = np.minimum(weights, float(args.class_weight_cap))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=args.device))
    if args.pretrained_checkpoint.strip():
        checkpoint_path = Path(args.pretrained_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(state)
        pretrain_history = []
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        pretrain_history = train_epochs(model, source_loader, optimizer, criterion, args.device, effective_prior_strength, args.pretrain_epochs)
    before_source = predict(model, source_eval, args.device, effective_prior_strength)
    before_support = predict(model, support_eval, args.device, effective_prior_strength)
    before_query = predict(model, query_eval, args.device, effective_prior_strength)
    torch.save(model.state_dict(), run_dir / "pretrained_source_levels_gnn.pt")

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier")
    finetune_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.finetune_learning_rate,
        weight_decay=1e-4,
    )
    finetune_history = train_epochs(model, support_loader, finetune_optimizer, criterion, args.device, effective_prior_strength, args.finetune_epochs)
    after_support = predict(model, support_eval, args.device, effective_prior_strength)
    after_query = predict(model, query_eval, args.device, effective_prior_strength)
    torch.save(model.state_dict(), run_dir / "finetuned_target_levels_gnn.pt")

    def split_metrics(mask: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
        truth = target_level[mask]
        return {
            "sample_count": int(len(truth)),
            "label_counts": {LABEL_NAMES[i]: int((truth == i).sum()) for i in range(3)},
            **gnn_metrics(truth, prediction),
        }

    metrics = {
        "source_pretrain": split_metrics(source_mask, before_source),
        "target_query_before_finetune": split_metrics(query_mask, before_query),
        "target_support_after_finetune": split_metrics(support_mask, after_support),
        "target_query_after_finetune": split_metrics(query_mask, after_query),
        "target_query_persistence": {
            "sample_count": int(query_mask.sum()),
            "accuracy": float((input_current_level[query_mask] == target_level[query_mask]).mean()),
        },
    }
    output = {
        "task": "cross-well transfer for next-sample reviewed green/yellow/red risk level",
        "target_well": args.target_well,
        "source_wells": sorted(np.unique(groups[source_mask]).tolist()),
        "support_segments": support_segments,
        "query_segments": query_segments,
        "support_ratio": float(args.support_ratio),
        "window_size": int(args.window_size),
        "target_definition": "next sample about 10 seconds ahead: green=no interval, yellow=砂堵迹象, red=砂堵风险",
        "architecture": (
            f"KnowledgeTemporalRiskGNN; {args.window_size} time nodes + 3 rule nodes; causal historical input only"
            if args.causal_input
            else (
                f"KnowledgeTemporalRiskGNN; {args.window_size} time nodes + 3 rule nodes; live causal rule-state one-hot prior"
                if args.causal_rule_state
                else f"KnowledgeTemporalRiskGNN; {args.window_size} time nodes + 3 rule nodes; offline current risk state as one-hot prior"
            )
        ),
        "transfer_strategy": "pretrain source wells, freeze GNN backbone, fine-tune classifier on target support segments",
        "pretrained_checkpoint": args.pretrained_checkpoint.strip() or None,
        "input_mode": (
            "strict causal: no current-risk label or prior"
            if args.causal_input
            else (
                "strict causal: live rule-state one-hot plus prior"
                if args.causal_rule_state
                else "rule-guided: offline current risk-state one-hot plus prior"
            )
        ),
        "configured_prior_strength": float(args.prior_strength),
        "prior_strength": effective_prior_strength,
        "causal_input": bool(args.causal_input),
        "causal_rule_state": bool(args.causal_rule_state),
        "class_weight_cap": float(args.class_weight_cap),
        "metrics": metrics,
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
    }
    (run_dir / "transfer_metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
