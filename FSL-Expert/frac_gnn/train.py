from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from .data import PreparedData, build_prepared_for_window, prepare_base_datasets, prepare_datasets, save_json
from .model import TemporalSegmentGNN


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    sample_count: int


def compute_class_weights(graphs, num_classes: int, device: str) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for graph in graphs:
        counts[int(graph.y)] += 1.0
    return counts.to(device)


def build_loss_weights(
    counts: torch.Tensor,
    exponent: float,
    max_ratio: float | None,
) -> torch.Tensor:
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    if nonzero.any():
        base = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
        if exponent != 1.0:
            base = torch.pow(base, exponent)
        if max_ratio is not None:
            base = torch.clamp(base, max=max_ratio)
        weights[nonzero] = base
        weights = weights / weights[nonzero].mean()
    return weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a hydraulic fracturing GNN classifier.")
    parser.add_argument("--data-path", required=True, help="Segment dataset directory or total dataset file.")
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--include-columns", nargs="*", default=None)
    parser.add_argument("--exclude-columns", nargs="*", default=None)
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=None)
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--binary-normal-abnormal", action="store_true")
    parser.add_argument("--abnormal-label", default="异常")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--transfer-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--class-weight-exponent", type=float, default=0.5)
    parser.add_argument("--class-weight-max-ratio", type=float, default=3.0)
    parser.add_argument("--oversample-minority", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="runs/frac_gnn")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def evaluate(model, loader, criterion, device) -> tuple[Metrics, dict]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            losses.append(loss.item())
            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    if not y_true:
        metrics = Metrics(loss=0.0, accuracy=0.0, macro_f1=0.0, sample_count=0)
        return metrics, {}

    metrics = Metrics(
        loss=float(np.mean(losses)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        sample_count=len(y_true),
    )
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, report


def build_train_loader(args, graphs, class_counts: torch.Tensor) -> DataLoader:
    if not args.oversample_minority:
        return DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
    sample_weights: list[float] = []
    for graph in graphs:
        class_index = int(graph.y)
        count = float(class_counts[class_index].item())
        sample_weights.append(0.0 if count <= 0 else 1.0 / count)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(graphs),
        replacement=True,
    )
    return DataLoader(graphs, batch_size=args.batch_size, sampler=sampler)


def train_one_window(args, prepared: PreparedData, window_size: int, run_root: Path) -> dict:
    window_dir = run_root / f"window_{window_size}"
    window_dir.mkdir(parents=True, exist_ok=True)

    class_counts = compute_class_weights(prepared.train_graphs, len(prepared.classes), args.device)
    class_weights = build_loss_weights(
        class_counts,
        exponent=args.class_weight_exponent,
        max_ratio=args.class_weight_max_ratio,
    )

    save_json(
        window_dir / "dataset_manifest.json",
        {
            "window_size": window_size,
            "feature_columns": prepared.feature_columns,
            "classes": prepared.classes,
            "segment_sizes": prepared.segment_sizes,
            "split_manifest": prepared.split_manifest,
            "graph_counts": {
                "train": len(prepared.train_graphs),
                "val": len(prepared.val_graphs),
                "test": len(prepared.test_graphs),
                "transfer": len(prepared.transfer_graphs),
            },
            "class_counts": class_counts.detach().cpu().tolist(),
            "class_weights": class_weights.detach().cpu().tolist(),
            "binary_normal_abnormal": args.binary_normal_abnormal,
            "oversample_minority": args.oversample_minority,
        },
    )

    if not prepared.train_graphs:
        raise ValueError(f"No training graphs were generated for window size {window_size}.")

    train_loader = build_train_loader(args, prepared.train_graphs, class_counts)
    val_loader = DataLoader(prepared.val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(prepared.test_graphs, batch_size=args.batch_size, shuffle=False)
    transfer_loader = DataLoader(prepared.transfer_graphs, batch_size=args.batch_size, shuffle=False)

    model = TemporalSegmentGNN(
        input_dim=len(prepared.feature_columns),
        hidden_dim=args.hidden_dim,
        output_dim=len(prepared.classes),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(
        weight=None if args.disable_class_weights else class_weights
    )

    best_state = None
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics, _ = evaluate(model, val_loader, criterion, args.device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }
        )

        if val_metrics.sample_count == 0:
            monitored_f1 = 0.0
        else:
            monitored_f1 = val_metrics.macro_f1

        if monitored_f1 > best_val_f1:
            best_val_f1 = monitored_f1
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), window_dir / "model.pt")

    train_metrics, train_report = evaluate(model, train_loader, criterion, args.device)
    val_metrics, val_report = evaluate(model, val_loader, criterion, args.device)
    test_metrics, test_report = evaluate(model, test_loader, criterion, args.device)
    transfer_metrics, transfer_report = evaluate(model, transfer_loader, criterion, args.device)

    result = {
        "window_size": window_size,
        "history": history,
        "metrics": {
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
            "test": asdict(test_metrics),
            "transfer": asdict(transfer_metrics),
        },
        "reports": {
            "train": train_report,
            "val": val_report,
            "test": test_report,
            "transfer": transfer_report,
        },
    }
    save_json(window_dir / "metrics.json", result)
    return result


def run_experiment(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    run_root = Path(args.run_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(run_root / "config.json", vars(args))

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
        binary_normal_abnormal=args.binary_normal_abnormal,
        abnormal_label=args.abnormal_label,
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        transfer_ratio=args.transfer_ratio,
    )

    summary: dict[str, dict] = {}
    for window_size in args.window_sizes:
        prepared = build_prepared_for_window(base_prepared, window_size)
        summary[str(window_size)] = train_one_window(args, prepared, window_size, run_root)

    overview = {
        "run_root": str(run_root),
        "window_results": {
            window_size: {
                "val_macro_f1": result["metrics"]["val"]["macro_f1"],
                "test_macro_f1": result["metrics"]["test"]["macro_f1"],
                "transfer_macro_f1": result["metrics"]["transfer"]["macro_f1"],
            }
            for window_size, result in summary.items()
        },
    }
    save_json(run_root / "summary.json", overview)
    return overview


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    overview = run_experiment(args)
    print(json.dumps(overview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
