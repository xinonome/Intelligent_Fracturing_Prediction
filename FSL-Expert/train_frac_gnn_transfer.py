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

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets, save_json
from frac_gnn.model import TemporalSegmentGNN


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    sample_count: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GNN transfer learning for frac working type classification.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--include-columns", nargs="*", default=None)
    parser.add_argument("--exclude-columns", nargs="*", default=None)
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=None)
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--binary-normal-abnormal", action="store_true")
    parser.add_argument("--abnormal-label", default="异常")
    parser.add_argument("--trim-before-sand", action="store_true")
    parser.add_argument("--sand-column", default="SB")
    parser.add_argument("--sand-threshold", type=float, default=0.0)
    parser.add_argument("--add-dynamic-features", action="store_true")
    parser.add_argument("--dynamic-feature-columns", nargs="*", default=None)
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[3, 5, 10])
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--transfer-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-exponent", type=float, default=0.5)
    parser.add_argument("--class-weight-max-ratio", type=float, default=5.0)
    parser.add_argument("--oversample-minority", action="store_true")
    parser.add_argument("--transfer-support-ratio", type=float, default=0.3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="runs/frac_gnn_transfer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_support_query(graphs, support_ratio: float, seed: int):
    if not graphs:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)
    support_count = max(1, int(round(len(graphs) * support_ratio)))
    support_count = min(support_count, len(graphs) - 1) if len(graphs) > 1 else len(graphs)
    support_idx = set(indices[:support_count].tolist())
    support = [graph for i, graph in enumerate(graphs) if i in support_idx]
    query = [graph for i, graph in enumerate(graphs) if i not in support_idx]
    return support, query


def compute_class_counts(graphs, num_classes: int, device: str) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for graph in graphs:
        counts[int(graph.y)] += 1.0
    return counts.to(device)


def build_loss_weights(counts: torch.Tensor, exponent: float, max_ratio: float) -> torch.Tensor:
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    if nonzero.any():
        base = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
        base = torch.pow(base, exponent)
        base = torch.clamp(base, max=max_ratio)
        weights[nonzero] = base
        weights = weights / weights[nonzero].mean()
    return weights


def build_loader(graphs, batch_size: int, shuffle: bool, oversample: bool, class_counts: torch.Tensor | None = None):
    if not graphs:
        return DataLoader(graphs, batch_size=batch_size, shuffle=False)
    if not oversample or class_counts is None:
        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
    sample_weights = []
    for graph in graphs:
        count = float(class_counts[int(graph.y)].item())
        sample_weights.append(0.0 if count <= 0 else 1.0 / count)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(graphs),
        replacement=True,
    )
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


def evaluate(model, loader, criterion, device: str) -> tuple[EvalMetrics, dict]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            losses.append(float(loss.item()))
            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    if not y_true:
        return EvalMetrics(0.0, 0.0, 0.0, 0), {}
    return (
        EvalMetrics(
            loss=float(np.mean(losses)) if losses else 0.0,
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            sample_count=len(y_true),
        ),
        classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    )


def train_epochs(model, loader, optimizer, criterion, device: str, epochs: int) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0})
    return history


def freeze_feature_extractor(model: TemporalSegmentGNN) -> None:
    for name, parameter in model.named_parameters():
        if "classifier" not in name and "lin" not in name:
            parameter.requires_grad = False


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    run_root = Path(args.run_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(run_root / "config.json", vars(args))

    base = prepare_base_datasets(
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
        trim_before_sand=args.trim_before_sand,
        sand_column=args.sand_column,
        sand_threshold=args.sand_threshold,
        add_dynamic_features=args.add_dynamic_features,
        dynamic_feature_columns=args.dynamic_feature_columns,
        rolling_windows=args.rolling_windows,
    )
    prepared = build_prepared_for_window(base, args.window_size)
    support_graphs, query_graphs = split_support_query(
        prepared.transfer_graphs,
        support_ratio=args.transfer_support_ratio,
        seed=args.seed,
    )

    class_counts = compute_class_counts(prepared.train_graphs, len(prepared.classes), args.device)
    class_weights = build_loss_weights(
        class_counts,
        exponent=args.class_weight_exponent,
        max_ratio=args.class_weight_max_ratio,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = build_loader(
        prepared.train_graphs,
        args.batch_size,
        shuffle=True,
        oversample=args.oversample_minority,
        class_counts=class_counts,
    )
    test_loader = build_loader(prepared.test_graphs, args.batch_size, shuffle=False, oversample=False)
    support_loader = build_loader(support_graphs, args.batch_size, shuffle=True, oversample=False)
    query_loader = build_loader(query_graphs, args.batch_size, shuffle=False, oversample=False)

    model = TemporalSegmentGNN(
        input_dim=len(prepared.feature_columns),
        hidden_dim=args.hidden_dim,
        output_dim=len(prepared.classes),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)

    pretrain_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pretrain_history = train_epochs(
        model,
        train_loader,
        pretrain_optimizer,
        criterion,
        args.device,
        args.pretrain_epochs,
    )

    before_query_metrics, before_query_report = evaluate(model, query_loader, criterion, args.device)
    test_metrics, test_report = evaluate(model, test_loader, criterion, args.device)
    torch.save(model.state_dict(), run_root / "pretrained_model.pt")

    if args.freeze_backbone:
        freeze_feature_extractor(model)
    finetune_optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
    )
    finetune_history = train_epochs(
        model,
        support_loader,
        finetune_optimizer,
        criterion,
        args.device,
        args.finetune_epochs,
    )
    after_query_metrics, after_query_report = evaluate(model, query_loader, criterion, args.device)
    torch.save(model.state_dict(), run_root / "finetuned_model.pt")

    result = {
        "classes": prepared.classes,
        "feature_columns": prepared.feature_columns,
        "window_size": args.window_size,
        "split_manifest": prepared.split_manifest,
        "graph_counts": {
            "train": len(prepared.train_graphs),
            "test": len(prepared.test_graphs),
            "transfer_total": len(prepared.transfer_graphs),
            "transfer_support": len(support_graphs),
            "transfer_query": len(query_graphs),
        },
        "class_counts": class_counts.detach().cpu().tolist(),
        "class_weights": class_weights.detach().cpu().tolist(),
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
        "metrics": {
            "source_test_before_finetune": asdict(test_metrics),
            "transfer_query_before_finetune": asdict(before_query_metrics),
            "transfer_query_after_finetune": asdict(after_query_metrics),
        },
        "reports": {
            "source_test_before_finetune": test_report,
            "transfer_query_before_finetune": before_query_report,
            "transfer_query_after_finetune": after_query_report,
        },
    }
    save_json(run_root / "transfer_metrics.json", result)
    print(json.dumps({"run_root": str(run_root), "metrics": result["metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
