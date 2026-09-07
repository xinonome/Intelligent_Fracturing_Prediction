"""Pair-specific cross-well transfer jobs launched by the acceptance APP.

The original transfer script evaluates a frozen random segment split.  This
service is deliberately narrower: the operator-selected source well supplies
pre-training windows and the selected target well supplies support/query
windows.  Outputs are isolated under ``outputs/app/transfer_runs`` so source
data and registered artifacts remain read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from ..core.paths import PATHS


FSL_ROOT = PATHS.root / "FSL-Expert"
if str(FSL_ROOT) not in sys.path:
    sys.path.insert(0, str(FSL_ROOT))

from frac_gnn.data import (  # noqa: E402
    add_dynamic_features_to_frames,
    build_graphs_for_segments,
    discover_segment_frames,
    fit_label_encoder,
    fit_preprocessors,
    normalize_labels_in_frames,
    select_feature_columns,
    sort_segment_frames,
    trim_segments_from_first_sand,
)
from frac_gnn.model import TemporalSegmentGNN  # noqa: E402
from train_frac_gnn_transfer import (  # noqa: E402
    build_loader,
    build_loss_weights,
    compute_class_counts,
    evaluate,
    set_seed,
    split_support_query,
    train_epochs,
)


DEFAULT_DYNAMIC_COLUMNS = [
    "BZJDH",
    "YTND",
    "SGBY",
    "PL",
    "SB",
    "LJSL",
    "ZDCLYL",
    "ZDJ",
    "LJYL",
    "BZJD",
]
DEFAULT_EXCLUDES = [
    "WITHfiltered",
    "segment_working_type_labels",
    "split_point_label_summary",
    "split_window_label_summary",
    "new_dataset_segment_label_distribution",
    "便签数据",
    "综合",
    "aggregate",
    "combined",
]


def _well_name(frame: pd.DataFrame, fallback: str) -> str:
    if "JTBH" not in frame.columns:
        return fallback
    values = frame["JTBH"].dropna().astype(str).map(str.strip)
    values = values[values != ""]
    return values.iloc[0] if not values.empty else fallback


def _load_frames(args) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    frames = discover_segment_frames(
        args.data_path,
        "FDBH",
        "WORKING_TYPE",
        args.reference_header_path,
        DEFAULT_EXCLUDES,
    )
    frames = sort_segment_frames(frames, "SGSJ")
    frames = normalize_labels_in_frames(frames, "WORKING_TYPE", "NORMAL")
    frames = trim_segments_from_first_sand(frames, sand_column="SB", sand_threshold=0.0)
    frames = add_dynamic_features_to_frames(
        frames,
        base_columns=DEFAULT_DYNAMIC_COLUMNS,
        sand_column="SB",
        rolling_windows=[3, 5, 10],
    )
    source_keys = [key for key, frame in frames.items() if _well_name(frame, key) == args.source_well]
    target_keys = [key for key, frame in frames.items() if _well_name(frame, key) == args.target_well]
    if not source_keys:
        raise ValueError(f"未找到迁移井数据：{args.source_well}")
    if not target_keys:
        raise ValueError(f"未找到目标井数据：{args.target_well}")
    if set(source_keys) & set(target_keys):
        raise ValueError("迁移井和目标井必须不同")
    return frames, source_keys, target_keys


def _trace(model, graphs, classes: list[str], device: str) -> dict[str, list[Any]]:
    model.eval()
    observed: list[int] = []
    predicted: list[int] = []
    window_end: list[int] = []
    loader = build_loader(graphs, 1, False, False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            observed.append(int(batch.y.reshape(-1)[0].item()))
            predicted.append(int(logits.argmax(dim=-1).reshape(-1)[0].item()))
            value = getattr(batch, "window_end", len(window_end))
            window_end.append(int(value.reshape(-1)[0].item()) if hasattr(value, "reshape") else int(value))
    return {
        "window_end": window_end,
        "observed_class_index": observed,
        "predicted_class_index": predicted,
        "observed_label": [classes[index] for index in observed],
        "predicted_label": [classes[index] for index in predicted],
    }


def _model(input_dim: int, output_dim: int, args) -> TemporalSegmentGNN:
    return TemporalSegmentGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)


def train_pair(args) -> dict[str, Any]:
    set_seed(args.seed)
    frames, source_keys, target_keys = _load_frames(args)
    print(
        f"[准备] 迁移井 {args.source_well}：{len(source_keys)} 个井段；"
        f"目标井 {args.target_well}：{len(target_keys)} 个井段。",
        flush=True,
    )
    pair_frames = {key: frames[key] for key in source_keys + target_keys}
    feature_columns = select_feature_columns(
        pair_frames,
        "WORKING_TYPE",
        "FDBH",
        "SGSJ",
        None,
        None,
    )
    label_encoder = fit_label_encoder(pair_frames, "WORKING_TYPE")
    imputer, scaler = fit_preprocessors(pair_frames, source_keys, feature_columns)
    source_graphs = build_graphs_for_segments(
        pair_frames,
        source_keys,
        feature_columns,
        "WORKING_TYPE",
        label_encoder,
        imputer,
        scaler,
        args.window_size,
    )
    target_graphs = build_graphs_for_segments(
        pair_frames,
        target_keys,
        feature_columns,
        "WORKING_TYPE",
        label_encoder,
        imputer,
        scaler,
        args.window_size,
    )
    if not source_graphs:
        raise ValueError("迁移井没有形成有效训练窗口")
    if len(target_graphs) < 2:
        raise ValueError("目标井没有形成足够的支持集和查询集窗口")
    support_graphs, query_graphs = split_support_query(target_graphs, args.support_ratio, args.seed)
    if not support_graphs or not query_graphs:
        raise ValueError("目标井支持集或查询集为空")

    print(
        f"[准备] 有效窗口：迁移井 {len(source_graphs)}，"
        f"目标井支持集 {len(support_graphs)}，查询集 {len(query_graphs)}。",
        flush=True,
    )

    classes = label_encoder.classes_.tolist()
    class_counts = compute_class_counts(source_graphs, len(classes), args.device)
    weights = build_loss_weights(class_counts, exponent=0.5, max_ratio=5.0)
    source_criterion = torch.nn.CrossEntropyLoss(weight=weights)
    target_criterion = torch.nn.CrossEntropyLoss()
    source_loader = build_loader(source_graphs, args.batch_size, True, True, class_counts)
    support_loader = build_loader(support_graphs, args.batch_size, True, False)
    query_loader = build_loader(query_graphs, args.batch_size, False, False)

    input_dim = int(source_graphs[0].x.shape[-1])
    model = _model(input_dim, len(classes), args)
    print("正在使用迁移井训练基础模型…", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    pretrain_history = train_epochs(
        model,
        source_loader,
        optimizer,
        source_criterion,
        args.device,
        args.pretrain_epochs,
        progress_callback=lambda record: print(
            f"[基础训练] 第 {record['epoch']}/{args.pretrain_epochs} 轮，loss={record['loss']:.6f}",
            flush=True,
        ),
    )
    before_metrics, _ = evaluate(model, query_loader, target_criterion, args.device)
    before_trace = _trace(model, query_graphs, classes, args.device)
    torch.save(model.state_dict(), args.output_dir / "pretrained_model.pt")

    print("正在使用目标井支持集进行微调…", flush=True)
    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_learning_rate, weight_decay=1.0e-4)
    finetune_history = train_epochs(
        model,
        support_loader,
        finetune_optimizer,
        target_criterion,
        args.device,
        args.finetune_epochs,
        progress_callback=lambda record: print(
            f"[目标井微调] 第 {record['epoch']}/{args.finetune_epochs} 轮，loss={record['loss']:.6f}",
            flush=True,
        ),
    )
    after_metrics, _ = evaluate(model, query_loader, target_criterion, args.device)
    after_trace = _trace(model, query_graphs, classes, args.device)
    model_path = args.output_dir / "finetuned_model.pt"
    torch.save(model.state_dict(), model_path)

    package_path = args.output_dir / "transfer_package.joblib"
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "input_dim": input_dim,
            "classes": classes,
            "label_encoder": label_encoder,
            "imputer": imputer,
            "scaler": scaler,
            "window_size": args.window_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "source_well": args.source_well,
        },
        package_path,
    )
    result = {
        "status": "completed",
        "mode": "train",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_well": args.source_well,
        "target_well": args.target_well,
        "source_segments": source_keys,
        "target_segments": target_keys,
        "classes": classes,
        "graph_counts": {
            "source_train": len(source_graphs),
            "target_support": len(support_graphs),
            "target_query": len(query_graphs),
        },
        "metrics": {
            "before": before_metrics.__dict__,
            "after": after_metrics.__dict__,
        },
        "traces": {
            "window_end": before_trace["window_end"],
            "observed": before_trace["observed_class_index"],
            "before": before_trace["predicted_class_index"],
            "after": after_trace["predicted_class_index"],
        },
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
        "model_path": str(model_path),
        "package_path": str(package_path),
    }
    return result


def apply_package(args) -> dict[str, Any]:
    package_path = Path(args.package)
    model_path = Path(args.model)
    if not package_path.exists() or not model_path.exists():
        raise FileNotFoundError("模型文件或配套预处理包不存在")
    package = joblib.load(package_path)
    frames, _source_keys, target_keys = _load_frames(args)
    target_frames = {key: frames[key] for key in target_keys}
    graphs = build_graphs_for_segments(
        target_frames,
        target_keys,
        list(package["feature_columns"]),
        "WORKING_TYPE",
        package["label_encoder"],
        package["imputer"],
        package["scaler"],
        int(package["window_size"]),
    )
    if not graphs:
        raise ValueError("目标井没有形成可应用模型的有效窗口")
    model = TemporalSegmentGNN(
        input_dim=int(package.get("input_dim") or graphs[0].x.shape[-1]),
        hidden_dim=int(package["hidden_dim"]),
        output_dim=len(package["classes"]),
        num_layers=int(package["num_layers"]),
        dropout=float(package["dropout"]),
    ).to(args.device)
    try:
        state = torch.load(model_path, map_location=args.device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state)
    trace = _trace(model, graphs, list(package["classes"]), args.device)
    return {
        "status": "completed",
        "mode": "apply",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_well": str(package.get("source_well", args.source_well)),
        "target_well": args.target_well,
        "target_segments": target_keys,
        "classes": list(package["classes"]),
        "graph_counts": {"target_query": len(graphs)},
        "traces": {
            "window_end": trace["window_end"],
            "observed": trace["observed_class_index"],
            "after": trace["predicted_class_index"],
        },
        "model_path": str(model_path),
        "package_path": str(package_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run operator-selected cross-well transfer")
    parser.add_argument("--mode", choices=("train", "apply"), required=True)
    parser.add_argument("--source-well", required=True)
    parser.add_argument("--target-well", required=True)
    parser.add_argument("--data-path", default=str(PATHS.data / "raw_frac"))
    parser.add_argument("--reference-header-path", default=str(PATHS.data / "raw_frac" / "FDBH26.xlsx"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--package", default="")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = train_pair(args) if args.mode == "train" else apply_package(args)
        result_path = args.output_dir / "transfer_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "completed", "result_path": str(result_path)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        (args.output_dir / "transfer_result.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
