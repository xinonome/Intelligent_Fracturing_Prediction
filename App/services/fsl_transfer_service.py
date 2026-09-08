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
    requested_files = [Path(value) for value in (getattr(args, "source_files", None) or [])]
    if requested_files:
        frames: dict[str, pd.DataFrame] = {}
        for source_file in requested_files:
            loaded = discover_segment_frames(
                str(source_file), "FDBH", "WORKING_TYPE", args.reference_header_path,
                DEFAULT_EXCLUDES, well_names=(args.source_well, args.target_well),
            )
            for key, frame in loaded.items():
                unique = key
                suffix = 2
                while unique in frames:
                    unique = f"{key}__{suffix}"
                    suffix += 1
                frames[unique] = frame
    else:
        frames = discover_segment_frames(
            args.data_path,
            "FDBH",
            "WORKING_TYPE",
            args.reference_header_path,
            DEFAULT_EXCLUDES,
            well_names=(args.source_well, args.target_well),
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


def _trace(
    model,
    graphs,
    classes: list[str],
    device: str,
    on_partial=None,
) -> dict[str, list[Any]]:
    """Run inference and optionally publish partial points for live plotting."""

    model.eval()
    observed: list[int | None] = []
    predicted: list[int] = []
    window_end: list[int] = []
    observed_labels: list[str] = []
    loader = build_loader(graphs, 1, False, False)
    total = max(len(graphs), 1)
    publish_every = max(total // 30, 1)
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            logits = model(batch)
            known_value = getattr(batch, "label_known", True)
            is_known = bool(known_value.reshape(-1)[0].item()) if hasattr(known_value, "reshape") else bool(known_value)
            observed.append(int(batch.y.reshape(-1)[0].item()) if is_known else None)
            predicted.append(int(logits.argmax(dim=-1).reshape(-1)[0].item()))
            raw_label = getattr(batch, "observed_label", None)
            if isinstance(raw_label, list) and raw_label:
                observed_labels.append(str(raw_label[0]))
            else:
                observed_labels.append(classes[observed[-1]] if observed[-1] is not None else "未编码")
            value = getattr(batch, "window_end", len(window_end))
            window_end.append(int(value.reshape(-1)[0].item()) if hasattr(value, "reshape") else int(value))
            if on_partial and (index == total or index % publish_every == 0):
                on_partial(
                    {
                        "window_end": list(window_end),
                        "observed_class_index": list(observed),
                        "predicted_class_index": list(predicted),
                    },
                    index,
                    total,
                )
    return {
        "window_end": window_end,
        "observed_class_index": observed,
        "predicted_class_index": predicted,
        "observed_label": observed_labels,
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


def _observed_trace(graphs) -> dict[str, list[Any]]:
    """Read the target labels before model inference so the chart can start early."""

    observed: list[int | None] = []
    window_end: list[int] = []
    loader = build_loader(graphs, 1, False, False)
    for batch in loader:
        known_value = getattr(batch, "label_known", True)
        is_known = bool(known_value.reshape(-1)[0].item()) if hasattr(known_value, "reshape") else bool(known_value)
        observed.append(int(batch.y.reshape(-1)[0].item()) if is_known else None)
        value = getattr(batch, "window_end", len(window_end))
        window_end.append(int(value.reshape(-1)[0].item()) if hasattr(value, "reshape") else int(value))
    return {"window_end": window_end, "observed": observed}


def _write_progress(output_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a readable in-flight trace for the Qt workbench."""

    path = output_dir / "transfer_progress.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_pair(args) -> dict[str, Any]:
    set_seed(args.seed)
    frames, source_keys, target_keys = _load_frames(args)
    _write_progress(
        args.output_dir,
        {
            "status": "running",
            "mode": "train",
            "phase": f"数据读取完成（迁移井 {len(source_keys)} 段、目标井 {len(target_keys)} 段），正在构造模型窗口",
            "progress": 0.10,
            "source_well": args.source_well,
            "target_well": args.target_well,
            "classes": [],
            "traces": {"window_end": [], "observed": [], "before": [], "after": []},
        },
    )
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
    observed_trace = _observed_trace(query_graphs)
    progress_traces = {
        "window_end": observed_trace["window_end"],
        "observed": observed_trace["observed"],
        "before": [],
        "after": [],
    }

    def publish_progress(phase: str, progress: float, traces: dict[str, list[Any]] | None = None) -> None:
        _write_progress(
            args.output_dir,
            {
                "status": "running",
                "mode": "train",
                "phase": phase,
                "progress": round(float(progress), 3),
                "source_well": args.source_well,
                "target_well": args.target_well,
                "classes": classes,
                "graph_counts": {
                    "source_train": len(source_graphs),
                    "target_support": len(support_graphs),
                    "target_query": len(query_graphs),
                },
                "traces": traces or progress_traces,
            },
        )

    publish_progress("已生成目标井实际工况曲线，正在准备基础模型", 0.0)
    print("已生成目标井实际工况曲线，正在使用迁移井训练基础模型…", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)

    def on_pretrain_progress(record):
        publish_progress(
            f"基础模型训练：第 {record['epoch']}/{args.pretrain_epochs} 轮",
            0.05 + 0.45 * float(record["epoch"]) / max(args.pretrain_epochs, 1),
        )
        print(
            f"[基础训练] 第 {record['epoch']}/{args.pretrain_epochs} 轮，loss={record['loss']:.6f}",
            flush=True,
        )

    pretrain_history = train_epochs(
        model,
        source_loader,
        optimizer,
        source_criterion,
        args.device,
        args.pretrain_epochs,
        progress_callback=on_pretrain_progress,
    )
    before_metrics, _ = evaluate(model, query_loader, target_criterion, args.device)
    def publish_before_partial(partial, count, total):
        progress_traces["window_end"] = partial["window_end"]
        progress_traces["before"] = partial["predicted_class_index"]
        publish_progress(f"正在生成迁移前预测曲线（{count}/{total}）", 0.50 + 0.02 * count / max(total, 1))

    before_trace = _trace(model, query_graphs, classes, args.device, on_partial=publish_before_partial)
    progress_traces["before"] = before_trace["predicted_class_index"]
    progress_traces["window_end"] = before_trace["window_end"]
    publish_progress("基础模型完成，正在显示迁移前预测", 0.52)
    print("[阶段] 已生成迁移前预测曲线，正在使用目标井支持集微调…", flush=True)
    torch.save(model.state_dict(), args.output_dir / "pretrained_model.pt")

    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_learning_rate, weight_decay=1.0e-4)

    def on_finetune_progress(record):
        publish_progress(
            f"目标井微调：第 {record['epoch']}/{args.finetune_epochs} 轮",
            0.52 + 0.43 * float(record["epoch"]) / max(args.finetune_epochs, 1),
        )
        print(
            f"[目标井微调] 第 {record['epoch']}/{args.finetune_epochs} 轮，loss={record['loss']:.6f}",
            flush=True,
        )

    finetune_history = train_epochs(
        model,
        support_loader,
        finetune_optimizer,
        target_criterion,
        args.device,
        args.finetune_epochs,
        progress_callback=on_finetune_progress,
    )
    after_metrics, _ = evaluate(model, query_loader, target_criterion, args.device)
    def publish_after_partial(partial, count, total):
        progress_traces["window_end"] = partial["window_end"]
        progress_traces["after"] = partial["predicted_class_index"]
        publish_progress(f"正在生成迁移后预测曲线（{count}/{total}）", 0.95 + 0.03 * count / max(total, 1))

    after_trace = _trace(model, query_graphs, classes, args.device, on_partial=publish_after_partial)
    progress_traces["after"] = after_trace["predicted_class_index"]
    progress_traces["window_end"] = after_trace["window_end"]
    publish_progress("迁移后预测完成，正在整理结果", 0.98)
    model_path = args.output_dir / "finetuned_model.pt"
    torch.save(model.state_dict(), model_path)

    package_path = args.output_dir / "transfer_package.joblib"
    created_at = datetime.now().isoformat(timespec="seconds")
    joblib.dump(
        {
            "schema_version": 2,
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
            "target_well": args.target_well,
            "metadata": {
                "created_at": created_at,
                "source_segments": source_keys,
                "target_segments": target_keys,
                "pretrain_epochs": args.pretrain_epochs,
                "finetune_epochs": args.finetune_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "finetune_learning_rate": args.finetune_learning_rate,
                "seed": args.seed,
                "weights_file": model_path.name,
            },
        },
        package_path,
    )
    result = {
        "status": "completed",
        "mode": "train",
        "created_at": created_at,
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
    _write_progress(
        args.output_dir,
        {
            "status": "running",
            "mode": "apply",
            "phase": f"数据读取完成（目标井 {len(target_keys)} 段），正在构造模型窗口",
            "progress": 0.10,
            "source_well": str(package.get("source_well", args.source_well)),
            "target_well": args.target_well,
            "classes": list(package.get("classes", [])),
            "traces": {"window_end": [], "observed": [], "before": [], "after": []},
        },
    )
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
        unknown_label=("NORMAL" if "NORMAL" in package["classes"] else package["classes"][0]),
    )
    if not graphs:
        raise ValueError("目标井没有形成可应用模型的有效窗口")
    observed_trace = _observed_trace(graphs)
    _write_progress(
        args.output_dir,
        {
            "status": "running",
            "mode": "apply",
            "phase": "已生成目标井实际工况曲线，正在应用已有模型",
            "progress": 0.35,
            "source_well": str(package.get("source_well", args.source_well)),
            "target_well": args.target_well,
            "classes": list(package["classes"]),
            "traces": {
                "window_end": observed_trace["window_end"],
                "observed": observed_trace["observed"],
                "before": [],
                "after": [],
            },
        },
    )
    print("已生成目标井实际工况曲线，正在应用已有模型…", flush=True)
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
    classes = list(package["classes"])

    def publish_apply_partial(partial, count, total):
        _write_progress(
            args.output_dir,
            {
                "status": "running",
                "mode": "apply",
                "phase": f"正在生成预测曲线（{count}/{total}）",
                "progress": round(0.35 + 0.63 * count / max(total, 1), 3),
                "source_well": str(package.get("source_well", args.source_well)),
                "target_well": args.target_well,
                "classes": classes,
                "traces": {
                    "window_end": partial["window_end"],
                    "observed": observed_trace["observed"][: len(partial["window_end"])],
                    "before": [],
                    "after": partial["predicted_class_index"],
                },
            },
        )

    trace = _trace(model, graphs, classes, args.device, on_partial=publish_apply_partial)
    _write_progress(
        args.output_dir,
        {
            "status": "running",
            "mode": "apply",
            "phase": "已有模型预测完成，正在整理结果",
            "progress": 0.98,
            "source_well": str(package.get("source_well", args.source_well)),
            "target_well": args.target_well,
            "classes": classes,
            "traces": {
                "window_end": trace["window_end"],
                "observed": trace["observed_class_index"],
                "before": [],
                "after": trace["predicted_class_index"],
            },
        },
    )
    return {
        "status": "completed",
        "mode": "apply",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_well": str(package.get("source_well", args.source_well)),
        "target_well": args.target_well,
        "target_segments": target_keys,
        "classes": classes,
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
    parser.add_argument("--source-files", nargs="*", default=None,
                        help="Exact registered source workbooks for the selected source/target wells.")
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
    # Publish immediately.  Loading the workbook and constructing rolling
    # features can take a while; the Qt workbench should show that the child
    # process is alive instead of looking like it is waiting forever.
    _write_progress(
        args.output_dir,
        {
            "status": "running",
            "mode": args.mode,
            "phase": "正在加载迁移井和目标井数据",
            "progress": 0.02,
            "source_well": args.source_well,
            "target_well": args.target_well,
            "classes": [],
            "traces": {"window_end": [], "observed": [], "before": [], "after": []},
        },
    )
    try:
        result = train_pair(args) if args.mode == "train" else apply_package(args)
        result_path = args.output_dir / "transfer_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_progress(
            args.output_dir,
            {
                "status": "completed",
                "mode": result.get("mode", args.mode),
                "phase": "任务完成",
                "progress": 1.0,
                "source_well": result.get("source_well", args.source_well),
                "target_well": result.get("target_well", args.target_well),
                "classes": result.get("classes", []),
                "traces": result.get("traces", {}),
            },
        )
        print(json.dumps({"status": "completed", "result_path": str(result_path)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _write_progress(
            args.output_dir,
            {
                "status": "failed",
                "mode": args.mode,
                "phase": "任务失败",
                "progress": 1.0,
                "source_well": args.source_well,
                "target_well": args.target_well,
                "error": failure["error"],
                "traces": {"window_end": [], "observed": [], "before": [], "after": []},
            },
        )
        (args.output_dir / "transfer_result.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
