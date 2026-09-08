"""Generate segment-specific advisory actions for the no-DAS APP path.

The no-DAS page cannot reuse the global HMI replay from the DAS stage or from
another well.  This tool builds the same 300-second state window used by the
trained policy from one selected raw construction segment, runs the currently
configured policy on CPU, and stores only that segment's advisory rows.

This is an offline advisory replay.  It does not control equipment and it does
not fabricate cluster observations; the pressure-only DT cache remains the
source of the no-DAS physical state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HMI_ROOT = ROOT / "HMI-KE"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HMI_ROOT))
if str(ROOT / "DT-Crack") not in sys.path:
    sys.path.insert(0, str(ROOT / "DT-Crack"))

from data_pipeline import (  # noqa: E402
    build_dataset,
    load_or_discover_segment_frames,
    sort_frame,
)
from pump_schedule_adapter import (  # noqa: E402
    SCHEDULE_NUMERIC_COLUMNS,
    attach_schedule_to_frame,
    load_pump_schedule,
)
from decision_engine.integrated_reward import IntegratedRewardConfig  # noqa: E402
from decision_engine.pump_schedule_constraints import get_schedule_constraint  # noqa: E402
from rl.digital_twin_env import (  # noqa: E402
    HierarchicalDigitalTwinEnvConfig,
    HierarchicalDigitalTwinFracturingControlEnv,
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _normalise_id(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _select_segment(frames: dict[str, pd.DataFrame], stage_id: str) -> tuple[str, pd.DataFrame]:
    wanted = _normalise_id(stage_id)
    exact = [(key, frame) for key, frame in frames.items() if _normalise_id(key) == wanted]
    if exact:
        return exact[0]
    suffix = wanted.replace("fdbh", "")
    candidates = [
        (key, frame)
        for key, frame in frames.items()
        if _normalise_id(key).replace("fdbh", "") == suffix
    ]
    if candidates:
        return max(candidates, key=lambda item: len(item[1]))
    if len(frames) == 1:
        return next(iter(frames.items()))
    # A raw workbook occasionally contains a numeric FDBH field whose value
    # does not match its filename.  Choosing the largest group is still
    # deterministic, but record the actual segment key in the output.
    return max(frames.items(), key=lambda item: len(item[1]))


def _regularise_rows(frame: pd.DataFrame, time_column: str, target_interval_s: float = 10.0) -> pd.DataFrame:
    """Make the policy window time scale comparable across mixed raw exports."""

    out = sort_frame(frame, time_column)
    parsed = pd.to_datetime(out[time_column], errors="coerce") if time_column in out else pd.Series(dtype=object)
    if len(parsed) < 3 or not parsed.notna().any():
        return out.reset_index(drop=True)
    valid = parsed.dropna()
    deltas = valid.diff().dt.total_seconds().dropna()
    deltas = deltas[(deltas > 0.0) & (deltas < 3600.0)]
    if not len(deltas):
        return out.reset_index(drop=True)
    source_interval = float(deltas.median())
    if source_interval >= target_interval_s * 0.8:
        return out.reset_index(drop=True)
    elapsed = (parsed - parsed.iloc[0]).dt.total_seconds()
    out = out.assign(_elapsed_s=elapsed, _time_bucket=np.floor(elapsed / target_interval_s))
    out = out.loc[out["_elapsed_s"].notna()].groupby("_time_bucket", sort=True, as_index=False).first()
    return out.drop(columns=["_elapsed_s", "_time_bucket"], errors="ignore").reset_index(drop=True)


def _default_header_path() -> Path | None:
    data_root = ROOT / "Data" / "raw_frac"
    candidates = [data_root / "FDBH26.xlsx"] if (data_root / "FDBH26.xlsx").exists() else []
    if not candidates:
        candidates = sorted(data_root.glob("FDBH*.xlsx"))
    return candidates[0] if candidates else None


def _policy(selection, algorithm: str | None = None, model_path: str | Path | None = None):
    from stable_baselines3 import PPO, SAC, TD3

    classes = {"ppo": PPO, "sac": SAC, "td3": TD3}
    selected_algorithm = str(algorithm or selection.agent_policy).strip().lower()
    selected_path = Path(model_path) if model_path else None
    if selected_path is None and selected_algorithm == selection.agent_policy:
        selected_path = Path(selection.policy_model_path) if selection.policy_model_path else None
    if selected_path is None:
        from App.data.hmi_loader import discover_agent_models
        from App.data.registry_loader import RegistryLoader

        model = next(
            (item for item in discover_agent_models(RegistryLoader()) if item.get("model_id") == selected_algorithm),
            None,
        )
        selected_path = Path(model["policy_path"]) if model and model.get("policy_path") else None
    if selected_algorithm not in classes or not selected_path or not selected_path.exists():
        raise RuntimeError("当前 APP 没有可加载的 PPO/SAC/TD3 策略模型")
    return classes[selected_algorithm].load(selected_path, device="cpu"), selected_algorithm


def _run_policy(features: np.ndarray, meta: pd.DataFrame, model, algorithm: str) -> list[dict]:
    expected_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    action_encoding = "centered_delta" if algorithm == "td3" else "legacy"
    rows: list[dict] = []
    target = 0
    # A fresh episode at each block prevents an unsafe simulated response from
    # hiding the rest of the construction curve.  The measured controls are
    # re-anchored at every cursor by the environment before decoding action.
    block_size = 60
    while target < len(features) - 1:
        block_offset = int(target)
        block_end = min(block_offset + block_size + 1, len(features))
        block_features = features[block_offset:block_end]
        block_meta = meta.iloc[block_offset:block_end].reset_index(drop=True)
        block_context = pd.DataFrame(index=np.arange(len(block_meta)))
        env_config = HierarchicalDigitalTwinEnvConfig(
            episode_steps=max(len(block_features), 2),
            action_seconds=60.0,
            high_level_interval_steps=6,
            action_encoding=action_encoding,
            schedule_reward_weight=0.75 if algorithm == "td3" else 0.25,
            action_boundary_weight=0.35,
            terminate_on_unsafe=True,
        )
        env = HierarchicalDigitalTwinFracturingControlEnv(
            block_features,
            block_meta,
            block_context,
            get_schedule_constraint("continuous"),
            IntegratedRewardConfig(),
            env_config,
            seed=2026 + target,
            random_start=False,
        )
        actual_shape = tuple(env.observation_space.shape)
        if expected_shape and tuple(expected_shape) != actual_shape:
            raise RuntimeError(
                "策略模型与当前完整观测维度不一致："
                f"模型需要 {tuple(expected_shape)}，当前环境生成 {actual_shape}。"
                "请使用与当前状态窗口和上下文配置一致的模型。"
            )
        obs, _ = env.reset(options={"start_index": 0})
        block_steps = 0
        for _ in range(block_size):
            local_index = int(env._cursor)
            if local_index >= len(block_features) - 1:
                target = min(block_offset + block_steps, len(features))
                break
            source_index = block_offset + local_index
            started = perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = perf_counter() - started
            row = {
                "source_index": source_index,
                "time": str(meta.iloc[source_index].get("time", "")),
                "segment_id": str(meta.iloc[source_index].get("segment_id", "")),
                "current_flow_m3_min": float(info.get("pre_action_flow_m3_min", np.nan)),
                "current_sand_ratio_percent": float(info.get("pre_action_sand_ratio_percent", np.nan)),
                "flow_m3_min": float(info.get("flow_m3_min", np.nan)),
                "sand_ratio_percent": float(info.get("sand_ratio_percent", np.nan)),
                "recommended_sand_ratio_percent": float(info.get("recommended_sand_ratio_percent", np.nan)),
                "decision_compute_seconds": float(elapsed),
                "reward": float(reward),
                "algorithm": algorithm.upper(),
            }
            for key in (
                "high_sand_ratio",
                "sand_ratio_requires_confirmation",
                "action_clipped",
                "unsafe",
                "uncertain",
                "policy_sand_action_effective",
                "current_sand_above_absolute_limit",
            ):
                row[key] = info.get(key, False)
            for key in (
                "sand_delta_from_current_percent",
                "sand_delta_from_reference_percent",
                "bottomhole_pressure_mpa",
                "net_pressure_mpa",
                "posterior_error",
                "abnormal_probability",
                "sand_plug_probability",
                "cluster_balance_degree",
            ):
                row[key] = float(info.get(key, np.nan))
            row["recommendation_source"] = "segment_policy_inference"
            rows.append(row)
            block_steps += 1
            target = source_index + 1
            if terminated or truncated:
                break
        if block_steps == 0:
            target = min(target + 1, len(features))
    return rows


def _set_cache_hmi(
    cache_path: Path,
    source_path: Path,
    rows: list[dict],
    algorithm: str,
) -> None:
    cache = _read_json(cache_path)
    relative = source_path.resolve().relative_to(ROOT.resolve()).as_posix()

    def update_payload(payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        meta = payload.setdefault("meta", {})
        meta["hmi_available"] = True
        by_model = meta.setdefault("hmi_recommendations_by_model", {})
        if not isinstance(by_model, dict):
            by_model = {}
            meta["hmi_recommendations_by_model"] = by_model
        by_model[str(algorithm).lower()] = relative
        if not meta.get("hmi_recommendations_source"):
            meta["hmi_recommendations_source"] = relative
            meta["hmi_recommendation_rows"] = len(rows)
            meta["hmi_policy"] = str(algorithm).upper()
        meta["hmi_recommendation_status"] = "segment_policy_inference"

    update_payload(cache)
    scenarios = cache.get("scenarios", {})
    if isinstance(scenarios, dict):
        update_payload(scenarios.get("no_das_pressure_only", {}))
    _write_json(cache_path, cache)


def build_for_dataset(
    dataset_id: str,
    max_points: int,
    header_path: Path | None,
    algorithm: str | None = None,
    model_path: str | Path | None = None,
    pump_schedule_path: str | Path | None = None,
    include_schedule_context: bool = False,
) -> dict:
    from App.core.model_runtime import resolve_runtime_selection
    from App.data.dt_dataset_registry import get_dataset

    dataset = get_dataset(dataset_id)
    if dataset.get("adapter") != "raw_frac_construction":
        raise ValueError(f"{dataset_id} 不是无 DAS 单井段数据集")
    source = ROOT / str(dataset["pressure_source"])
    frames = load_or_discover_segment_frames(
        str(source),
        str(header_path) if header_path else None,
        "FDBH",
        "SGSJ",
        ["SGBY", "PL", "SB", "WORKING_TYPE"],
        ["WITHfiltered", "便签数据", "综合", "aggregate", "combined"],
        0,
        0,
        None,
    )
    segment_key, frame = _select_segment(frames, str(dataset.get("stage_id", dataset_id)))
    schedule_metadata = {"enabled": False}
    if pump_schedule_path:
        schedule = load_pump_schedule(pump_schedule_path)
        frame, schedule_metadata = attach_schedule_to_frame(frame, schedule, "SGSJ", strict_identity=True)
    frame = _regularise_rows(frame, "SGSJ")
    state_columns = ["SGBY", "PL", "SB"]
    if include_schedule_context:
        state_columns.extend(SCHEDULE_NUMERIC_COLUMNS)
    bundle = build_dataset(
        {segment_key: frame},
        state_columns,
        ["PL", "SB"],
        "SGSJ",
        30,
        6,
        "WORKING_TYPE",
    )
    if max_points > 0 and len(bundle.x) > max_points:
        chosen = np.linspace(0, len(bundle.x) - 1, max_points, dtype=int)
        features = bundle.x[chosen]
        meta = bundle.meta.iloc[chosen].reset_index(drop=True)
    else:
        features = bundle.x
        meta = bundle.meta.reset_index(drop=True)

    selection = resolve_runtime_selection()
    model, selected_algorithm = _policy(selection, algorithm=algorithm, model_path=model_path)
    rows = _run_policy(features, meta, model, selected_algorithm)
    output_name = "hmi_recommendations.csv" if selected_algorithm == selection.agent_policy else f"hmi_recommendations_{selected_algorithm}.csv"
    output = ROOT / "outputs" / "app" / "datasets" / dataset_id / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["source_index", "recommendation_source"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cache_path = ROOT / str(dataset["cache_source"])
    _set_cache_hmi(cache_path, output, rows, selected_algorithm)
    return {
        "dataset_id": dataset_id,
        "segment_key": segment_key,
        "source_rows": int(len(frame)),
        "policy_rows": int(len(rows)),
        "policy": str(selected_algorithm).upper(),
        "output": str(output),
        "cache": str(cache_path),
        "pump_schedule": schedule_metadata,
        "schedule_context_enabled": bool(include_schedule_context),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment-specific no-DAS agent recommendation caches.")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--all", action="store_true", help="Build all discovered raw single-stage datasets.")
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--pump-schedule-path", default=None, help="可选，与当前井段身份匹配的施工泵序表")
    parser.add_argument("--include-schedule-context", action="store_true", help="把泵序参考排量/砂比/进度加入策略特征")
    parser.add_argument("--algorithm", choices=["ppo", "sac", "td3"], default=None)
    parser.add_argument("--model", default=None, help="可选的策略文件；缺省按 --algorithm 发现已登记模型")
    args = parser.parse_args()
    if args.include_schedule_context and not args.pump_schedule_path:
        parser.error("--include-schedule-context 需要同时提供 --pump-schedule-path")

    from App.data.dt_dataset_registry import list_datasets

    header = Path(args.reference_header_path).resolve() if args.reference_header_path else _default_header_path()
    if args.all:
        dataset_ids = [
            str(item["dataset_id"])
            for item in list_datasets()
            if item.get("adapter") == "raw_frac_construction"
        ]
    elif args.dataset_id:
        dataset_ids = [str(args.dataset_id)]
    else:
        parser.error("请提供 --dataset-id 或 --all")

    results = []
    for dataset_id in dataset_ids:
        started = perf_counter()
        try:
            result = build_for_dataset(
                dataset_id,
                args.max_points,
                header,
                algorithm=args.algorithm,
                model_path=args.model,
                pump_schedule_path=args.pump_schedule_path,
                include_schedule_context=args.include_schedule_context,
            )
            result["elapsed_seconds"] = round(perf_counter() - started, 3)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:  # keep other wells running in --all mode
            result = {
                "dataset_id": dataset_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(perf_counter() - started, 3),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.all:
                raise
    manifest = ROOT / "outputs" / "app" / "no_das_agent_cache_manifest.json"
    _write_json(
        manifest,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "results": results,
            "policy_input": "raw single-stage construction features, 300-second history window",
            "status": "completed" if all(item.get("status", "completed") != "failed" for item in results) else "partial",
        },
    )


if __name__ == "__main__":
    main()
