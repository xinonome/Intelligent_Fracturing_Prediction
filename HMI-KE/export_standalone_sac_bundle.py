"""Export the current per-second SAC advisory policy as one portable pickle.

The output intentionally contains the deterministic actor only, together with
the exact feature normalization, hierarchical option rules, action decoding
configuration, safety projection configuration and the existing pressure
predictor.  Runtime inference therefore needs Python, NumPy and the pressure
model dependency, but does not need Stable-Baselines3 or the project source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline import (  # noqa: E402
    build_dataset,
    estimate_sample_interval_seconds,
    load_or_discover_segment_frames,
    segment_split,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "hmi"
    / "retrain_raw_fdbh_20260905"
    / "sac_20260905_160043"
    / "sac_fracturing_policy.zip"
)
DEFAULT_SUMMARY = DEFAULT_MODEL.with_name("summary.json")
DEFAULT_PRESSURE_MODEL = (
    PROJECT_ROOT
    / "deliverables"
    / "txt_realtime_prediction"
    / "调控算法包"
    / "realtime_pressure_flow_models.pkl"
)
DEFAULT_OUTPUT = DEFAULT_PRESSURE_MODEL.with_name("project_sac_agent_bundle.pkl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actor_weights(model) -> dict[str, np.ndarray]:
    state = model.policy.actor.state_dict()
    names = (
        "latent_pi.0.weight",
        "latent_pi.0.bias",
        "latent_pi.2.weight",
        "latent_pi.2.bias",
        "mu.weight",
        "mu.bias",
    )
    return {
        name: state[name].detach().cpu().numpy().astype(np.float32)
        for name in names
    }


def _numpy_actor(observation: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(observation, dtype=np.float32)
    value = np.maximum(value @ weights["latent_pi.0.weight"].T + weights["latent_pi.0.bias"], 0.0)
    value = np.maximum(value @ weights["latent_pi.2.weight"].T + weights["latent_pi.2.bias"], 0.0)
    return np.tanh(value @ weights["mu.weight"].T + weights["mu.bias"])


def _select_option(pressure: float, environment: dict) -> int:
    pressure_limit = 110.0
    return 3 if pressure > pressure_limit * float(environment["high_pressure_ratio"]) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="导出逐秒300点SAC独立推理包")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pressure-model", type=Path, default=DEFAULT_PRESSURE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from stable_baselines3 import SAC

    summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    frames = load_or_discover_segment_frames(
        str(PROJECT_ROOT / "Data" / "raw_frac"),
        None,
        "FDBH",
        "SGSJ",
        ["SGBY", "PL", "SB", "WORKING_TYPE"],
        ["WITHfiltered", "便签数据", "综合", "aggregate", "combined"],
        0,
        0,
        None,
    )
    interval = estimate_sample_interval_seconds(frames, "SGSJ", 10.0)
    state_points = max(2, int(round(300.0 / interval)))
    action_points = max(1, int(round(60.0 / interval)))
    dataset = build_dataset(
        frames,
        ["SGBY", "PL", "SB"],
        ["PL", "SB"],
        "SGSJ",
        state_points,
        action_points,
        "WORKING_TYPE",
    )
    train_idx, _, _ = segment_split(dataset.meta, 0.75, 0.1, 2026)
    train_features = np.asarray(dataset.x[train_idx], dtype=np.float64)
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(train_features.std(axis=0), 1.0e-5).astype(np.float32)
    train_meta = dataset.meta.iloc[train_idx].reset_index(drop=True)
    pressure_reference = float(np.nanmedian(train_meta["current_pressure"]))
    pressure_scale = max(float(np.nanstd(train_meta["current_pressure"])), 1.0)

    model = SAC.load(args.model, device="cpu")
    weights = _actor_weights(model)
    environment = dict(summary["environment"])
    schedule = dict(summary["pump_schedule_constraint"])

    test_index = int(train_idx[0])
    feature = np.asarray(dataset.x[test_index], dtype=np.float32)
    row = dataset.meta.iloc[test_index]
    option = _select_option(float(row["current_pressure"]), environment)
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[option] = 1.0
    observation = np.concatenate(
        [
            np.clip((feature - feature_mean) / feature_std, -10.0, 10.0),
            np.asarray(
                [
                    float(row["current_pressure"]) / max(pressure_reference, 1.0),
                    float(row["current_flow"]) / max(float(schedule["max_flow_m3_min"]), 1.0),
                    float(row["current_sand_ratio"]) / max(float(schedule["sand_ratio_scale_percent"]), 1.0),
                ],
                dtype=np.float32,
            ),
            one_hot,
            np.asarray([0.0], dtype=np.float32),
        ]
    )
    expected_action, _ = model.predict(observation, deterministic=True)
    portable_action = _numpy_actor(observation, weights)
    error = float(np.max(np.abs(np.asarray(expected_action) - portable_action)))
    if observation.shape != model.observation_space.shape:
        raise RuntimeError(
            f"导出观测维度不一致：构造={observation.shape}，模型={model.observation_space.shape}"
        )
    if error > 1.0e-5:
        raise RuntimeError(f"NumPy actor 与 SAC 不一致，最大误差={error}")

    # Store the auxiliary predictor as opaque bytes.  Loading the outer SAC
    # package then needs only the standard library and NumPy; LightGBM is
    # imported only when the pressure predictor is actually deserialized.
    pressure_payload = args.pressure_model.read_bytes() if args.pressure_model.exists() else None

    package = {
        "kind": "intelligent_fracturing_project_sac_agent_bundle",
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scientific_status": summary.get("quality_gate", {}).get("status", "development_only"),
        "policy": {
            "algorithm": "SAC",
            "seed": 2026,
            "total_timesteps": int(summary.get("total_timesteps", 0)),
            "source_model": str(args.model),
            "source_sha256": _sha256(args.model),
            "observation_size": int(observation.size),
            "action_size": 2,
            "weights": weights,
        },
        "features": {
            "state_columns": ["SGBY", "PL", "SB"],
            "sample_interval_seconds": float(interval),
            "state_seconds": 300.0,
            "state_points": int(state_points),
            "action_seconds": 60.0,
            "action_points": int(action_points),
            "feature_names": list(dataset.feature_names),
            "mean": feature_mean,
            "std": feature_std,
            "pressure_reference": pressure_reference,
            "pressure_scale": pressure_scale,
        },
        "hierarchy": {
            "options": ["hold", "grow", "divert", "safe"],
            "initial_option_progress": 0.0,
        },
        "environment": environment,
        "reward_config": dict(summary["reward_config"]),
        "safety_projection": schedule,
        "pressure_predictor": pressure_payload,
        "self_test": {
            "observation": observation.astype(np.float32),
            "expected_action": np.asarray(expected_action, dtype=np.float32),
            "portable_action": portable_action.astype(np.float32),
            "max_abs_error": error,
        },
        "training_data": {
            "frame_count": len(frames),
            "dataset_samples": int(len(dataset.x)),
            "train_samples": int(len(train_idx)),
            "source": "Data/raw_frac",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    manifest = args.output.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "kind": package["kind"],
                "format_version": package["format_version"],
                "created_at": package["created_at"],
                "scientific_status": package["scientific_status"],
                "policy": {key: value for key, value in package["policy"].items() if key != "weights"},
                "features": {
                    key: value
                    for key, value in package["features"].items()
                    if key not in {"mean", "std", "feature_names"}
                },
                "safety_projection": package["safety_projection"],
                "self_test_max_abs_error": error,
                "output": str(args.output),
                "output_sha256": _sha256(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest), "self_test_error": error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
