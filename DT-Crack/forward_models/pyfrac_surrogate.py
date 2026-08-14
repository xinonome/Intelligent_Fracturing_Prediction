"""PyFrac teacher data and residual surrogate for the DT-Crack pipeline.

The surrogate is deliberately a residual model:

    PyFrac state = PKN state + learned residual

PyFrac remains an offline teacher.  The online EnKF gate is conservative and
only opens after strict scenario-group holdout scores pass for length and
pressure residuals.  The default online path therefore remains PKN + EnKF.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DT_ROOT = ROOT / "DT-Crack"
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from forward_models.fracture_length_models import calc_pkn
from forward_models.pyfrac_adapter import PyFracAdapter
from forward_models.pyfrac_config import PyFracConfig


# The first ten columns are kept compatible with the original teacher table.
# The remaining columns expose the physical scenario explicitly.  In
# particular, the six allocation weights are inputs, not free EnKF states.
FEATURE_COLUMNS = [
    "e_prime_gpa",
    "leakoff_m_sqrt_s",
    "viscosity_pa_s",
    "min_stress_mpa",
    "fracture_toughness_pa_sqrt_m",
    "height_m",
    "q_m3_s",
    "time_s",
    "cluster_spacing_m",
    "allocation_weight",
    "cluster_id",
    "q_total_m3_s",
    "cumulative_injection_m3",
    "q_ramp_m3_s2",
    "leakoff_volume_fraction",
    "stress_delta_mpa",
    "viscosity_ratio",
    "allocation_entropy",
    "allocation_min",
    "allocation_max",
    "allocation_imbalance",
    "stress_shadow_mpa",
    "scenario_type_code",
    "pkn_half_length_m",
    "pkn_aperture_mm",
    "pkn_pressure_mpa",
    "allocation_w1",
    "allocation_w2",
    "allocation_w3",
    "allocation_w4",
    "allocation_w5",
    "allocation_w6",
    "q_over_viscosity",
    "pressure_stress_ratio",
    "sqrt_time_s",
    "log_time_s",
    "leakoff_sqrt_time",
    "rate_change_fraction",
]

TARGET_COLUMNS = ["delta_length_m", "delta_aperture_mm", "delta_pressure_mpa"]
GATED_TARGETS = ["delta_length_m", "delta_pressure_mpa"]
ALLOCATION_COLUMNS = [f"allocation_w{i}" for i in range(1, 7)]

SCENARIO_TYPE_CODES = {
    "baseline": 0,
    "rate_step_up": 1,
    "rate_step_down": 2,
    "rate_pulse": 3,
    "enhanced_leakoff": 4,
    "stress_shift": 5,
    "viscosity_shift": 6,
    "uniform_allocation": 7,
    "heel_dominant": 8,
    "toe_dominant": 9,
    "middle_dominant": 10,
    "alternating_allocation": 11,
    "edge_dominant": 12,
    "combined_stress_rate": 13,
    "combined_leakoff_viscosity": 14,
}


@dataclass(frozen=True)
class SurrogateConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64, 32)
    max_iter: int = 1000
    random_state: int = 20260810
    min_test_r2: float = 0.80


class PyFracResidualSurrogate:
    """Residual model used after a PKN forward run.

    Target scaling is fitted on the training groups only.  This prevents a
    large pressure residual from dominating the length residual and avoids
    leaking test-group statistics into the model.
    """

    def __init__(
        self,
        model: Pipeline,
        target_scaler: StandardScaler,
        metrics: dict[str, object],
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.target_scaler = target_scaler
        self.metrics = metrics
        self.metadata = metadata or {}

    @classmethod
    def train(
        cls,
        teacher: pd.DataFrame,
        config: SurrogateConfig | None = None,
    ) -> "PyFracResidualSurrogate":
        config = config or SurrogateConfig()
        frame = _prepare_feature_frame(teacher)
        missing = [column for column in TARGET_COLUMNS if column not in frame]
        if missing:
            raise ValueError(f"Teacher table is missing target columns: {missing}")
        frame = frame.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
        if len(frame) < 24:
            raise ValueError("At least 24 valid teacher rows are required")

        groups = _group_values(frame)
        train_mask, valid_mask, test_mask, split_info = _strict_group_split(
            groups,
            random_state=config.random_state,
        )
        if test_mask.sum() < 2 or valid_mask.sum() < 2:
            raise ValueError("Strict scenario holdout produced too few validation/test rows")

        train_count = int(train_mask.sum())
        use_early_stopping = train_count >= 48
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=config.hidden_layer_sizes,
                        activation="relu",
                        solver="adam",
                        alpha=1.0e-4,
                        batch_size=min(256, max(8, train_count)),
                        learning_rate_init=5.0e-4,
                        max_iter=config.max_iter,
                        early_stopping=use_early_stopping,
                        validation_fraction=0.15,
                        n_iter_no_change=50 if use_early_stopping else 15,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
        # Train one residual head per physical output.  Independent heads
        # prevent the much larger length residual from dominating pressure.
        model = MultiOutputRegressor(estimator, n_jobs=-1)
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = frame[TARGET_COLUMNS].to_numpy(dtype=float)
        target_mode = "relative_to_pkn"
        target_scales = _target_scales(frame)
        y_model = y / target_scales
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_model[train_mask])
        model.fit(x[train_mask], y_train_scaled)
        predictions = target_scaler.inverse_transform(model.predict(x)) * target_scales

        metrics: dict[str, object] = {
            "samples": int(len(frame)),
            "train_samples": int(train_mask.sum()),
            "validation_samples": int(valid_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "scenario_groups": int(len(np.unique(groups))),
            "split_method": "strict_scenario_group_holdout_70_15_15",
            "split_manifest": split_info,
            "early_stopping": use_early_stopping,
            "network": {
                "type": "MultiOutputRegressor(MLPRegressor)",
                "hidden_layer_sizes": list(config.hidden_layer_sizes),
                "independent_target_heads": TARGET_COLUMNS,
                "target_scaling": "StandardScaler_fit_on_train_groups_only",
            },
            "train_r2": _r2_by_target(y[train_mask], predictions[train_mask]),
            "validation_r2": _r2_by_target(y[valid_mask], predictions[valid_mask]),
            "test_r2": _r2_by_target(y[test_mask], predictions[test_mask]),
            "test_mae": _mae_by_target(y[test_mask], predictions[test_mask]),
            "test_max_abs_error": _max_error_by_target(y[test_mask], predictions[test_mask]),
            "test_by_scenario_type": _metrics_by_scenario_type(
                frame, y, predictions, test_mask
            ),
            "required_online_gate_targets": GATED_TARGETS,
        }
        return cls(
            model,
            target_scaler,
            metrics,
            metadata={
                "teacher": "PyFrac",
                "backbone": "PKN",
                "target": "PyFrac minus PKN residual",
                "target_mode": target_mode,
                "target_scale": {
                    "delta_length_m": "max(abs(pkn_half_length_m), 25m)",
                    "delta_aperture_mm": "max(abs(pkn_aperture_mm), 0.25mm)",
                    "delta_pressure_mpa": "max(abs(pkn_pressure_mpa), 5MPa)",
                },
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
                "gated_targets": GATED_TARGETS,
                "scenario_type_codes": SCENARIO_TYPE_CODES,
            },
        )

    def online_readiness(self, min_test_r2: float | None = None) -> dict[str, object]:
        """Return a conservative gate for online EnKF integration.

        Length and pressure are mandatory. Aperture is reported but is not a
        required gate because the current contract's online observation path
        does not have an independent aperture truth channel.
        """

        threshold = float(min_test_r2 if min_test_r2 is not None else 0.80)
        r2 = self.metrics.get("test_r2", {})
        values = {column: r2.get(column) for column in GATED_TARGETS}
        ready = bool(values) and all(
            value is not None and np.isfinite(float(value)) and float(value) >= threshold
            for value in values.values()
        )
        return {
            "ready": ready,
            "min_required_test_r2": threshold,
            "required_targets": GATED_TARGETS,
            "test_r2": r2,
            "reason": (
                "length and pressure residual held-out R2 gates passed"
                if ready
                else "length or pressure residual held-out R2 gate not met"
            ),
        }

    def predict_delta(self, features: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame: pd.DataFrame | None = None
        if isinstance(features, pd.DataFrame):
            frame = _prepare_feature_frame(features)
            values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        else:
            values = np.asarray(features, dtype=float)
        prediction_scaled = np.asarray(self.model.predict(values), dtype=float)
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        if self.metadata.get("target_mode") == "relative_to_pkn":
            if frame is None:
                raise ValueError("Relative residual surrogate requires a DataFrame with PKN baseline columns")
            prediction = prediction * _target_scales(frame)
        return pd.DataFrame(prediction, columns=TARGET_COLUMNS)

    def predict_online_state(
        self,
        features: pd.DataFrame,
        pkn_half_length_m: np.ndarray | pd.Series,
        pkn_aperture_mm: np.ndarray | pd.Series,
        pkn_pressure_mpa: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Apply residuals after a PKN run; never overwrite length directly."""

        prepared = features.copy()
        prepared["pkn_half_length_m"] = np.asarray(pkn_half_length_m, dtype=float)
        prepared["pkn_aperture_mm"] = np.asarray(pkn_aperture_mm, dtype=float)
        prepared["pkn_pressure_mpa"] = np.asarray(pkn_pressure_mpa, dtype=float)
        delta = self.predict_delta(prepared)
        result = pd.DataFrame(
            {
                "half_length_m": np.asarray(pkn_half_length_m, dtype=float) + delta["delta_length_m"].to_numpy(),
                "max_aperture_mm": np.asarray(pkn_aperture_mm, dtype=float) + delta["delta_aperture_mm"].to_numpy(),
                "net_pressure_mpa": np.asarray(pkn_pressure_mpa, dtype=float) + delta["delta_pressure_mpa"].to_numpy(),
            }
        )
        result["half_length_m"] = result["half_length_m"].clip(lower=0.0)
        result["max_aperture_mm"] = result["max_aperture_mm"].clip(lower=0.0)
        return result

    def save(self, path: str | Path) -> None:
        payload = {
            "model": self.model,
            "target_scaler": self.target_scaler,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "format_version": 2,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        path.with_suffix(".metrics.json").write_text(
            json.dumps(
                {"metrics": self.metrics, "metadata": self.metadata},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PyFracResidualSurrogate":
        payload = joblib.load(path)
        # Loading old artifacts remains possible.  Their raw-output model is
        # not approved for the new online gate until retrained.
        target_scaler = payload.get("target_scaler")
        if target_scaler is None:
            target_scaler = _identity_target_scaler(len(TARGET_COLUMNS))
        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("format_version", payload.get("format_version", 1))
        return cls(payload["model"], target_scaler, payload["metrics"], metadata)


def generate_teacher_dataset(
    output_path: str | Path,
    samples: int = 96,
    seed: int = 20260810,
    pyfrac_mode: str = "snapshot",
    steps_per_scenario: int = 4,
    n_clusters: int = 6,
) -> pd.DataFrame:
    """Generate grouped PyFrac teacher data over deliberately varied cases.

    ``samples`` now means scenario groups. Each scenario contains multiple
    time states and six cluster rows. This makes the default dataset much
    larger than the old 24-row prototype while preserving group isolation.
    """

    rng = np.random.default_rng(seed)
    adapter = PyFracAdapter(PyFracConfig())
    scenario_types = list(SCENARIO_TYPE_CODES)
    rows: list[dict[str, object]] = []
    for scenario_id in range(max(12, int(samples))):
        scenario_type = scenario_types[scenario_id % len(scenario_types)]
        base = _sample_base_scenario(rng, scenario_type, n_clusters)
        allocation = _allocation_profile(scenario_type, n_clusters, rng)
        previous_q = base["q_total_m3_s"]
        previous_stress = base["min_stress_mpa"]
        previous_viscosity = base["viscosity_pa_s"]
        for local_step in range(max(2, int(steps_per_scenario))):
            time_s = float(np.exp(rng.uniform(np.log(90.0), np.log(7200.0))))
            q_total, q_ramp = _rate_state(scenario_type, base["q_total_m3_s"], previous_q, local_step, time_s)
            leakoff = _leakoff_state(scenario_type, base["leakoff_m_sqrt_s"], local_step)
            stress = _stress_state(scenario_type, base["min_stress_mpa"], local_step)
            viscosity = _viscosity_state(scenario_type, base["viscosity_pa_s"], local_step)
            cumulative_volume = q_total * time_s
            allocation = _smooth_allocation(allocation, rng, scenario_type)
            entropy = _allocation_entropy(allocation)
            shadow = _stress_shadow_from_allocation(allocation, base["cluster_spacing_m"])
            allocation_values = dict(zip(ALLOCATION_COLUMNS, allocation))
            for cluster_index in range(n_clusters):
                q_cluster = q_total * allocation[cluster_index]
                pkn_w, pkn_l = calc_pkn(
                    np.asarray([q_cluster]),
                    viscosity,
                    base["e_prime_gpa"] * 1.0e9,
                    base["height_m"],
                    time_s,
                )
                pkn_pressure = stress + 30.0 * float(pkn_w[0]) * 1000.0 / max(base["height_m"], 1.0)
                result = adapter.run(
                    injection_rate_m3_s=q_cluster,
                    time_s=time_s,
                    mode=pyfrac_mode,
                    height_m=base["height_m"],
                    viscosity_pa_s=viscosity,
                    e_prime_pa=base["e_prime_gpa"] * 1.0e9,
                    leakoff_coefficient_m_sqrt_s=leakoff,
                    min_horizontal_stress_pa=stress * 1.0e6,
                    fracture_toughness_pa_sqrt_m=base["fracture_toughness_pa_sqrt_m"],
                )
                if not result.success:
                    continue
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_group": f"scenario_{scenario_id:05d}_{scenario_type}",
                        "scenario_type": scenario_type,
                        "e_prime_gpa": base["e_prime_gpa"],
                        "leakoff_m_sqrt_s": leakoff,
                        "viscosity_pa_s": viscosity,
                        "min_stress_mpa": stress,
                        "fracture_toughness_pa_sqrt_m": base["fracture_toughness_pa_sqrt_m"],
                        "height_m": base["height_m"],
                        "q_m3_s": q_cluster,
                        "q_total_m3_s": q_total,
                        "q_ramp_m3_s2": q_ramp,
                        "time_s": time_s,
                        "cumulative_injection_m3": cumulative_volume,
                        "cluster_spacing_m": base["cluster_spacing_m"],
                        "cluster_id": cluster_index + 1,
                        "allocation_weight": allocation[cluster_index],
                        "leakoff_volume_fraction": min(
                            0.95,
                            leakoff * np.sqrt(time_s) * 12.0 / max(q_total, 1.0e-6),
                        ),
                        "stress_delta_mpa": stress - previous_stress,
                        "viscosity_ratio": viscosity / max(previous_viscosity, 1.0e-9),
                        "allocation_entropy": entropy,
                        "allocation_min": float(np.min(allocation)),
                        "allocation_max": float(np.max(allocation)),
                        "allocation_imbalance": float(np.max(allocation) - np.min(allocation)),
                        "stress_shadow_mpa": shadow[cluster_index],
                        "scenario_type_code": SCENARIO_TYPE_CODES[scenario_type],
                        "pkn_half_length_m": float(pkn_l[0]),
                        "pyfrac_half_length_m": result.half_length_m,
                        "pkn_aperture_mm": float(2.0 * pkn_w[0] * 1000.0),
                        "pyfrac_aperture_mm": result.max_aperture_mm,
                        "pkn_pressure_mpa": pkn_pressure,
                        "pyfrac_pressure_mpa": result.bottomhole_pressure_mpa,
                        "delta_length_m": result.half_length_m - float(pkn_l[0]),
                        "delta_aperture_mm": result.max_aperture_mm - float(2.0 * pkn_w[0] * 1000.0),
                        "delta_pressure_mpa": result.bottomhole_pressure_mpa - pkn_pressure,
                        "pyfrac_runtime_seconds": result.runtime_seconds,
                        "q_over_viscosity": q_total / max(viscosity, 1.0e-9),
                        "pressure_stress_ratio": pkn_pressure / max(stress, 1.0e-6),
                        "sqrt_time_s": np.sqrt(max(time_s, 1.0)),
                        "log_time_s": np.log1p(max(time_s, 0.0)),
                        "leakoff_sqrt_time": leakoff * np.sqrt(max(time_s, 1.0)),
                        "rate_change_fraction": q_ramp / max(q_total, 1.0e-9),
                        **allocation_values,
                    }
                )
            previous_q = q_total
            previous_stress = stress
            previous_viscosity = viscosity
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("PyFrac teacher generation returned no successful samples")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def _prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill engineered columns so old online callers remain compatible."""

    result = frame.copy()
    defaults: dict[str, float] = {
        "e_prime_gpa": 32.0,
        "leakoff_m_sqrt_s": 1.0e-6,
        "viscosity_pa_s": 0.1,
        "min_stress_mpa": 60.0,
        "fracture_toughness_pa_sqrt_m": 5.0e5,
        "height_m": 30.0,
        "q_m3_s": 0.01,
        "time_s": 600.0,
        "cluster_spacing_m": 25.0,
        "allocation_weight": 1.0 / 6.0,
        "cluster_id": 1.0,
        "q_total_m3_s": np.nan,
        "cumulative_injection_m3": np.nan,
        "q_ramp_m3_s2": 0.0,
        "leakoff_volume_fraction": 0.0,
        "stress_delta_mpa": 0.0,
        "viscosity_ratio": 1.0,
        "allocation_entropy": 1.0,
        "allocation_min": 1.0 / 6.0,
        "allocation_max": 1.0 / 6.0,
        "allocation_imbalance": 0.0,
        "stress_shadow_mpa": 0.0,
        "scenario_type_code": 0.0,
        "pkn_half_length_m": 0.0,
        "pkn_aperture_mm": 0.0,
        "pkn_pressure_mpa": 0.0,
        "q_over_viscosity": 0.1,
        "pressure_stress_ratio": 1.0,
        "sqrt_time_s": np.sqrt(600.0),
        "log_time_s": np.log1p(600.0),
        "leakoff_sqrt_time": 1.0e-6 * np.sqrt(600.0),
        "rate_change_fraction": 0.0,
    }
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = defaults.get(column, np.nan)
    for index, column in enumerate(ALLOCATION_COLUMNS, start=1):
        if column not in frame:
            result[column] = pd.to_numeric(result["allocation_weight"], errors="coerce").fillna(1.0 / 6.0)
    numeric = result.apply(pd.to_numeric, errors="coerce")
    q_total = numeric["q_total_m3_s"].fillna(
        numeric["q_m3_s"] / np.maximum(numeric["allocation_weight"], 1.0e-6)
    )
    numeric["q_total_m3_s"] = q_total
    numeric["cumulative_injection_m3"] = numeric["cumulative_injection_m3"].fillna(
        q_total * np.maximum(numeric["time_s"], 1.0)
    )
    allocation_matrix = numeric[ALLOCATION_COLUMNS].to_numpy(dtype=float)
    allocation_matrix = np.maximum(allocation_matrix, 1.0e-9)
    allocation_matrix /= np.maximum(allocation_matrix.sum(axis=1, keepdims=True), 1.0e-9)
    entropy = -np.sum(allocation_matrix * np.log(allocation_matrix), axis=1) / np.log(6.0)
    numeric["allocation_entropy"] = entropy
    numeric["allocation_min"] = allocation_matrix.min(axis=1)
    numeric["allocation_max"] = allocation_matrix.max(axis=1)
    numeric["allocation_imbalance"] = numeric["allocation_max"] - numeric["allocation_min"]
    numeric["q_over_viscosity"] = numeric["q_total_m3_s"] / np.maximum(
        numeric["viscosity_pa_s"], 1.0e-9
    )
    numeric["pressure_stress_ratio"] = numeric["pkn_pressure_mpa"] / np.maximum(
        numeric["min_stress_mpa"], 1.0e-6
    )
    numeric["sqrt_time_s"] = np.sqrt(np.maximum(numeric["time_s"], 1.0))
    numeric["log_time_s"] = np.log1p(np.maximum(numeric["time_s"], 0.0))
    numeric["leakoff_sqrt_time"] = numeric["leakoff_m_sqrt_s"] * numeric["sqrt_time_s"]
    numeric["rate_change_fraction"] = numeric["q_ramp_m3_s2"] / np.maximum(
        numeric["q_total_m3_s"], 1.0e-9
    )
    numeric = numeric[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Keep targets and split metadata alongside the numeric feature matrix.
    # Online callers only consume FEATURE_COLUMNS, while training needs the
    # target and scenario columns for strict group evaluation.
    prepared = result.copy()
    for column in FEATURE_COLUMNS:
        prepared[column] = numeric[column]
    return prepared


def _group_values(frame: pd.DataFrame) -> np.ndarray:
    if "scenario_group" in frame:
        return frame["scenario_group"].fillna(frame.get("scenario_id", np.arange(len(frame)))).astype(str).to_numpy()
    if "scenario_id" in frame:
        return frame["scenario_id"].fillna(-1).astype(str).to_numpy()
    return np.arange(len(frame)).astype(str)


def _strict_group_split(
    groups: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    unique = np.unique(groups.astype(str))
    if len(unique) < 6:
        raise ValueError("At least 6 scenario groups are required for strict 70/15/15 holdout")
    rng = np.random.default_rng(random_state)
    ordered = rng.permutation(unique)
    n = len(ordered)
    n_train = max(1, int(np.floor(0.70 * n)))
    n_valid = max(1, int(np.floor(0.15 * n)))
    if n_train + n_valid >= n:
        n_valid = 1
        n_train = n - 2
    train_groups = ordered[:n_train]
    valid_groups = ordered[n_train : n_train + n_valid]
    test_groups = ordered[n_train + n_valid :]
    train_mask = np.isin(groups, train_groups)
    valid_mask = np.isin(groups, valid_groups)
    test_mask = np.isin(groups, test_groups)
    manifest = {
        "train": train_groups.astype(str).tolist(),
        "validation": valid_groups.astype(str).tolist(),
        "test": test_groups.astype(str).tolist(),
    }
    return train_mask, valid_mask, test_mask, manifest


def _r2_by_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        column: _safe_r2(y_true[:, index], y_pred[:, index])
        for index, column in enumerate(TARGET_COLUMNS)
    }


def _mae_by_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        column: float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        for index, column in enumerate(TARGET_COLUMNS)
    }


def _max_error_by_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        column: float(np.max(np.abs(y_true[:, index] - y_pred[:, index])))
        for index, column in enumerate(TARGET_COLUMNS)
    }


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return None
    value = float(r2_score(y_true, y_pred))
    return value if np.isfinite(value) else None


def _metrics_by_scenario_type(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    if "scenario_type" not in frame:
        return {}
    result: dict[str, object] = {}
    for scenario_type in sorted(frame.loc[mask, "scenario_type"].astype(str).unique()):
        type_mask = mask & (frame["scenario_type"].astype(str).to_numpy() == scenario_type)
        result[scenario_type] = {
            "samples": int(type_mask.sum()),
            "r2": _r2_by_target(y_true[type_mask], y_pred[type_mask]) if type_mask.sum() >= 2 else {},
            "mae": _mae_by_target(y_true[type_mask], y_pred[type_mask]) if type_mask.sum() else {},
        }
    return result


def _identity_target_scaler(width: int) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(width, dtype=float)
    scaler.scale_ = np.ones(width, dtype=float)
    scaler.var_ = np.ones(width, dtype=float)
    scaler.n_features_in_ = width
    scaler.n_samples_seen_ = 1
    return scaler


def _target_scales(frame: pd.DataFrame) -> np.ndarray:
    """Return physical scales used to normalize residual targets."""

    prepared = frame
    length = np.maximum(np.abs(pd.to_numeric(prepared["pkn_half_length_m"], errors="coerce")), 25.0)
    aperture = np.maximum(np.abs(pd.to_numeric(prepared["pkn_aperture_mm"], errors="coerce")), 0.25)
    pressure = np.maximum(np.abs(pd.to_numeric(prepared["pkn_pressure_mpa"], errors="coerce")), 5.0)
    return np.column_stack([length, aperture, pressure]).astype(float)


def _sample_base_scenario(rng: np.random.Generator, scenario_type: str, n_clusters: int) -> dict[str, float]:
    return {
        "e_prime_gpa": float(rng.uniform(20.0, 45.0)),
        "leakoff_m_sqrt_s": float(np.exp(rng.uniform(np.log(2.0e-7), np.log(5.0e-6)))),
        "viscosity_pa_s": float(np.exp(rng.uniform(np.log(0.025), np.log(0.30)))),
        "min_stress_mpa": float(rng.uniform(48.0, 92.0)),
        "fracture_toughness_pa_sqrt_m": float(rng.uniform(2.0e5, 9.0e5)),
        "height_m": float(rng.uniform(24.0, 42.0)),
        "q_total_m3_s": float(np.exp(rng.uniform(np.log(0.012), np.log(0.22)))),
        "cluster_spacing_m": float(rng.uniform(15.0, 38.0)),
    }


def _allocation_profile(
    scenario_type: str,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    profiles = {
        "uniform_allocation": [1, 1, 1, 1, 1, 1],
        "heel_dominant": [1.70, 1.35, 1.00, 0.80, 0.65, 0.50],
        "toe_dominant": [0.50, 0.65, 0.80, 1.00, 1.35, 1.70],
        "middle_dominant": [0.60, 0.90, 1.50, 1.50, 0.90, 0.60],
        "alternating_allocation": [1.45, 0.65, 1.35, 0.65, 1.35, 0.65],
        "edge_dominant": [1.50, 0.78, 0.70, 0.70, 0.78, 1.50],
    }
    if scenario_type in profiles:
        values = np.asarray(profiles[scenario_type], dtype=float)
    elif scenario_type == "baseline":
        values = rng.uniform(0.85, 1.15, n_clusters)
    else:
        values = rng.lognormal(0.0, 0.28, n_clusters)
    values = np.resize(values, n_clusters)
    return values / np.maximum(values.sum(), 1.0e-12)


def _smooth_allocation(
    allocation: np.ndarray,
    rng: np.random.Generator,
    scenario_type: str,
) -> np.ndarray:
    if scenario_type in {"rate_step_up", "rate_step_down", "enhanced_leakoff", "stress_shift", "viscosity_shift"}:
        # Small measurement-like variation, followed by conservation normalization.
        values = allocation * np.exp(rng.normal(0.0, 0.025, len(allocation)))
        return values / np.maximum(values.sum(), 1.0e-12)
    return allocation / np.maximum(allocation.sum(), 1.0e-12)


def _rate_state(
    scenario_type: str,
    base_q: float,
    previous_q: float,
    step: int,
    time_s: float,
) -> tuple[float, float]:
    if scenario_type == "rate_step_up":
        multiplier = [0.70, 1.0, 1.30, 1.55][min(step, 3)]
    elif scenario_type == "rate_step_down":
        multiplier = [1.45, 1.15, 0.85, 0.65][min(step, 3)]
    elif scenario_type == "rate_pulse":
        multiplier = [0.85, 1.55, 0.75, 1.20][min(step, 3)]
    elif scenario_type == "combined_stress_rate":
        multiplier = [0.75, 1.0, 1.35, 1.15][min(step, 3)]
    else:
        multiplier = 1.0 + 0.04 * np.sin(step + time_s / 1200.0)
    current = max(base_q * multiplier, 1.0e-5)
    return float(current), float((current - previous_q) / max(time_s, 1.0))


def _leakoff_state(scenario_type: str, base: float, step: int) -> float:
    multiplier = 2.5 if scenario_type in {"enhanced_leakoff", "combined_leakoff_viscosity"} else 1.0
    return float(base * multiplier * (1.0 + 0.08 * step))


def _stress_state(scenario_type: str, base: float, step: int) -> float:
    if scenario_type in {"stress_shift", "combined_stress_rate"}:
        return float(base + [-6.0, -2.0, 5.0, 10.0][min(step, 3)])
    return float(base)


def _viscosity_state(scenario_type: str, base: float, step: int) -> float:
    if scenario_type in {"viscosity_shift", "combined_leakoff_viscosity"}:
        return float(base * [0.75, 0.90, 1.20, 1.55][min(step, 3)])
    return float(base)


def _allocation_entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=float), 1.0e-12)
    values = values / values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def _stress_shadow_from_allocation(values: np.ndarray, spacing: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    distance = np.abs(np.arange(len(values))[:, None] - np.arange(len(values))[None, :])
    kernel = np.exp(-distance / max(spacing / 20.0, 1.0))
    np.fill_diagonal(kernel, 0.0)
    return 4.0 * (kernel @ values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "train"], required=True)
    parser.add_argument("--teacher-csv", default="outputs/dt/pyfrac_teacher/teacher_samples.csv")
    parser.add_argument("--model-out", default="outputs/dt/pyfrac_teacher/pyfrac_residual_surrogate.joblib")
    parser.add_argument("--samples", type=int, default=96, help="number of scenario groups")
    parser.add_argument("--steps-per-scenario", type=int, default=4)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--pyfrac-mode", choices=["snapshot", "native"], default="snapshot")
    parser.add_argument("--min-test-r2", type=float, default=0.80)
    args = parser.parse_args()
    if args.mode == "generate":
        started = time.perf_counter()
        frame = generate_teacher_dataset(
            args.teacher_csv,
            samples=args.samples,
            pyfrac_mode=args.pyfrac_mode,
            steps_per_scenario=args.steps_per_scenario,
            n_clusters=args.n_clusters,
        )
        print(
            json.dumps(
                {
                    "teacher_csv": str(Path(args.teacher_csv).resolve()),
                    "samples": len(frame),
                    "scenario_groups": int(frame["scenario_group"].nunique()),
                    "scenario_types": sorted(frame["scenario_type"].unique().tolist()),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    frame = pd.read_csv(args.teacher_csv)
    surrogate = PyFracResidualSurrogate.train(
        frame,
        SurrogateConfig(min_test_r2=args.min_test_r2),
    )
    surrogate.save(args.model_out)
    print(json.dumps({"model_out": str(Path(args.model_out).resolve()), "metrics": surrogate.metrics, "gate": surrogate.online_readiness(args.min_test_r2)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
