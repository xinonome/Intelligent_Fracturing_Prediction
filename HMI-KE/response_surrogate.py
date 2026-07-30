from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), np.clip(self.probability, 0.0, 1.0), dtype=float)
        return np.column_stack([1.0 - p, p])


def compact_history_indices(feature_names: list[str]) -> np.ndarray:
    suffixes = ("_last", "_mean", "_std", "_slope")
    return np.asarray([index for index, name in enumerate(feature_names) if name.endswith(suffixes)], dtype=int)


class ActionResponseSurrogate:
    """Predict 60-second operational response for a candidate flow/sand action.

    PKN-EnKF remains responsible for fracture geometry and physical parameters.
    This model only learns field-data residual response: pressure and condition risk.
    """

    version = "action_response_surrogate_v1"

    def __init__(self, feature_names: list[str], action_bounds: dict, seed: int = 2026) -> None:
        self.feature_names = list(feature_names)
        self.history_indices = compact_history_indices(feature_names)
        self.action_bounds = action_bounds
        self.seed = seed
        common = dict(max_iter=180, max_leaf_nodes=31, learning_rate=0.06, l2_regularization=1.0, random_state=seed)
        self.pressure_mean_model = HistGradientBoostingRegressor(loss="absolute_error", **common)
        self.pressure_max_model = HistGradientBoostingRegressor(loss="absolute_error", **common)
        self.abnormal_model = None
        self.sand_plug_model = None

    def _design(self, x: np.ndarray, meta, actions: np.ndarray) -> np.ndarray:
        # build_dataset always appends four statistics per state variable.
        # Reading them from the tail keeps the model compatible when two data
        # sources infer different raw points per 300-second window.
        compact_width = len(self.history_indices)
        compact = np.asarray(x, dtype=float)[:, -compact_width:]
        current_flow = np.asarray(meta["current_flow"], dtype=float)
        current_sand = np.asarray(meta["current_sand_ratio"], dtype=float)
        delta = actions - np.column_stack([current_flow, current_sand])
        return np.column_stack([compact, actions, delta])

    @staticmethod
    def _fit_classifier(design: np.ndarray, target: np.ndarray, seed: int):
        unique = np.unique(target)
        if len(unique) < 2:
            return ConstantProbabilityModel(float(np.mean(target)))
        model = HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=23, learning_rate=0.06,
            l2_regularization=1.5, class_weight="balanced", random_state=seed,
        )
        return model.fit(design, target)

    def fit(self, x: np.ndarray, meta, actions: np.ndarray) -> "ActionResponseSurrogate":
        design = self._design(x, meta, actions)
        current_pressure = np.asarray(meta["current_pressure"], dtype=float)
        self.pressure_mean_model.fit(design, np.asarray(meta["future_pressure_mean"], dtype=float) - current_pressure)
        self.pressure_max_model.fit(design, np.asarray(meta["future_pressure_max"], dtype=float) - current_pressure)
        self.abnormal_model = self._fit_classifier(design, np.asarray(meta["future_abnormal"], dtype=int), self.seed)
        self.sand_plug_model = self._fit_classifier(design, np.asarray(meta["future_sand_plug"], dtype=int), self.seed + 1)
        return self

    def predict_batch(self, x: np.ndarray, meta, actions: np.ndarray) -> dict[str, np.ndarray]:
        design = self._design(x, meta, actions)
        current_pressure = np.asarray(meta["current_pressure"], dtype=float)
        return {
            "pressure_mean": current_pressure + self.pressure_mean_model.predict(design),
            "pressure_max": current_pressure + self.pressure_max_model.predict(design),
            "abnormal_probability": self.abnormal_model.predict_proba(design)[:, 1],
            "sand_plug_probability": self.sand_plug_model.predict_proba(design)[:, 1],
        }

    def predict_one(self, x: np.ndarray, meta_row, flow: float, sand: float) -> dict[str, float]:
        predictions = self.predict_batch(
            np.asarray(x, dtype=float).reshape(1, -1),
            meta_row.to_frame().T,
            np.asarray([[flow, sand]], dtype=float),
        )
        flow_bounds = self.action_bounds.get("PL", {})
        sand_bounds = self.action_bounds.get("SB", {})
        flow_scale = max(float(flow_bounds.get("p99", flow)) - float(flow_bounds.get("p01", flow)), 1.0)
        sand_scale = max(float(sand_bounds.get("p99", sand)) - float(sand_bounds.get("p01", sand)), 1.0)
        flow_ood = max(float(flow_bounds.get("p01", flow)) - flow, 0.0, flow - float(flow_bounds.get("p99", flow))) / flow_scale
        sand_ood = max(float(sand_bounds.get("p01", sand)) - sand, 0.0, sand - float(sand_bounds.get("p99", sand))) / sand_scale
        return {key: float(value[0]) for key, value in predictions.items()} | {
            "ood_score": float(np.clip(max(flow_ood, sand_ood), 0.0, 1.0))
        }

    def evaluate(self, x: np.ndarray, meta, actions: np.ndarray) -> dict:
        pred = self.predict_batch(x, meta, actions)
        metrics = {}
        for name, target_name in (("pressure_mean", "future_pressure_mean"), ("pressure_max", "future_pressure_max")):
            target = np.asarray(meta[target_name], dtype=float)
            metrics[name] = {
                "mae": float(mean_absolute_error(target, pred[name])),
                "rmse": float(mean_squared_error(target, pred[name]) ** 0.5),
                "r2": float(r2_score(target, pred[name])),
            }
        for name, target_name in (("abnormal_probability", "future_abnormal"), ("sand_plug_probability", "future_sand_plug")):
            target = np.asarray(meta[target_name], dtype=int)
            metrics[name] = {
                "positive_rate": float(np.mean(target)),
                "roc_auc": float(roc_auc_score(target, pred[name])) if len(np.unique(target)) > 1 else None,
            }
        return metrics

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "ActionResponseSurrogate":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected surrogate type: {type(model)!r}")
        return model
