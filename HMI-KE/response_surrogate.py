from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


@dataclass
class ConstantProbabilityModel:
    """Backward-compatible reader for pre-v3 one-class joblib artifacts.

    New training never creates this model. ``load`` disables it so a legacy
    constant prediction cannot be mistaken for a learned risk classifier.
    """

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

    version = "action_response_surrogate_v3_safety_gated"
    risk_label_policy = "explicit_abnormal_labels_only"

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
        self.label_statistics: dict[str, dict[str, object]] = {}

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
            # A one-class label set cannot support a risk classifier.  Keep the
            # model unavailable so deployment can explicitly fall back to the
            # PKN-EnKF/rule path instead of treating a constant as evidence.
            return None
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
        abnormal_target = np.asarray(meta["future_abnormal"], dtype=int)
        sand_plug_target = np.asarray(meta["future_sand_plug"], dtype=int)
        self.abnormal_model = self._fit_classifier(design, abnormal_target, self.seed)
        self.sand_plug_model = self._fit_classifier(design, sand_plug_target, self.seed + 1)
        self.label_statistics = {
            "abnormal": {
                "positive_count": int(np.sum(abnormal_target == 1)),
                "negative_count": int(np.sum(abnormal_target == 0)),
                "available": bool(self.abnormal_model is not None),
                "reason": None if self.abnormal_model is not None else "one_class_labels",
            },
            "sand_plug": {
                "positive_count": int(np.sum(sand_plug_target == 1)),
                "negative_count": int(np.sum(sand_plug_target == 0)),
                "available": bool(self.sand_plug_model is not None),
                "reason": None if self.sand_plug_model is not None else "one_class_labels",
            },
        }
        return self

    @staticmethod
    def _predict_probability(model, design: np.ndarray) -> np.ndarray:
        if model is None:
            return np.full(len(design), np.nan, dtype=float)
        return model.predict_proba(design)[:, 1]

    def predict_batch(self, x: np.ndarray, meta, actions: np.ndarray) -> dict[str, np.ndarray]:
        design = self._design(x, meta, actions)
        current_pressure = np.asarray(meta["current_pressure"], dtype=float)
        return {
            "pressure_mean": current_pressure + self.pressure_mean_model.predict(design),
            "pressure_max": current_pressure + self.pressure_max_model.predict(design),
            "abnormal_probability": self._predict_probability(self.abnormal_model, design),
            "sand_plug_probability": self._predict_probability(self.sand_plug_model, design),
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
            available = self.abnormal_model is not None if name == "abnormal_probability" else self.sand_plug_model is not None
            metrics[name] = {
                "positive_rate": float(np.mean(target)),
                "available": bool(available and len(np.unique(target)) > 1),
                "roc_auc": float(roc_auc_score(target, pred[name]))
                if available and len(np.unique(target)) > 1 else None,
                "reason": None
                if available and len(np.unique(target)) > 1
                else "one_class_labels_or_model_unavailable",
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
        # Artifacts produced before v3 used a constant probability model when
        # labels had only one class.  Keep them loadable, but disable the
        # constant risk output and force the DT environment to use its guarded
        # PKN-EnKF/rule path.
        for name in ("abnormal_model", "sand_plug_model"):
            if isinstance(getattr(model, name, None), ConstantProbabilityModel):
                setattr(model, name, None)
        if not hasattr(model, "label_statistics"):
            model.label_statistics = {}
        return model
