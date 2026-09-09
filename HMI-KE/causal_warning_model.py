from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


def compact_history_indices(feature_names: list[str]) -> np.ndarray:
    """Return indices of statistics computed from the completed history window."""

    suffixes = ("_last", "_mean", "_std", "_slope")
    return np.asarray(
        [index for index, name in enumerate(feature_names) if name.endswith(suffixes)],
        dtype=int,
    )


class CausalWarningModel300s:
    """Predict labels in the next 300 seconds from data available at decision time.

    Unlike the action-response surrogate, this warning model never receives the
    measured flow or sand ratio from the future target window.  Its inputs are
    completed-history statistics and the current operating point only.
    """

    version = "causal_warning_300s_v1"
    input_contract = "past_window_and_current_operating_point_only"
    risk_label_policy = "explicit_abnormal_labels_only"

    def __init__(self, feature_names: list[str], seed: int = 2026) -> None:
        self.feature_names = list(feature_names)
        self.history_indices = compact_history_indices(self.feature_names)
        if not len(self.history_indices):
            raise ValueError("No completed-history statistic features were found.")
        self.seed = int(seed)
        self.abnormal_model = None
        self.sand_plug_model = None
        self.label_statistics: dict[str, dict[str, object]] = {}

    def _design(self, x: np.ndarray, meta) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        compact = values[:, self.history_indices]
        # Preserve temporal shape without passing a very wide one-value-per-second
        # vector to the classifier.  Every raw state channel is summarized into
        # chronological bins plus robust range/quantile descriptors.
        raw_width = int(self.history_indices[0])
        raw = values[:, :raw_width]
        state_count = 3
        state_points = raw_width // state_count
        temporal_parts: list[np.ndarray] = []
        if state_points > 0 and state_points * state_count == raw_width:
            for state_index in range(state_count):
                channel = raw[:, state_index * state_points : (state_index + 1) * state_points]
                bins = min(20, state_points)
                for index_group in np.array_split(np.arange(state_points), bins):
                    temporal_parts.append(np.nanmean(channel[:, index_group], axis=1, keepdims=True))
                temporal_parts.extend(
                    [
                        np.nanmin(channel, axis=1, keepdims=True),
                        np.nanmax(channel, axis=1, keepdims=True),
                        np.nanquantile(channel, 0.10, axis=1, keepdims=True),
                        np.nanquantile(channel, 0.50, axis=1, keepdims=True),
                        np.nanquantile(channel, 0.90, axis=1, keepdims=True),
                    ]
                )
        temporal = np.column_stack(temporal_parts) if temporal_parts else np.empty((len(values), 0))
        current = np.column_stack(
            [
                np.asarray(meta["current_pressure"], dtype=float),
                np.asarray(meta["current_flow"], dtype=float),
                np.asarray(meta["current_sand_ratio"], dtype=float),
            ]
        )
        return np.nan_to_num(
            np.column_stack([compact, temporal, current]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _fit_classifier(design: np.ndarray, target: np.ndarray, seed: int):
        if len(np.unique(target)) < 2:
            return None
        gradient_model = HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=23,
            learning_rate=0.05,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=seed,
        )
        tree_model = ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        return (
            gradient_model.fit(design, target),
            tree_model.fit(design, target),
        )

    def fit(self, x: np.ndarray, meta) -> "CausalWarningModel300s":
        design = self._design(x, meta)
        abnormal = np.asarray(meta["future_abnormal"], dtype=int)
        sand_plug = np.asarray(meta["future_sand_plug"], dtype=int)
        self.abnormal_model = self._fit_classifier(design, abnormal, self.seed)
        self.sand_plug_model = self._fit_classifier(design, sand_plug, self.seed + 1)
        self.label_statistics = {
            "abnormal": self._label_summary(abnormal, self.abnormal_model),
            "sand_plug": self._label_summary(sand_plug, self.sand_plug_model),
        }
        return self

    @staticmethod
    def _label_summary(target: np.ndarray, model) -> dict[str, object]:
        return {
            "positive_count": int(np.sum(target == 1)),
            "negative_count": int(np.sum(target == 0)),
            "available": bool(model is not None),
            "reason": None if model is not None else "one_class_labels",
        }

    @staticmethod
    def _probability(model, design: np.ndarray) -> np.ndarray:
        if model is None:
            return np.full(len(design), np.nan, dtype=float)
        if isinstance(model, tuple):
            probabilities = [estimator.predict_proba(design)[:, 1] for estimator in model]
            return np.mean(probabilities, axis=0)
        return model.predict_proba(design)[:, 1]

    def predict_batch(self, x: np.ndarray, meta) -> dict[str, np.ndarray]:
        design = self._design(x, meta)
        return {
            "abnormal_probability": self._probability(self.abnormal_model, design),
            "sand_plug_probability": self._probability(self.sand_plug_model, design),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "CausalWarningModel300s":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected warning-model artifact: {type(model)!r}")
        return model
