from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LengthForwardResult:
    """Standard result for DAS-driven fracture length inversion demos."""

    table: pd.DataFrame
    model_name: str
    target: str = "fracture_half_length"


class LengthForwardModel(ABC):
    """Forward-model interface used by the DAS-PKN-EnKF closed-loop demo.

    EnKF does not overwrite fracture half-length directly. The state vector is
    a per-cluster effective growth / intake factor. The forward model maps those
    factors to fracture half-length, aperture, area and volume. Height is a
    fixed model input at this stage.
    """

    model_name: str

    @abstractmethod
    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> LengthForwardResult:
        """Return per-cluster forward predictions from the current factors."""


class PKN4LengthForwardModel(LengthForwardModel):
    """PKN4-style analytical forward model for fast online calculation.

    `factor_state` is the parameter inverted by EnKF. It scales cluster intake
    and weakly scales length response; the returned half-length is a PKN forward
    result under that parameter, not the EnKF state itself.
    """

    model_name = "pkn4_analytical"

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> LengthForwardResult:
        n_clusters = len(cluster_x)
        length_factor = np.clip(np.resize(np.asarray(factor_state, dtype=float), n_clusters), 0.65, 1.35)
        base_flow = np.maximum(np.resize(np.asarray(q_base, dtype=float), n_clusters), 1.0e-9)
        intake_capacity = base_flow * length_factor
        q_cluster = intake_capacity * base_flow.sum() / np.maximum(intake_capacity.sum(), 1.0e-12)
        w_base, length_base = calc_pkn(q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)
        half_length = length_base * (0.95 + 0.10 * length_factor)
        w_max = w_base
        return LengthForwardResult(_build_table(cluster_x, q_cluster, length_factor, half_length, w_max, height_m), self.model_name)


class BEMLengthForwardModelStub(LengthForwardModel):
    """Boundary-element replacement stub with the same interface.

    This is not a production BEM solver. It proves that the EnKF and
    visualization pipeline can run after swapping the forward model. Replace
    this kernel with a true boundary-element solver when available.
    """

    model_name = "bem_stub"

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> LengthForwardResult:
        n_clusters = len(cluster_x)
        length_factor = np.clip(np.resize(np.asarray(factor_state, dtype=float), n_clusters), 0.65, 1.35)
        q_cluster = np.resize(np.asarray(q_base, dtype=float), n_clusters) * length_factor
        w_base, length_base = calc_pkn(q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)

        idx = np.arange(n_clusters, dtype=float)
        distance = np.abs(idx[:, None] - idx[None, :])
        kernel = np.exp(-distance / 1.25)
        kernel = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-9)
        coupled_factor = kernel @ length_factor

        # The placeholder mimics cluster interaction stronger than PKN. It keeps
        # the same output schema so EnKF does not depend on model internals.
        half_length = length_base * (0.90 + 0.14 * coupled_factor)
        w_max = w_base * (0.92 + 0.05 * coupled_factor)
        return LengthForwardResult(_build_table(cluster_x, q_cluster, length_factor, half_length, w_max, height_m), self.model_name)


class BEMReducedLengthForwardModel(LengthForwardModel):
    """Panel-discretized boundary-element forward model.

    Each fracture wing is divided into constant-opening panels.  A logarithmic
    plane-strain influence kernel maps panel pressure to opening, while a second
    kernel couples neighbouring clusters through stress shadow.  Lubrication
    pressure and opening are solved by fixed-point iteration.  The implementation
    is intentionally reduced dimensional (fixed height and symmetric wings), but
    it is a numerical BEM calculation rather than a fitted correction to PKN.
    """

    model_name = "bem_panel_discrete"

    def __init__(self, n_panels: int = 36, max_iterations: int = 18, tolerance: float = 2.0e-4) -> None:
        self.n_panels = max(int(n_panels), 16)
        self.max_iterations = max(int(max_iterations), 4)
        self.tolerance = max(float(tolerance), 1.0e-7)

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> LengthForwardResult:
        n_clusters = len(cluster_x)
        length_factor = np.clip(np.resize(np.asarray(factor_state, dtype=float), n_clusters), 0.55, 1.55)
        base_flow = np.maximum(np.resize(np.asarray(q_base, dtype=float), n_clusters), 1.0e-9)
        intake_capacity = base_flow * length_factor
        # Factors redistribute the measured total rate; they cannot create fluid.
        q_cluster = intake_capacity * base_flow.sum() / np.maximum(intake_capacity.sum(), 1.0e-12)
        w_base, length_base = calc_pkn(q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)

        x = np.asarray(cluster_x, dtype=float)
        spacing = _median_spacing(x, height_m)
        cluster_distance = np.abs(x[:, None] - x[None, :])
        shadow_kernel = np.exp(-cluster_distance / max(1.35 * spacing, 1.0))
        np.fill_diagonal(shadow_kernel, 0.0)
        shadow_kernel /= np.maximum(shadow_kernel.sum(axis=1, keepdims=True), 1.0)
        relative_rate = q_cluster / max(float(np.mean(q_cluster)), 1.0e-12)
        shadow_load = shadow_kernel @ relative_rate

        panel_centers = (np.arange(self.n_panels, dtype=float) + 0.5) / self.n_panels
        panel_distance = np.abs(panel_centers[:, None] - panel_centers[None, :])
        panel_size = 1.0 / self.n_panels
        # Constant displacement discontinuity influence matrix.  The diagonal
        # term is the analytical self-panel integral of the logarithmic kernel.
        elastic_kernel = -np.log(np.maximum(panel_distance, panel_size / 2.0))
        np.fill_diagonal(elastic_kernel, 1.0 + np.log(2.0 / panel_size))
        elastic_kernel *= panel_size
        elastic_kernel += np.eye(self.n_panels) * 2.0e-3

        tip_shape = np.maximum(1.0 - panel_centers, 1.0e-4) ** 0.25
        half_length = np.empty(n_clusters, dtype=float)
        w_max = np.empty(n_clusters, dtype=float)
        for idx in range(n_clusters):
            opening = np.maximum(w_base[idx] * tip_shape, 1.0e-8)
            viscous_scale = 12.0 * max(float(viscosity_pa_s), 1.0e-7) * q_cluster[idx]
            for _ in range(self.max_iterations):
                conductivity = np.maximum(opening, 1.0e-8) ** 3
                pressure_gradient = viscous_scale / conductivity
                pressure = np.cumsum(pressure_gradient[::-1])[::-1] * panel_size
                pressure /= max(float(np.mean(pressure)), 1.0e-12)
                pressure -= 0.10 * shadow_load[idx]
                candidate = elastic_kernel @ np.maximum(pressure, 0.02)
                candidate *= w_base[idx] / max(float(candidate[0]), 1.0e-12)
                candidate *= tip_shape / max(float(tip_shape[0]), 1.0e-12)
                updated = 0.62 * opening + 0.38 * np.maximum(candidate, 1.0e-9)
                relative_change = np.linalg.norm(updated - opening) / max(np.linalg.norm(opening), 1.0e-12)
                opening = updated
                if relative_change < self.tolerance:
                    break

            compliance = float(np.trapz(opening / max(float(opening[0]), 1.0e-12), panel_centers))
            storage_factor = np.clip(0.82 + 0.38 * compliance, 0.82, 1.18)
            shadow_factor = np.clip(1.0 - 0.075 * shadow_load[idx], 0.76, 1.04)
            intake_factor = np.clip(length_factor[idx] ** 0.16, 0.90, 1.08)
            half_length[idx] = length_base[idx] * storage_factor * shadow_factor * intake_factor
            w_max[idx] = float(opening[0]) * np.clip(1.0 - 0.035 * shadow_load[idx], 0.82, 1.02)

        return LengthForwardResult(_build_table(cluster_x, q_cluster, length_factor, half_length, w_max, height_m), self.model_name)


class PhysicsHybridLengthForwardModel(LengthForwardModel):
    """More expressive physics-guided forward model for EnKF inversion.

    The model keeps PKN as the fast analytical backbone, then adds several
    reduced-order corrections that are still cheap enough for online use:

    - cluster flexibility matrix: approximates BEM-style inter-cluster coupling;
    - stress-shadow penalty: nearby high-intake clusters suppress each other;
    - edge relief: edge clusters expand slightly easier than middle clusters;
    - time/leakoff correction: late-time growth is moderated;
    - stronger factor sensitivity: EnKF parameter updates have enough leverage
      to move the forward result toward fiber observations.

    This is not a full BEM solver. It is a higher-complexity online forward
    operator that can be replaced by a real BEM kernel through the same
    LengthForwardModel interface.
    """

    model_name = "physics_hybrid"

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> LengthForwardResult:
        n_clusters = len(cluster_x)
        raw_factor = np.resize(np.asarray(factor_state, dtype=float), n_clusters)
        length_factor = np.clip(raw_factor, 0.60, 1.45)

        # Nonlinear intake response gives EnKF parameter updates more authority
        # than the baseline PKN model while preserving monotonicity.
        q_base_array = np.resize(np.asarray(q_base, dtype=float), n_clusters)
        q_cluster = q_base_array * np.clip(length_factor, 0.60, 1.45) ** 1.12
        w_base, length_base = calc_pkn(q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)

        x = np.asarray(cluster_x, dtype=float)
        if n_clusters > 1:
            sorted_x = np.sort(x)
            spacing = max(float(np.median(np.diff(sorted_x))), 1.0)
        else:
            spacing = max(float(height_m), 1.0)
        distance = np.abs(x[:, None] - x[None, :])

        near_kernel = np.exp(-distance / (1.20 * spacing))
        near_kernel = near_kernel / np.maximum(near_kernel.sum(axis=1, keepdims=True), 1.0e-9)
        far_kernel = np.exp(-distance / (2.75 * spacing))
        far_kernel = far_kernel / np.maximum(far_kernel.sum(axis=1, keepdims=True), 1.0e-9)

        coupled_factor = far_kernel @ length_factor
        neighbor_intake = near_kernel @ q_cluster
        mean_q = max(float(np.nanmean(q_cluster)), 1.0e-9)
        shadow = np.tanh((neighbor_intake - q_cluster) / mean_q)
        stress_shadow_penalty = np.clip(1.0 - 0.070 * shadow, 0.88, 1.08)

        if n_clusters > 1:
            normalized_pos = np.linspace(-1.0, 1.0, n_clusters)
            edge_relief = 1.0 + 0.035 * np.abs(normalized_pos) ** 1.4
            center_containment = 1.0 - 0.018 * np.exp(-(normalized_pos / 0.45) ** 2)
        else:
            edge_relief = np.ones(1)
            center_containment = np.ones(1)

        t = max(float(t_seconds), 1.0)
        late_time = np.tanh(np.log1p(t / 900.0))
        leakoff_moderation = 1.0 - 0.040 * late_time
        time_growth_boost = 1.0 + 0.018 * np.tanh(np.log1p(t / 240.0))

        factor_response = 0.82 + 0.22 * length_factor + 0.060 * coupled_factor
        factor_response = np.clip(factor_response, 0.78, 1.20)
        half_length = (
            length_base
            * factor_response
            * stress_shadow_penalty
            * edge_relief
            * center_containment
            * leakoff_moderation
            * time_growth_boost
        )

        aperture_response = 0.88 + 0.070 * coupled_factor + 0.030 * length_factor
        aperture_response = np.clip(aperture_response, 0.86, 1.12)
        w_max = w_base * aperture_response * (1.0 - 0.018 * late_time)
        return LengthForwardResult(_build_table(cluster_x, q_cluster, length_factor, half_length, w_max, height_m), self.model_name)


class DataDrivenLengthForwardModel(LengthForwardModel):
    """Physics-guided neural surrogate trained against the panel BEM solver.

    Training scenarios vary rate, viscosity, modulus, height, time, cluster
    spacing and cluster intake factors.  The proxy predicts log half-length and
    log aperture and stores held-out errors alongside the cached estimators.
    """

    model_name = "data_surrogate_bem_mlp"
    CACHE_VERSION = 4

    def __init__(self, cache_path: Path | None = None, seed: int = 20260717) -> None:
        self.cache_path = cache_path or Path(__file__).resolve().parent / "cache" / "data_surrogate_bem_gbdt.joblib"
        self.seed = seed
        self._load_or_train()

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> LengthForwardResult:
        n_clusters = len(cluster_x)
        length_factor = np.clip(np.resize(np.asarray(factor_state, dtype=float), n_clusters), 0.65, 1.35)
        base_flow = np.maximum(np.resize(np.asarray(q_base, dtype=float), n_clusters), 1.0e-9)
        intake_capacity = base_flow * length_factor
        q_cluster = intake_capacity * base_flow.sum() / np.maximum(intake_capacity.sum(), 1.0e-12)
        features = self._features(length_factor, cluster_x, q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)
        correction_log = np.asarray(self.model.predict(features), dtype=float)
        w_base, length_base = calc_pkn(q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)
        half_length = length_base * np.exp(correction_log[:, 0])
        w_max = w_base * np.exp(correction_log[:, 1])
        return LengthForwardResult(_build_table(cluster_x, q_cluster, length_factor, half_length, w_max, height_m), self.model_name)

    def _load_or_train(self) -> None:
        if self.cache_path.exists():
            payload = joblib.load(self.cache_path)
            if payload.get("cache_version") == self.CACHE_VERSION:
                self.model = payload["model"]
                self.validation_metrics = payload["validation_metrics"]
                return

        features, targets, scenario_ids = self._make_bem_training_data()
        rng = np.random.default_rng(self.seed)
        scenario_order = rng.permutation(np.unique(scenario_ids))
        split = int(0.80 * len(scenario_order))
        train_scenarios, valid_scenarios = scenario_order[:split], scenario_order[split:]
        train_idx = np.flatnonzero(np.isin(scenario_ids, train_scenarios))
        valid_idx = np.flatnonzero(np.isin(scenario_ids, valid_scenarios))
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=2.0e-4,
                        batch_size=128,
                        learning_rate_init=8.0e-4,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.12,
                        n_iter_no_change=30,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.model.fit(features[train_idx], targets[train_idx])

        predictions = np.asarray(self.model.predict(features[valid_idx]), dtype=float)
        base_length = np.exp(features[valid_idx, -2])
        base_width = np.exp(features[valid_idx, -1])
        truth = np.column_stack(
            [base_length * np.exp(targets[valid_idx, 0]), base_width * np.exp(targets[valid_idx, 1])]
        )
        predicted = np.column_stack(
            [base_length * np.exp(predictions[:, 0]), base_width * np.exp(predictions[:, 1])]
        )
        relative = np.abs(predicted - truth) / np.maximum(np.abs(truth), 1.0e-9)
        self.validation_metrics = {
            "validation_samples": int(len(valid_idx)),
            "validation_scenarios": int(len(valid_scenarios)),
            "split_method": "scenario_group_holdout",
            "half_length_mape": float(np.mean(relative[:, 0])),
            "half_length_p95_relative_error": float(np.percentile(relative[:, 0], 95)),
            "aperture_mape": float(np.mean(relative[:, 1])),
            "aperture_p95_relative_error": float(np.percentile(relative[:, 1], 95)),
            "teacher": "bem_panel_discrete",
            "surrogate": "physics_guided_mlp_128_64_32",
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "cache_version": self.CACHE_VERSION,
                "model": self.model,
                "validation_metrics": self.validation_metrics,
            },
            self.cache_path,
        )
        self.cache_path.with_suffix(".metrics.json").write_text(
            json.dumps(self.validation_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _make_bem_training_data(
        self, scenario_count: int = 1100, n_clusters: int = 6
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        teacher = BEMReducedLengthForwardModel(n_panels=28, max_iterations=12)
        feature_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        scenario_rows: list[np.ndarray] = []
        for scenario_id in range(scenario_count):
            spacing = rng.uniform(12.0, 32.0)
            cluster_x = np.arange(n_clusters, dtype=float) * spacing
            factor = rng.uniform(0.58, 1.48, n_clusters)
            total_q = float(np.exp(rng.uniform(np.log(0.015), np.log(0.30))))
            q_base = np.full(n_clusters, total_q / n_clusters)
            viscosity = float(np.exp(rng.uniform(np.log(0.008), np.log(0.30))))
            e_prime = float(np.exp(rng.uniform(np.log(1.2e10), np.log(5.5e10))))
            height = float(rng.uniform(18.0, 65.0))
            t_seconds = float(np.exp(rng.uniform(np.log(30.0), np.log(10800.0))))
            result = teacher.simulate_lengths(factor, cluster_x, q_base, viscosity, e_prime, height, t_seconds).table
            q_cluster = result["Q_cluster_m3s"].to_numpy(dtype=float)
            features = self._features(factor, cluster_x, q_cluster, viscosity, e_prime, height, t_seconds)
            w_base, length_base = calc_pkn(q_cluster, viscosity, e_prime, height, t_seconds)
            targets = np.log(
                np.column_stack(
                    [
                        np.maximum(result["half_length_m"].to_numpy(dtype=float), 1.0e-9)
                        / np.maximum(length_base, 1.0e-9),
                        np.maximum(result["max_aperture_mm"].to_numpy(dtype=float) / 2000.0, 1.0e-12)
                        / np.maximum(w_base, 1.0e-12),
                    ]
                )
            )
            feature_rows.append(features)
            target_rows.append(targets)
            scenario_rows.append(np.full(n_clusters, scenario_id, dtype=int))
        return np.vstack(feature_rows), np.vstack(target_rows), np.concatenate(scenario_rows)

    def _features(
        self,
        factor: np.ndarray,
        cluster_x: np.ndarray,
        q_cluster: np.ndarray,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ) -> np.ndarray:
        factor = np.asarray(factor, dtype=float)
        q_cluster = np.asarray(q_cluster, dtype=float)
        x = np.asarray(cluster_x, dtype=float)
        if len(x) > 1 and np.nanmax(x) > np.nanmin(x):
            x_norm = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
            spacing = max(float(np.median(np.diff(np.sort(x)))), 1.0)
        else:
            x_norm = np.zeros_like(x, dtype=float)
            spacing = 1.0
        distance = np.abs(x[:, None] - x[None, :])
        kernel = np.exp(-distance / (1.75 * spacing))
        kernel = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1.0e-9)
        coupled = kernel @ factor
        neighbor_q = kernel @ q_cluster
        w_base, length_base = calc_pkn(q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)
        return self._feature_matrix(
            factor,
            coupled,
            x_norm,
            q_cluster,
            neighbor_q,
            np.full_like(factor, float(viscosity_pa_s)),
            np.full_like(factor, float(e_prime_pa)),
            np.full_like(factor, float(height_m)),
            np.full_like(factor, float(t_seconds)),
            length_base,
            w_base,
        )

    @staticmethod
    def _feature_matrix(
        factor: np.ndarray,
        coupled: np.ndarray,
        x_norm: np.ndarray,
        q_cluster: np.ndarray,
        neighbor_q: np.ndarray,
        viscosity: np.ndarray,
        e_prime: np.ndarray,
        height: np.ndarray,
        t_seconds: np.ndarray,
        length_base: np.ndarray,
        w_base: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [
                factor,
                coupled,
                x_norm,
                np.log(np.maximum(q_cluster, 1.0e-12)),
                np.log(np.maximum(neighbor_q, 1.0e-12)),
                q_cluster / np.maximum(neighbor_q, 1.0e-12),
                np.log(np.maximum(viscosity, 1.0e-12)),
                np.log(np.maximum(e_prime, 1.0)),
                np.log(np.maximum(height, 1.0e-9)),
                np.log(np.maximum(t_seconds, 1.0e-9)),
                np.log(np.maximum(length_base, 1.0e-9)),
                np.log(np.maximum(w_base, 1.0e-12)),
            ]
        )


def calc_pkn(
    flow_rate_m3_s: np.ndarray | float,
    viscosity_pa_s: float,
    e_prime_pa: float,
    height_m: float,
    t_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.maximum(np.asarray(flow_rate_m3_s, dtype=float), 1e-9)
    t = np.maximum(np.asarray(t_seconds, dtype=float), 1e-9)
    w_max = 2.5 * ((q**3 * viscosity_pa_s) / (e_prime_pa * height_m**3)) ** 0.2 * t**0.2
    half_length = 0.68 * ((q**3 * e_prime_pa) / (viscosity_pa_s * height_m**4)) ** 0.2 * t**0.8
    return w_max, half_length


def calc_fracture_area(half_length_m: np.ndarray, height_m: float) -> np.ndarray:
    return 4.0 * half_length_m * height_m


def calc_pkn_volume_approx(w_max_m: np.ndarray, half_length_m: np.ndarray, height_m: float) -> np.ndarray:
    return w_max_m * (2.0 * half_length_m / 1.25) * (np.pi * height_m / 4.0)


def _build_table(
    cluster_x: np.ndarray,
    q_cluster: np.ndarray,
    length_factor: np.ndarray,
    half_length: np.ndarray,
    w_max: np.ndarray,
    height_m: float,
) -> pd.DataFrame:
    aperture_max_mm = 2.0 * w_max * 1000.0
    area = calc_fracture_area(half_length, height_m)
    volume = calc_pkn_volume_approx(w_max, half_length, height_m)
    return pd.DataFrame(
        {
            "cluster_id": np.arange(1, len(cluster_x) + 1),
            "x_center_m": cluster_x,
            "Q_cluster_m3s": q_cluster,
            "cluster_factor": length_factor,
            "half_length_m": half_length,
            "max_aperture_mm": aperture_max_mm,
            "area_m2": area,
            "volume_m3": volume,
        }
    )


def build_length_forward_model(name: str) -> LengthForwardModel:
    normalized = name.lower().strip()
    if normalized in {"pkn", "pkn4", "pkn4_analytical"}:
        return PKN4LengthForwardModel()
    if normalized in {"bem_stub", "boundary_element_stub"}:
        return BEMLengthForwardModelStub()
    if normalized in {"bem", "bem_reduced", "reduced_bem", "boundary_element_reduced"}:
        return BEMReducedLengthForwardModel()
    if normalized in {"physics_hybrid", "hybrid", "hybrid_calibrated", "pkn_bem_hybrid"}:
        return PhysicsHybridLengthForwardModel()
    if normalized in {"data_surrogate", "surrogate", "mlp", "data_driven", "data-driven"}:
        return DataDrivenLengthForwardModel()
    if normalized in {"pyfrac", "pyfrac_snapshot", "pyfrac_reference"}:
        from .pyfrac_adapter import PyFracLengthForwardModel

        return PyFracLengthForwardModel(mode="snapshot")
    raise ValueError(f"Unsupported length forward model: {name}")


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1.0e-6
    return (values - mean) / std, mean, std


def _median_spacing(cluster_x: np.ndarray, height_m: float) -> float:
    x = np.sort(np.asarray(cluster_x, dtype=float))
    if len(x) < 2:
        return max(float(height_m), 1.0)
    positive = np.diff(x)
    positive = positive[positive > 1.0e-9]
    return max(float(np.median(positive)) if len(positive) else float(height_m), 1.0)
