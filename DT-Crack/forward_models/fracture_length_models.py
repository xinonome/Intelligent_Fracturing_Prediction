from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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
        q_cluster = np.resize(np.asarray(q_base, dtype=float), n_clusters) * length_factor
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
    """Reduced-order boundary-element style forward model.

    This is a fast online surrogate of BEM behavior, not a full BEM solver. It
    builds a flexibility matrix from cluster spacing, applies cluster interaction
    and stress-shadow style penalties, and keeps the same output schema as PKN.
    The purpose is to validate that the EnKF loop can swap in a more complex
    forward operator without changing the rest of the pipeline.
    """

    model_name = "bem_reduced"

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

        x = np.asarray(cluster_x, dtype=float)
        if n_clusters > 1:
            spacing = np.median(np.diff(np.sort(x)))
        else:
            spacing = max(float(height_m), 1.0)
        spacing = max(float(spacing), 1.0)
        distance = np.abs(x[:, None] - x[None, :])
        flexibility = np.exp(-distance / (1.75 * spacing))
        flexibility = flexibility / np.maximum(flexibility.sum(axis=1, keepdims=True), 1e-9)

        coupled_factor = flexibility @ length_factor
        local_shadow = flexibility @ q_cluster - q_cluster
        shadow_penalty = 1.0 - 0.045 * np.tanh(local_shadow / np.maximum(np.mean(q_cluster), 1e-9))
        edge_relief = 1.0 + 0.025 * np.abs(np.linspace(-1.0, 1.0, n_clusters))

        half_length = length_base * (0.91 + 0.12 * coupled_factor) * shadow_penalty * edge_relief
        w_max = w_base * (0.94 + 0.045 * coupled_factor)
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
    """Cached data-driven surrogate for length and aperture prediction.

    The model is a small random-feature neural surrogate trained from synthetic
    PKN/reduced-BEM samples. It is intentionally lightweight so the benchmark can
    validate the "data-driven forward model" path without adding training-time
    dependencies or changing the EnKF interface.
    """

    model_name = "data_surrogate"

    def __init__(self, cache_path: Path | None = None, seed: int = 20260717, hidden_dim: int = 96) -> None:
        self.cache_path = cache_path or Path(__file__).resolve().parent / "cache" / "data_surrogate_mlp.npz"
        self.seed = seed
        self.hidden_dim = hidden_dim
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
        q_cluster = np.resize(np.asarray(q_base, dtype=float), n_clusters) * length_factor
        features = self._features(length_factor, cluster_x, q_cluster, viscosity_pa_s, e_prime_pa, height_m, t_seconds)
        hidden = np.tanh((features - self.x_mean) / self.x_std @ self.hidden_w + self.hidden_b)
        y_scaled = hidden @ self.beta
        y_log = y_scaled * self.y_std + self.y_mean
        half_length = np.exp(y_log[:, 0])
        w_max = np.exp(y_log[:, 1])
        return LengthForwardResult(_build_table(cluster_x, q_cluster, length_factor, half_length, w_max, height_m), self.model_name)

    def _load_or_train(self) -> None:
        if self.cache_path.exists():
            payload = np.load(self.cache_path)
            self.x_mean = payload["x_mean"]
            self.x_std = payload["x_std"]
            self.y_mean = payload["y_mean"]
            self.y_std = payload["y_std"]
            self.hidden_w = payload["hidden_w"]
            self.hidden_b = payload["hidden_b"]
            self.beta = payload["beta"]
            return

        rng = np.random.default_rng(self.seed)
        sample_count = 7000
        n_clusters = 6
        factor = rng.uniform(0.65, 1.35, sample_count)
        q_cluster = np.exp(rng.uniform(np.log(1.0e-4), np.log(0.18), sample_count))
        viscosity = np.exp(rng.uniform(np.log(0.01), np.log(0.25), sample_count))
        e_prime = np.exp(rng.uniform(np.log(1.5e10), np.log(4.5e10), sample_count))
        height = rng.uniform(20.0, 60.0, sample_count)
        t_seconds = np.exp(rng.uniform(np.log(10.0), np.log(7200.0), sample_count))
        cluster_x = rng.uniform(0.0, 120.0, sample_count)
        coupled = np.clip(0.72 * factor + 0.28 * rng.uniform(0.65, 1.35, sample_count), 0.65, 1.35)

        features = self._feature_matrix(factor, coupled, cluster_x / 120.0, q_cluster, viscosity, e_prime, height, t_seconds)
        w_base, length_base = calc_pkn(q_cluster, viscosity, e_prime, height, t_seconds)
        half_length = length_base * (0.92 + 0.12 * coupled) * (0.985 + 0.03 * rng.random(sample_count))
        w_max = w_base * (0.95 + 0.055 * coupled) * (0.99 + 0.02 * rng.random(sample_count))
        targets = np.log(np.column_stack([np.maximum(half_length, 1.0e-9), np.maximum(w_max, 1.0e-12)]))

        self.x_mean = features.mean(axis=0, keepdims=True)
        self.x_std = features.std(axis=0, keepdims=True) + 1.0e-6
        y_scaled, self.y_mean, self.y_std = _standardize(targets)
        x_scaled = (features - self.x_mean) / self.x_std
        self.hidden_w = rng.normal(0.0, 0.65, size=(features.shape[1], self.hidden_dim))
        self.hidden_b = rng.normal(0.0, 0.20, size=(self.hidden_dim,))
        hidden = np.tanh(x_scaled @ self.hidden_w + self.hidden_b)
        ridge = 1.0e-4
        self.beta = np.linalg.solve(hidden.T @ hidden + ridge * np.eye(self.hidden_dim), hidden.T @ y_scaled)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.cache_path,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            hidden_w=self.hidden_w,
            hidden_b=self.hidden_b,
            beta=self.beta,
        )

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
        return self._feature_matrix(
            factor,
            coupled,
            x_norm,
            q_cluster,
            np.full_like(factor, float(viscosity_pa_s)),
            np.full_like(factor, float(e_prime_pa)),
            np.full_like(factor, float(height_m)),
            np.full_like(factor, float(t_seconds)),
        )

    @staticmethod
    def _feature_matrix(
        factor: np.ndarray,
        coupled: np.ndarray,
        x_norm: np.ndarray,
        q_cluster: np.ndarray,
        viscosity: np.ndarray,
        e_prime: np.ndarray,
        height: np.ndarray,
        t_seconds: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [
                factor,
                coupled,
                x_norm,
                np.log(np.maximum(q_cluster, 1.0e-12)),
                np.log(np.maximum(viscosity, 1.0e-12)),
                np.log(np.maximum(e_prime, 1.0)),
                np.log(np.maximum(height, 1.0e-9)),
                np.log(np.maximum(t_seconds, 1.0e-9)),
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
    raise ValueError(f"Unsupported length forward model: {name}")


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1.0e-6
    return (values - mean) / std, mean, std
