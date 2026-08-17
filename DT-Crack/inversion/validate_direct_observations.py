from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion import controls_for_step, load_frac_monitor_text, load_stage_pressure_schedule, pressure_for_step
from data_fusion.pressure_schedule_adapter import load_pressure_model_config
from inversion import (
    PhysicalEnKFConfig,
    denkf_update,
    enkf_update,
    physical_values,
    pkn_with_carter_leakoff,
    state_record,
)
from inversion.knowledge_guided_enkf import (
    KnowledgeGuidedPriorConfig,
    apply_knowledge_guided_observation_std,
    build_knowledge_guided_prior,
    project_knowledge_guided_update,
)
from forward_models.pyfrac_surrogate import PyFracResidualSurrogate
from inversion.pressure_only_enkf import PressureOnlyConfig, run_pressure_only_correction
from data_fusion.observation_quality import validate_cluster_controls


def configure_font() -> None:
    for path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def normalize_positive(values: np.ndarray) -> np.ndarray:
    clean = np.clip(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0), 0.0, None)
    total = float(clean.sum())
    return clean / total if total > 1e-12 else np.full(len(clean), 1.0 / max(len(clean), 1))


def observed_cluster_shares(step_controls: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    liquid = normalize_positive(step_controls["cumulative_liquid_volume_m3"].to_numpy(dtype=float))
    sand = normalize_positive(step_controls["cumulative_sand_mass_t"].to_numpy(dtype=float))
    return liquid, sand


def fiber_liquid_allocation(
    step_controls: pd.DataFrame,
    previous: np.ndarray | None = None,
    smoothing: float = 0.20,
) -> np.ndarray:
    """Return the measured incremental liquid allocation for PKN.

    ``allocation_weight`` is derived from the fiber cluster liquid increments,
    not from a six-cluster equal split.  Only a light exponential smoothing is
    applied to prevent one noisy sampling interval from moving all flow to one
    cluster.  The final vector is always non-negative and sums to one.
    """

    if "allocation_weight" in step_controls:
        raw = normalize_positive(step_controls["allocation_weight"].to_numpy(dtype=float))
    else:
        raw = normalize_positive(step_controls["liquid_volume_m3"].to_numpy(dtype=float))
    if previous is None or len(previous) != len(raw):
        return raw
    previous = normalize_positive(previous)
    memory_weight = float(np.clip(smoothing, 0.0, 0.95))
    return normalize_positive(memory_weight * previous + (1.0 - memory_weight) * raw)


def predicted_observation(
    forward: dict,
    n_clusters: int,
    sand_transport_factors: np.ndarray | None = None,
    pressure_bias_mpa: float = 0.0,
    cumulative_liquid: np.ndarray | None = None,
    cumulative_sand: np.ndarray | None = None,
    sand_transport_exponent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if bool(forward.get("measured_allocation_used", False)) and cumulative_liquid is None:
        # The liquid allocation has already been observed and supplied as a PKN
        # boundary condition.  Do not ask EnKF to infer this same allocation
        # from a second latent state.
        liquid_share = normalize_positive(np.asarray(forward["cluster_allocation"], dtype=float))
    else:
        liquid_values = (
            np.asarray(forward["q_effective_m3_s"], dtype=float)
            if cumulative_liquid is None
            else np.asarray(cumulative_liquid, dtype=float)
        )
        liquid_share = normalize_positive(liquid_values)
    transport = np.asarray(forward["q_effective_m3_s"], dtype=float) * np.maximum(
        np.asarray(forward["max_aperture_mm"], dtype=float), 1e-6
    )
    if sand_transport_factors is not None:
        transport *= np.asarray(sand_transport_factors, dtype=float)
    # A sublinear capacity exponent represents mixing/settling and prevents
    # q*aperture from over-concentrating all proppant in the highest-rate
    # cluster. The mean-preserving form keeps the total transport scale intact.
    exponent = max(float(sand_transport_exponent), 1.0e-6)
    transport_mean = max(float(np.mean(transport)), 1.0e-12)
    transport = transport * np.power(np.maximum(transport / transport_mean, 1.0e-12), exponent - 1.0)
    sand_values = transport if cumulative_sand is None else np.asarray(cumulative_sand, dtype=float)
    sand_share = normalize_positive(sand_values)
    # The last share in each group is implied by the sum-to-one constraint.
    observation = np.r_[
        liquid_share[: n_clusters - 1],
        sand_share[: n_clusters - 1],
        float(forward["bottomhole_pressure_mpa"]) + float(pressure_bias_mpa),
    ]
    return observation, np.r_[liquid_share, sand_share]


def propagate_cumulative_memory(
    forward: dict,
    previous_liquid: np.ndarray,
    previous_sand: np.ndarray,
    dt_seconds: float,
    sand_transport_factors: np.ndarray,
    sand_transport_exponent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the model allocation between two observation timestamps.

    The fiber fields are cumulative shares. Using the current instantaneous
    q/aperture as the observation directly introduces a structural mismatch.
    This lightweight state propagation retains cumulative liquid allocation
    and cumulative proppant-transport capacity while keeping the EnKF state
    itself limited to physical PKN parameters.
    """

    dt = max(float(dt_seconds), 1.0e-6)
    q_effective = np.maximum(np.asarray(forward["q_effective_m3_s"], dtype=float), 0.0)
    aperture_m = np.maximum(np.asarray(forward["max_aperture_mm"], dtype=float), 0.0) * 1.0e-3
    transport = np.maximum(np.asarray(sand_transport_factors, dtype=float), 0.0)
    liquid = np.maximum(np.asarray(previous_liquid, dtype=float), 0.0) + q_effective * dt
    capacity = q_effective * aperture_m * transport
    exponent = max(float(sand_transport_exponent), 1.0e-6)
    capacity_mean = max(float(np.mean(capacity)), 1.0e-12)
    capacity = capacity * np.power(np.maximum(capacity / capacity_mean, 1.0e-12), exponent - 1.0)
    sand = np.maximum(np.asarray(previous_sand, dtype=float), 0.0) + capacity * dt
    return liquid, sand


def adaptive_observation_std(
    args: argparse.Namespace,
    observed_liquid: np.ndarray,
    observed_sand: np.ndarray,
    observed_bhp: float,
) -> np.ndarray:
    """Set observation uncertainty without hiding the cluster signal.

    Small shares have higher relative counting/rounding uncertainty, while
    pressure uncertainty includes a percentage component for unmodelled
    friction. The lower bound prevents a very large cluster from dominating
    the whole update.
    """

    if not args.adaptive_observation_noise:
        return np.r_[
            np.full(2 * (len(observed_liquid) - 1), args.share_noise),
            args.bottomhole_pressure_noise_mpa,
        ]
    liquid_scale = np.clip(0.72 + 0.85 * np.sqrt(np.maximum(observed_liquid[:-1], 0.0) + 0.01), 0.72, 1.15)
    sand_scale = np.clip(0.72 + 0.85 * np.sqrt(np.maximum(observed_sand[:-1], 0.0) + 0.01), 0.72, 1.15)
    pressure_std = max(float(args.bottomhole_pressure_noise_mpa), 0.035 * abs(float(observed_bhp)))
    return np.r_[args.share_noise * liquid_scale, args.share_noise * sand_scale, pressure_std]


def adaptive_covariance_inflation(
    args: argparse.Namespace,
    innovation: np.ndarray,
    observation_std: np.ndarray,
) -> float:
    if not args.adaptive_inflation:
        return float(args.covariance_inflation)
    normalized = np.asarray(innovation, dtype=float) / np.maximum(np.asarray(observation_std, dtype=float), 1.0e-9)
    normalized = np.clip(normalized, -8.0, 8.0)
    normalized_innovation = float(np.mean(normalized**2))
    excess = np.clip(normalized_innovation - 1.0, 0.0, 4.0)
    return float(np.clip(args.covariance_inflation * (1.0 + 0.012 * excess), 1.0, 1.06))


def apply_residual_surrogate(
    forward: dict,
    state: np.ndarray,
    cfg: PhysicalEnKFConfig,
    n_clusters: int,
    t_seconds: float,
    cluster_spacing_m: float,
    surrogate: PyFracResidualSurrogate,
) -> dict:
    """Apply a PyFrac-minus-PKN residual after the PKN parameter update.

    The surrogate is deliberately downstream of ``pkn_with_carter_leakoff``:
    EnKF changes physical parameters, PKN is recomputed, and only then is the
    learned offline correction added.  No fracture length is treated as a
    filter state or copied from an observation.
    """

    values = physical_values(state, cfg, n_clusters)
    q_effective = np.asarray(forward["q_effective_m3_s"], dtype=float)
    allocation = normalize_positive(np.asarray(forward["cluster_allocation"], dtype=float))
    features = pd.DataFrame(
        {
            "e_prime_gpa": np.full(n_clusters, float(values["eprime_pa"]) / 1.0e9),
            "leakoff_m_sqrt_s": np.full(n_clusters, float(values["leakoff_m_sqrt_s"])),
            "viscosity_pa_s": np.full(n_clusters, float(values["viscosity_pa_s"])),
            "min_stress_mpa": np.full(n_clusters, float(values["min_horizontal_stress_mpa"])),
            "fracture_toughness_pa_sqrt_m": np.full(
                n_clusters, float(values["fracture_toughness_pa_sqrt_m"])
            ),
            "height_m": np.full(n_clusters, float(cfg.height_m)),
            "q_m3_s": q_effective,
            "time_s": np.full(n_clusters, float(t_seconds)),
            "cluster_spacing_m": np.full(n_clusters, float(cluster_spacing_m)),
            "allocation_weight": allocation,
            "cluster_id": np.arange(1, n_clusters + 1, dtype=float),
            "q_total_m3_s": np.full(n_clusters, float(q_effective.sum())),
            "cumulative_injection_m3": np.full(
                n_clusters, float(q_effective.sum()) * max(float(t_seconds), 1.0)
            ),
            "q_ramp_m3_s2": np.zeros(n_clusters, dtype=float),
            "leakoff_volume_fraction": np.full(
                n_clusters,
                min(
                    0.95,
                    float(values["leakoff_m_sqrt_s"])
                    * np.sqrt(max(float(t_seconds), 1.0))
                    * 12.0
                    / max(float(q_effective.sum()), 1.0e-6),
                ),
            ),
            "scenario_type_code": np.zeros(n_clusters, dtype=float),
        }
    )
    # The six-cluster liquid allocation is observed input to the forward
    # model. Expose the whole allocation vector to the residual surrogate so
    # it can learn cluster competition without introducing a free EnKF factor.
    for index in range(n_clusters):
        features[f"allocation_w{index + 1}"] = np.full(n_clusters, allocation[index], dtype=float)
    corrected = surrogate.predict_online_state(
        features,
        np.asarray(forward["half_length_m"], dtype=float),
        np.asarray(forward["max_aperture_mm"], dtype=float),
        np.full(n_clusters, float(forward["net_pressure_mpa"])),
    )
    result = dict(forward)
    result["half_length_m"] = corrected["half_length_m"].to_numpy(dtype=float)
    result["max_aperture_mm"] = corrected["max_aperture_mm"].to_numpy(dtype=float)
    result["net_pressure_mpa"] = float(corrected["net_pressure_mpa"].mean())
    result["bottomhole_pressure_mpa"] = float(values["min_horizontal_stress_mpa"]) + float(
        result["net_pressure_mpa"]
    )
    result["pyfrac_residual_surrogate_applied"] = True
    return result


def clip_augmented_state(
    state: np.ndarray,
    n_clusters: int,
    log_cluster_factor_state: bool = False,
) -> np.ndarray:
    """Clip the five physical EnKF parameters.

    The legacy arguments remain in the signature so old command lines do not
    fail, but no per-cluster factor is clipped or updated in the new state.
    """
    out = np.asarray(state, dtype=float).copy()
    out[..., 0] = np.clip(out[..., 0], np.log(0.45), np.log(2.2))
    out[..., 1] = np.clip(out[..., 1], np.log(0.1), np.log(8.0))
    out[..., 2] = np.clip(out[..., 2], np.log(0.2), np.log(5.0))
    out[..., 3] = np.clip(out[..., 3], 35.0, 90.0)
    if out.shape[-1] >= 5:
        out[..., 4] = np.clip(out[..., 4], np.log(0.25), np.log(4.0))
    return out


def total_variation(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(predicted) - np.asarray(observed)).sum())


def build_physical_localization(n_clusters: int) -> np.ndarray:
    """Map each observation primarily to parameters with a physical pathway."""

    state_size = 5
    obs_size = 2 * (n_clusters - 1) + 1
    pressure_col = obs_size - 1
    weights = np.zeros((state_size, obs_size), dtype=float)
    # Fiber allocation is now an observed PKN boundary condition.  Share
    # residuals are retained for validation, but they must not update a second
    # hidden allocation state.  Only the pressure channel updates global PKN
    # parameters in the reduced EnKF state.
    weights[:5, pressure_col] = [0.75, 0.25, 0.65, 1.00, 0.35]
    return weights


def batch_calibrate_state(
    args: argparse.Namespace,
    source_steps: np.ndarray,
    calibration_count: int,
    controls: pd.DataFrame,
    pressure: pd.DataFrame,
    cfg: PhysicalEnKFConfig,
    n_clusters: int,
    prior_mean: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Fit a static physical prior using only the calibration prefix.

    This is a MAP-style initialization, not a replacement for online EnKF.
    Static physical parameters are estimated from all available calibration
    observations, then EnKF continues to track time-varying cluster factors.
    A small Gaussian prior penalty keeps the inverse problem from selecting a
    numerically convenient but physically implausible equivalent solution.  The
    inverse state is limited to five global PKN parameters; fiber liquid
    allocation is supplied directly to the forward operator.
    """

    if not args.batch_calibrate:
        return prior_mean.copy(), {"enabled": False}
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return prior_mean.copy(), {"enabled": False, "reason": "scipy_unavailable"}

    calibration_steps = source_steps[:calibration_count]

    def residual(state: np.ndarray) -> np.ndarray:
        state = clip_augmented_state(state, n_clusters, args.log_cluster_factor_state)
        residuals: list[float] = []
        for source_step in calibration_steps:
            step_controls = controls_for_step(controls, int(source_step))
            t_seconds = max(float(source_step), 1.0)
            total_cumulative = float(step_controls["cumulative_liquid_volume_m3"].sum())
            total_rate = max(total_cumulative / t_seconds, 1e-8)
            current_total_rate = max(float(step_controls["flow_rate_m3_min"].sum()) / 60.0, 1e-8)
            cumulative_allocation, _ = observed_cluster_shares(step_controls)
            q_base = total_rate * cumulative_allocation
            q_current_allocation = fiber_liquid_allocation(step_controls, smoothing=args.fiber_allocation_smoothing)
            q_current = current_total_rate * q_current_allocation
            observed_liquid, observed_sand = observed_cluster_shares(step_controls)
            observed_bhp = float(pressure_for_step(pressure, int(source_step))["bottomhole_pressure_mpa"])
            forward = pkn_with_carter_leakoff(
                state,
                q_base,
                t_seconds,
                cfg,
                q_current,
                cluster_allocation=cumulative_allocation,
                cluster_current_allocation=q_current_allocation,
            )
            predicted, _ = predicted_observation(
                forward,
                n_clusters,
            )
            observed = np.r_[observed_liquid[: n_clusters - 1], observed_sand[: n_clusters - 1], observed_bhp]
            std = adaptive_observation_std(args, observed_liquid, observed_sand, observed_bhp)
            residuals.extend(((predicted - observed) / np.maximum(std, 1.0e-9)).tolist())

        # Weak prior regularization makes the parameters identifiable without
        # forcing them to remain at the engineering starting values.
        regularization = np.r_[
            state[0] / 0.30,
            state[1] / 0.70,
            state[2] / 0.45,
            (state[3] - args.base_min_stress_mpa) / 8.0,
            state[4] / 0.35,
        ]
        residuals.extend((0.12 * regularization).tolist())
        return np.asarray(residuals, dtype=float)

    lower = np.full_like(prior_mean, -np.inf, dtype=float)
    upper = np.full_like(prior_mean, np.inf, dtype=float)
    lower[:4] = [np.log(0.45), np.log(0.1), np.log(0.2), 35.0]
    upper[:5] = [np.log(2.2), np.log(8.0), np.log(5.0), 90.0, np.log(4.0)]
    start = time.perf_counter()
    initial_cost = float(0.5 * np.sum(residual(prior_mean) ** 2))
    result = least_squares(
        residual,
        np.clip(prior_mean, lower, upper),
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max(int(args.batch_max_nfev), 20),
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
    )
    center = clip_augmented_state(result.x, n_clusters, args.log_cluster_factor_state)
    return center, {
        "enabled": True,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "initial_cost": initial_cost,
        "final_cost": float(result.cost),
        "runtime_seconds": float(time.perf_counter() - start),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct-observation PKN-EnKF calibration and held-out validation.")
    parser.add_argument("--frac-monitor-text", required=False)
    parser.add_argument("--construction-pressure-xls", required=True)
    parser.add_argument("--pressure-calibration-config", default=None)
    parser.add_argument("--observation-mode", choices=["pressure_only", "pressure_plus_cluster"], default="pressure_plus_cluster")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--calibration-ratio", type=float, default=0.70)
    # 300 members materially reduce seed sensitivity while remaining far below
    # the 15-second online update budget on the current six-cluster problem.
    parser.add_argument("--ensemble-size", type=int, default=400)
    parser.add_argument("--share-noise", type=float, default=0.015)
    parser.add_argument("--bottomhole-pressure-noise-mpa", type=float, default=3.5)
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--base-min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--pressure-proxy-scale", type=float, default=30.0)
    parser.add_argument("--max-leakoff-fraction", type=float, default=0.50)
    parser.add_argument("--hydraulic-coupling-mode", choices=["legacy", "coupled"], default="coupled")
    parser.add_argument("--conductance-exponent", type=float, default=0.45)
    parser.add_argument("--measured-depth-m", type=float, default=5218.0)
    parser.add_argument("--vertical-depth-m", type=float, default=3196.94)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-mode", choices=["frozen", "online"], default="online")
    parser.add_argument("--filter-method", choices=["stochastic", "denkf"], default="stochastic")
    parser.add_argument("--assimilation-iterations", type=int, default=1)
    parser.add_argument("--covariance-inflation", type=float, default=1.012)
    parser.add_argument("--robust-innovation-threshold", type=float, default=4.0)
    parser.add_argument("--cluster-process-std", type=float, default=0.06)
    parser.add_argument("--sand-process-std", type=float, default=0.06)
    parser.add_argument("--pressure-bias-process-std", type=float, default=1.5)
    parser.add_argument("--dynamic-pressure-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-localization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Localize EnKF to pressure for physical parameters; share observations are measured allocation inputs.",
    )
    parser.add_argument("--physics-profile", choices=["legacy", "enhanced"], default="legacy")
    parser.add_argument(
        "--observation-memory-mode",
        choices=["instantaneous", "cumulative"],
        default="instantaneous",
        help="Use current allocation or history-integrated allocation for cumulative fiber shares.",
    )
    parser.add_argument(
        "--sand-observation-memory-mode",
        choices=["instantaneous", "cumulative"],
        default="cumulative",
        help="Use a cumulative transport memory for sand shares while liquid shares remain measured inputs.",
    )
    parser.add_argument(
        "--sand-transport-exponent",
        type=float,
        default=0.75,
        help="Mean-preserving sublinear exponent for reduced-order proppant transport capacity.",
    )
    parser.add_argument(
        "--fiber-allocation-smoothing",
        type=float,
        default=0.20,
        help="Previous-step weight for light smoothing of fiber incremental liquid allocation.",
    )
    parser.add_argument("--adaptive-observation-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-inflation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stress-shadow-feedback", type=float, default=0.0)
    parser.add_argument("--pressure-leakoff-exponent", type=float, default=0.0)
    parser.add_argument("--pressure-leakoff-reference-mpa", type=float, default=15.0)
    parser.add_argument("--log-cluster-factor-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-calibrate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-max-nfev", type=int, default=80)
    parser.add_argument(
        "--knowledge-guided-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Backward-compatible alias for --knowledge-guided-mode uncertainty_only.",
    )
    parser.add_argument(
        "--knowledge-guided-mode",
        choices=["off", "uncertainty_only", "soft_prior", "soft_correlated"],
        default="off",
        help=(
            "Knowledge-graph bridge mode. soft_prior adds a small signed prior hypothesis; "
            "soft_correlated additionally uses covariance, observation confidence and update bounds."
        ),
    )
    parser.add_argument(
        "--knowledge-graph-rules",
        default=None,
        help="Optional fused knowledge-graph rule JSON. Defaults to FSL-Expert/rule_fusion/rule_fusion/fused_sand_plug_rules.json.",
    )
    parser.add_argument(
        "--knowledge-guided-strength",
        type=float,
        default=0.35,
        help="0..1 strength of the KG prior bridge.",
    )
    parser.add_argument("--knowledge-guided-mean-shift-scale", type=float, default=1.0)
    parser.add_argument("--knowledge-guided-covariance-scale", type=float, default=1.0)
    parser.add_argument("--knowledge-guided-observation-noise-scale", type=float, default=0.25)
    parser.add_argument("--knowledge-guided-max-update-scale", type=float, default=2.0)
    parser.add_argument(
        "--surrogate-path",
        default=None,
        help="Optional PyFrac-minus-PKN residual surrogate; it is used only when its held-out quality gate passes.",
    )
    parser.add_argument(
        "--allow-unapproved-surrogate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly allow a surrogate that failed the online quality gate for development experiments.",
    )
    parser.add_argument("--surrogate-min-test-r2", type=float, default=0.80)
    parser.add_argument("--cluster-spacing-m", type=float, default=25.0)
    parser.add_argument("--run-dir", default=str(DT_ROOT.parent / "outputs" / "dt" / "direct_observation_enkf"))
    return parser


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if args.physics_profile == "enhanced":
        # The enhanced profile uses a calibrated physical prior and leaves the
        # online EnKF update conservative. This is more stable than merely
        # increasing ensemble size or repeatedly assimilating the same sample.
        args.batch_calibrate = True
        if abs(float(args.max_leakoff_fraction) - 0.50) < 1.0e-12:
            # 50% was the former conservative cap. The enhanced profile uses
            # 85% as a soft engineering prior, while mass conservation still
            # prevents leakoff from exceeding cumulative injection.
            args.max_leakoff_fraction = 0.85
        args.stress_shadow_feedback = max(float(args.stress_shadow_feedback), 0.75)
        args.pressure_leakoff_exponent = max(float(args.pressure_leakoff_exponent), 0.12)

    rng = np.random.default_rng(args.seed)
    pressure_cfg = load_pressure_model_config(args.pressure_calibration_config)
    if args.pressure_calibration_config is None:
        pressure_cfg = type(pressure_cfg)(**{**asdict(pressure_cfg), "min_horizontal_stress_mpa": args.base_min_stress_mpa})
    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls, pressure_cfg, args.measured_depth_m, args.vertical_depth_m
    )
    quality_meta: dict[str, object] = {}
    if args.observation_mode == "pressure_only":
        # A single synthetic boundary row represents the total stream only;
        # it is never reported as a measured six-cluster observation.
        increments = pressure["cumulative_liquid_m3"].diff().fillna(pressure["cumulative_liquid_m3"].iloc[0]).clip(lower=0.0)
        controls = pd.DataFrame({
            "step": pressure["step"].astype(int),
            "time": pd.to_datetime(pressure["time_s"], unit="s"),
            "fracture_id": 1,
            "cluster_id": 1,
            "allocation_weight": 1.0,
            "flow_rate_m3_min": pressure["flow_rate_m3_min"],
            "liquid_volume_m3": increments,
            "sand_mass_t": increments * pressure["sand_ratio_percent"] / 100.0,
            "cumulative_liquid_volume_m3": pressure["cumulative_liquid_m3"],
            "cumulative_sand_mass_t": pressure["cumulative_liquid_m3"] * pressure["sand_ratio_percent"] / 100.0,
            "balance_degree": np.nan,
            "cumulative_balance_degree": np.nan,
        })
        monitor_meta = {"source": None, "cluster_count": 0, "time_steps": int(len(controls)), "observation_mode": "pressure_only"}
        n_clusters = 1
        quality_meta = {"status": "not_available", "reason": "DAS not connected; cluster observations are not inferred"}
    else:
        if not args.frac_monitor_text:
            raise ValueError("--frac-monitor-text is required for pressure_plus_cluster mode")
        monitor = load_frac_monitor_text(args.frac_monitor_text, default_step_seconds=1.0)
        controls, quality = validate_cluster_controls(
            monitor.controls,
            expected_clusters=6,
            pressure_times=pressure["step"].to_numpy(dtype=float),
        )
        quality_meta = quality.to_dict()
        controls = controls[controls["qc_valid"]].drop(columns=["qc_valid", "qc_reason"])
        monitor_meta = dict(monitor.meta)
        monitor_meta["observation_quality"] = quality_meta
        n_clusters = int(controls["cluster_id"].nunique())
        if n_clusters != 6 or controls.empty:
            raise ValueError("DAS/FracMonitor data has no valid six-cluster observation steps")
    available_steps = np.asarray(sorted(controls["step"].unique()), dtype=int)
    count = min(max(args.max_steps, 2), len(available_steps))
    indices = np.linspace(0, len(available_steps) - 1, count).round().astype(int)
    source_steps = np.unique(available_steps[indices])
    calibration_count = max(1, min(len(source_steps) - 1, int(round(len(source_steps) * args.calibration_ratio))))

    cfg = PhysicalEnKFConfig(
        base_eprime_pa=args.base_eprime_pa,
        base_leakoff_m_sqrt_s=args.base_leakoff_m_sqrt_s,
        base_viscosity_pa_s=args.base_viscosity_pa_s,
        base_min_stress_mpa=args.base_min_stress_mpa,
        base_fracture_toughness_pa_sqrt_m=5.0e5,
        height_m=args.height_m,
        pressure_proxy_scale=args.pressure_proxy_scale,
        max_leakoff_fraction=args.max_leakoff_fraction,
        hydraulic_coupling_mode=args.hydraulic_coupling_mode,
        conductance_exponent=args.conductance_exponent,
        stress_shadow_feedback=args.stress_shadow_feedback,
        pressure_leakoff_exponent=args.pressure_leakoff_exponent,
        pressure_leakoff_reference_mpa=args.pressure_leakoff_reference_mpa,
        log_cluster_factor_state=args.log_cluster_factor_state,
    )
    surrogate = None
    surrogate_gate: dict[str, object] = {
        "requested": bool(args.surrogate_path),
        "path": str(Path(args.surrogate_path).resolve()) if args.surrogate_path else None,
        "online_enabled": False,
    }
    if args.surrogate_path:
        surrogate = PyFracResidualSurrogate.load(args.surrogate_path)
        surrogate_gate.update(surrogate.online_readiness(min_test_r2=args.surrogate_min_test_r2))
        if not surrogate_gate.get("ready", False) and not args.allow_unapproved_surrogate:
            surrogate = None
            surrogate_gate["reason"] = (
                str(surrogate_gate.get("reason", "quality gate failed"))
                + "; fallback to PKN because --allow-unapproved-surrogate was not set"
            )
        else:
            surrogate_gate["online_enabled"] = True
    prior_mean = np.r_[
        0.0,
        0.0,
        0.0,
        args.base_min_stress_mpa,
        0.0,
    ]
    spread = np.r_[
        0.18,
        0.45,
        0.30,
        5.0,
        0.25,
    ]
    process = np.r_[
        0.012,
        0.020,
        0.015,
        0.20,
        0.012,
    ]
    calibration_center, batch_calibration = batch_calibrate_state(
        args,
        source_steps,
        calibration_count,
        controls,
        pressure,
        cfg,
        n_clusters,
        prior_mean,
    )
    initial_spread = spread * (0.45 if batch_calibration.get("enabled") else 1.0)
    kg_mode = str(args.knowledge_guided_mode)
    if kg_mode == "off" and bool(args.knowledge_guided_prior):
        # Preserve the command line used by the original uncertainty-only
        # experiment while allowing the new modes to be selected explicitly.
        kg_mode = "uncertainty_only"
    kg_distribution = build_knowledge_guided_prior(
        controls,
        pressure,
        source_steps[:calibration_count],
        calibration_center,
        initial_spread,
        process,
        KnowledgeGuidedPriorConfig(
            enabled=kg_mode != "off",
            strength=float(args.knowledge_guided_strength),
            rules_path=args.knowledge_graph_rules,
            mode=kg_mode,
            mean_shift_scale=float(args.knowledge_guided_mean_shift_scale),
            covariance_scale=float(args.knowledge_guided_covariance_scale),
            observation_noise_scale=float(args.knowledge_guided_observation_noise_scale),
            max_update_scale=float(args.knowledge_guided_max_update_scale),
        ),
    )
    kg_spread = np.asarray(kg_distribution["spread"], dtype=float)
    kg_process_covariance = np.asarray(kg_distribution["process_covariance"], dtype=float)
    kg_prior = dict(kg_distribution["meta"])
    kg_prior_mean = np.asarray(kg_distribution["mean"], dtype=float)
    kg_prior_covariance = np.asarray(kg_distribution["covariance"], dtype=float)
    ensemble = rng.multivariate_normal(
        kg_prior_mean,
        kg_prior_covariance,
        size=args.ensemble_size,
        check_valid="ignore",
    )
    ensemble = clip_augmented_state(
        ensemble,
        n_clusters,
        args.log_cluster_factor_state,
    )
    rows: list[dict] = []
    cluster_rows: list[dict] = []
    localization = build_physical_localization(n_clusters)
    cumulative_liquid_memory = np.zeros((args.ensemble_size, n_clusters), dtype=float)
    cumulative_sand_memory = np.zeros((args.ensemble_size, n_clusters), dtype=float)
    previous_time_seconds: float | None = None
    previous_fiber_allocation: np.ndarray | None = None

    for sequence_index, source_step in enumerate(source_steps):
        step_started = time.perf_counter()
        phase = "calibration" if sequence_index < calibration_count else "validation"
        step_controls = controls_for_step(controls, int(source_step))
        t_seconds = max(float(source_step), 1.0)
        total_cumulative = float(step_controls["cumulative_liquid_volume_m3"].sum())
        total_rate = max(total_cumulative / t_seconds, 1e-8)
        current_total_rate = max(float(step_controls["flow_rate_m3_min"].sum()) / 60.0, 1e-8)
        observed_liquid, observed_sand = observed_cluster_shares(step_controls)
        # Cumulative fiber liquid shares control historical PKN growth.  The
        # incremental fiber allocation controls current-rate/aperture physics;
        # both are measured inputs, not EnKF latent cluster factors.
        q_base_allocation = observed_liquid
        fiber_allocation = fiber_liquid_allocation(
            step_controls,
            previous=previous_fiber_allocation,
            smoothing=args.fiber_allocation_smoothing,
        )
        previous_fiber_allocation = fiber_allocation
        q_base = total_rate * q_base_allocation
        q_current = current_total_rate * fiber_allocation
        observed_bhp = float(pressure_for_step(pressure, int(source_step))["bottomhole_pressure_mpa"])
        dt_seconds = t_seconds if previous_time_seconds is None else max(t_seconds - previous_time_seconds, 1.0)

        should_update = phase == "calibration" or args.validation_mode == "online"
        if should_update:
            process_noise = rng.multivariate_normal(
                np.zeros(len(prior_mean), dtype=float),
                kg_process_covariance,
                size=ensemble.shape[0],
                check_valid="ignore",
            )
            ensemble = clip_augmented_state(
                ensemble + process_noise,
                n_clusters,
                args.log_cluster_factor_state,
            )
        observed_obs = np.r_[observed_liquid[: n_clusters - 1], observed_sand[: n_clusters - 1], observed_bhp]
        obs_std = adaptive_observation_std(args, observed_liquid, observed_sand, observed_bhp)
        obs_std = apply_knowledge_guided_observation_std(obs_std, kg_prior, n_clusters)

        def evaluate_ensemble(
            current_ensemble: np.ndarray,
            previous_liquid: np.ndarray,
            previous_sand: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            observations = []
            next_liquid = []
            next_sand = []
            for member, member_liquid, member_sand in zip(current_ensemble, previous_liquid, previous_sand):
                item = pkn_with_carter_leakoff(
                    member,
                    q_base,
                    t_seconds,
                    cfg,
                    q_current,
                    cluster_allocation=q_base_allocation,
                    cluster_current_allocation=fiber_allocation,
                )
                if surrogate is not None:
                    item = apply_residual_surrogate(
                        item,
                        member,
                        cfg,
                        n_clusters,
                        t_seconds,
                        args.cluster_spacing_m,
                        surrogate,
                    )
                sand_factors = np.ones(n_clusters, dtype=float)
                liquid_memory, sand_memory = propagate_cumulative_memory(
                    item,
                    member_liquid,
                    member_sand,
                    dt_seconds,
                    sand_factors,
                    sand_transport_exponent=args.sand_transport_exponent,
                )
                observation, _ = predicted_observation(
                    item,
                    n_clusters,
                    cumulative_liquid=liquid_memory if args.observation_memory_mode == "cumulative" else None,
                    cumulative_sand=sand_memory if args.sand_observation_memory_mode == "cumulative" else None,
                    sand_transport_exponent=args.sand_transport_exponent,
                )
                observations.append(observation)
                next_liquid.append(liquid_memory)
                next_sand.append(sand_memory)
            return np.asarray(observations), np.asarray(next_liquid), np.asarray(next_sand)

        predicted_obs_rows, forecast_liquid, forecast_sand = evaluate_ensemble(
            ensemble, cumulative_liquid_memory, cumulative_sand_memory
        )

        prior_state = ensemble.mean(axis=0)
        prior = pkn_with_carter_leakoff(
            prior_state,
            q_base,
            t_seconds,
            cfg,
            q_current,
            cluster_allocation=q_base_allocation,
            cluster_current_allocation=fiber_allocation,
        )
        if surrogate is not None:
            prior = apply_residual_surrogate(
                prior,
                prior_state,
                cfg,
                n_clusters,
                t_seconds,
                args.cluster_spacing_m,
                surrogate,
            )
        if args.observation_memory_mode == "cumulative":
            prior_shares = np.r_[
                normalize_positive(forecast_liquid.mean(axis=0)),
                normalize_positive(forecast_sand.mean(axis=0)),
            ]
        else:
            _, prior_shares = predicted_observation(
                prior,
                n_clusters,
                sand_transport_exponent=args.sand_transport_exponent,
            )
        if args.observation_memory_mode == "instantaneous":
            prior_shares[:n_clusters] = observed_liquid
        if args.sand_observation_memory_mode == "cumulative":
            prior_shares[n_clusters:] = normalize_positive(forecast_sand.mean(axis=0))
        gain_mean = 0.0
        inflation_values = []
        if should_update:
            assimilation_reference = ensemble.copy()
            gain_values = []
            iteration_count = max(int(args.assimilation_iterations), 1)
            for iteration in range(iteration_count):
                if iteration:
                    predicted_obs_rows, _, _ = evaluate_ensemble(
                        ensemble, cumulative_liquid_memory, cumulative_sand_memory
                    )
                innovation = observed_obs - predicted_obs_rows.mean(axis=0)
                robust_scale = np.ones_like(obs_std)
                # Cluster-share imbalance is the signal that identifies intake
                # parameters and must not be suppressed as an outlier. Robust
                # down-weighting is reserved for the pressure channel, where
                # gauge spikes and unmodelled near-wellbore losses are plausible.
                robust_scale[-1] = max(
                    1.0,
                    abs(float(innovation[-1]))
                    / max(args.robust_innovation_threshold * float(obs_std[-1]), 1.0e-9),
                )
                # Tempering prevents repeated nonlinear updates from counting the
                # same observation more than once in likelihood terms.
                iteration_std = obs_std * np.sqrt(iteration_count) * robust_scale
                inflation = adaptive_covariance_inflation(args, innovation, iteration_std)
                if args.filter_method == "denkf":
                    ensemble, gain = denkf_update(
                        ensemble,
                        predicted_obs_rows,
                        observed_obs,
                        iteration_std,
                        covariance_inflation=inflation,
                        localization=localization,
                    )
                else:
                    ensemble, gain = enkf_update(
                        ensemble,
                        predicted_obs_rows,
                        observed_obs,
                        iteration_std,
                        rng,
                        localization=localization if args.use_localization else None,
                    )
                ensemble = project_knowledge_guided_update(
                    ensemble,
                    assimilation_reference,
                    kg_spread,
                    kg_prior,
                )
                ensemble = clip_augmented_state(ensemble, n_clusters, args.log_cluster_factor_state)
                gain_values.append(float(np.mean(np.abs(gain))))
                inflation_values.append(inflation)
            gain_mean = float(np.mean(gain_values))
        posterior_state = ensemble.mean(axis=0)
        posterior = pkn_with_carter_leakoff(
            posterior_state,
            q_base,
            t_seconds,
            cfg,
            q_current,
            cluster_allocation=q_base_allocation,
            cluster_current_allocation=fiber_allocation,
        )
        if surrogate is not None:
            posterior = apply_residual_surrogate(
                posterior,
                posterior_state,
                cfg,
                n_clusters,
                t_seconds,
                args.cluster_spacing_m,
                surrogate,
            )
        _, posterior_liquid_memory, posterior_sand_memory = evaluate_ensemble(
            ensemble, cumulative_liquid_memory, cumulative_sand_memory
        )
        if args.observation_memory_mode == "cumulative":
            posterior_shares = np.r_[
                normalize_positive(posterior_liquid_memory.mean(axis=0)),
                normalize_positive(posterior_sand_memory.mean(axis=0)),
            ]
        else:
            _, posterior_shares = predicted_observation(
                posterior,
                n_clusters,
                sand_transport_exponent=args.sand_transport_exponent,
            )
        if args.observation_memory_mode == "instantaneous":
            posterior_shares[:n_clusters] = observed_liquid
        if args.sand_observation_memory_mode == "cumulative":
            posterior_shares[n_clusters:] = normalize_positive(posterior_sand_memory.mean(axis=0))

        # Carry the posterior process state into the next timestamp. The
        # filtered physical parameters are still the only EnKF state variables;
        # cumulative memories are deterministic process-model bookkeeping.
        cumulative_liquid_memory = posterior_liquid_memory
        cumulative_sand_memory = posterior_sand_memory
        previous_time_seconds = t_seconds

        prior_liquid, prior_sand = prior_shares[:n_clusters], prior_shares[n_clusters:]
        post_liquid, post_sand = posterior_shares[:n_clusters], posterior_shares[n_clusters:]
        prior_pressure_bias = 0.0
        posterior_pressure_bias = 0.0
        prior_bhp = float(prior["bottomhole_pressure_mpa"]) + prior_pressure_bias
        posterior_bhp = float(posterior["bottomhole_pressure_mpa"]) + posterior_pressure_bias
        bhp_prior_error = abs(prior_bhp - observed_bhp) / max(abs(observed_bhp), 1.0)
        bhp_post_error = abs(posterior_bhp - observed_bhp) / max(abs(observed_bhp), 1.0)
        row = {
            "sequence_index": sequence_index, "source_step": int(source_step), "time_s": t_seconds, "phase": phase,
            "current_total_rate_m3_s": current_total_rate,
            "observed_bottomhole_pressure_mpa": observed_bhp,
            "prior_bottomhole_pressure_mpa": prior_bhp,
            "posterior_bottomhole_pressure_mpa": posterior_bhp,
            "prior_pkn_bottomhole_pressure_mpa": float(prior["bottomhole_pressure_mpa"]),
            "posterior_pkn_bottomhole_pressure_mpa": float(posterior["bottomhole_pressure_mpa"]),
            "prior_near_wellbore_pressure_bias_mpa": prior_pressure_bias,
            "posterior_near_wellbore_pressure_bias_mpa": posterior_pressure_bias,
            "prior_liquid_tvd": total_variation(prior_liquid, observed_liquid),
            "posterior_liquid_tvd": total_variation(post_liquid, observed_liquid),
            "prior_sand_tvd": total_variation(prior_sand, observed_sand),
            "posterior_sand_tvd": total_variation(post_sand, observed_sand),
            "prior_bhp_relative_error": bhp_prior_error,
            "posterior_bhp_relative_error": bhp_post_error,
            "mean_abs_kalman_gain": gain_mean,
            "mean_covariance_inflation": float(np.mean(inflation_values)) if inflation_values else 1.0,
            "observation_memory_mode": f"liquid:{args.observation_memory_mode};sand:{args.sand_observation_memory_mode}",
            "assimilation_iterations": max(int(args.assimilation_iterations), 1) if should_update else 0,
            "posterior_leakoff_fraction": float(posterior["leakoff_fraction"]),
            "posterior_rate_conservation_error": float(posterior["rate_conservation_error"]),
            "prior_total_half_length_m": float(np.asarray(prior["half_length_m"], dtype=float).sum()),
            "posterior_total_half_length_m": float(np.asarray(posterior["half_length_m"], dtype=float).sum()),
            **state_record("prior", prior_state, cfg, n_clusters),
            **state_record("posterior", posterior_state, cfg, n_clusters),
        }
        row["posterior_all_observations_within_15_percent"] = bool(
            row["posterior_liquid_tvd"] <= 0.15 and row["posterior_sand_tvd"] <= 0.15 and row["posterior_bhp_relative_error"] <= 0.15
        )
        row["step_compute_ms"] = (time.perf_counter() - step_started) * 1000.0
        rows.append(row)
        for idx, cluster_id in enumerate(step_controls["cluster_id"].astype(int)):
            cluster_rows.append({
                "sequence_index": sequence_index, "phase": phase, "time_s": t_seconds, "cluster_id": int(cluster_id),
                "observed_liquid_share": observed_liquid[idx], "prior_liquid_share": prior_liquid[idx], "posterior_liquid_share": post_liquid[idx],
                "observed_sand_share": observed_sand[idx], "prior_sand_share": prior_sand[idx], "posterior_sand_share": post_sand[idx],
                "prior_half_length_m": float(np.asarray(prior["half_length_m"], dtype=float)[idx]),
                "posterior_half_length_m": float(np.asarray(posterior["half_length_m"])[idx]),
                "fiber_liquid_allocation": float(fiber_allocation[idx]),
                "posterior_cluster_factor": 1.0,
                "posterior_sand_transport_factor": 1.0,
                "allocation_source": "fiber_incremental_liquid_weight",
            })

    history = pd.DataFrame(rows)
    clusters = pd.DataFrame(cluster_rows)
    pressure_bias_meta: dict[str, object] = {"enabled": False, "observation_mode": args.observation_mode}
    if args.observation_mode == "pressure_only" and not history.empty:
        correction = run_pressure_only_correction(
            history["prior_pkn_bottomhole_pressure_mpa"].to_numpy(dtype=float),
            history["observed_bottomhole_pressure_mpa"].to_numpy(dtype=float),
            PressureOnlyConfig(
                process_std_mpa=float(args.pressure_bias_process_std),
                observation_std_mpa=float(args.bottomhole_pressure_noise_mpa),
                ensemble_size=max(40, min(int(args.ensemble_size), 400)),
                seed=int(args.seed),
            ),
        )
        history["prior_near_wellbore_pressure_bias_mpa"] = correction["prior_bias_mpa"]
        history["posterior_near_wellbore_pressure_bias_mpa"] = correction["posterior_bias_mpa"]
        history["prior_bottomhole_pressure_mpa"] = history["prior_pkn_bottomhole_pressure_mpa"] + history["prior_near_wellbore_pressure_bias_mpa"]
        history["posterior_bottomhole_pressure_mpa"] = correction["posterior_bottomhole_mpa"]
        history["prior_bhp_relative_error"] = (
            np.abs(history["prior_bottomhole_pressure_mpa"] - history["observed_bottomhole_pressure_mpa"])
            / np.maximum(np.abs(history["observed_bottomhole_pressure_mpa"]), 1.0)
        )
        history["posterior_bhp_relative_error"] = (
            np.abs(history["posterior_bottomhole_pressure_mpa"] - history["observed_bottomhole_pressure_mpa"])
            / np.maximum(np.abs(history["observed_bottomhole_pressure_mpa"]), 1.0)
        )
        history["posterior_all_observations_within_15_percent"] = (
            (history["posterior_bhp_relative_error"] <= 0.15)
            & (history["posterior_liquid_tvd"] <= 0.15)
            & (history["posterior_sand_tvd"] <= 0.15)
        )
        pressure_bias_meta = dict(correction["metadata"])
        pressure_bias_meta["enabled"] = True
    validation = history[history["phase"].eq("validation")]
    length_change_records = []
    for cluster_id, group in clusters.groupby("cluster_id"):
        group = group.sort_values("time_s")
        lengths = group["posterior_half_length_m"].to_numpy(dtype=float)
        previous_lengths = lengths[:-1]
        current_lengths = lengths[1:]
        relative_change = (current_lengths - previous_lengths) / np.maximum(np.abs(previous_lengths), 1.0e-9)
        length_change_records.extend(relative_change.tolist())
    length_change_array = np.asarray(length_change_records, dtype=float)
    shrink_gt_5 = length_change_array < -0.05 if length_change_array.size else np.array([], dtype=bool)
    allocation_step_changes = []
    for _, group in clusters.groupby("cluster_id"):
        values = group.sort_values("time_s")["fiber_liquid_allocation"].to_numpy(dtype=float)
        if len(values) > 1:
            allocation_step_changes.extend(np.abs(np.diff(values)).tolist())
    metrics = {
        "calibration_steps": int((history["phase"] == "calibration").sum()),
        "validation_steps": int(len(validation)),
        "validation_liquid_tvd_mean": float(validation["posterior_liquid_tvd"].mean()),
        "validation_sand_tvd_mean": float(validation["posterior_sand_tvd"].mean()),
        "validation_bhp_relative_error_mean": float(validation["posterior_bhp_relative_error"].mean()),
        "validation_prior_liquid_tvd_mean": float(validation["prior_liquid_tvd"].mean()),
        "validation_prior_sand_tvd_mean": float(validation["prior_sand_tvd"].mean()),
        "validation_prior_bhp_relative_error_mean": float(validation["prior_bhp_relative_error"].mean()),
        "all_steps_compute_p50_ms": float(history["step_compute_ms"].quantile(0.50)),
        "all_steps_compute_p95_ms": float(history["step_compute_ms"].quantile(0.95)),
        "all_steps_under_15_seconds_rate": float((history["step_compute_ms"] < 15000.0).mean()),
        "max_rate_conservation_error": float(history["posterior_rate_conservation_error"].max()),
        "mean_posterior_leakoff_fraction": float(history["posterior_leakoff_fraction"].mean()),
        "validation_all_observations_within_15_percent_rate": float(validation["posterior_all_observations_within_15_percent"].mean()),
        "validation_pass": bool(
            validation["posterior_liquid_tvd"].mean() <= 0.15
            and validation["posterior_sand_tvd"].mean() <= 0.15
            and validation["posterior_bhp_relative_error"].mean() <= 0.15
        ),
        "filter_method": args.filter_method,
        "assimilation_iterations": int(args.assimilation_iterations),
        "covariance_inflation": float(args.covariance_inflation),
        "dynamic_pressure_bias": bool(args.observation_mode == "pressure_only"),
        "dynamic_pressure_bias_requested": bool(args.dynamic_pressure_bias),
        "free_cluster_growth_factors": False,
        "use_localization": bool(args.use_localization),
        "physics_profile": args.physics_profile,
        "observation_memory_mode": args.observation_memory_mode,
        "sand_observation_memory_mode": args.sand_observation_memory_mode,
        "sand_transport_exponent": float(args.sand_transport_exponent),
        "adaptive_observation_noise": bool(args.adaptive_observation_noise),
        "adaptive_inflation": bool(args.adaptive_inflation),
        "ensemble_size": int(args.ensemble_size),
        "batch_calibrate": bool(args.batch_calibrate),
        "knowledge_guided_prior": kg_prior,
        "state_dimension": 5,
        "state_vector": ["E_prime", "C_L", "mu", "sigma_min", "K_IC"],
        "observation_mode": args.observation_mode,
        "observation_vector": (["bottomhole_pressure_mpa"] if args.observation_mode == "pressure_only" else ["cumulative_liquid_share_by_cluster", "cumulative_sand_share_by_cluster", "bottomhole_pressure_mpa"]),
        "pressure_bias_update": pressure_bias_meta,
        "observation_quality": quality_meta,
        "pyfrac_residual_surrogate": surrogate_gate,
        "fiber_allocation_source": "incremental_liquid_volume_from_fiber",
        "fiber_allocation_smoothing_previous_weight": float(args.fiber_allocation_smoothing),
        "max_fiber_allocation_step_change": float(max(allocation_step_changes, default=0.0)),
        "length_shrink_gt_5_percent_count": int(shrink_gt_5.sum()),
        "length_transition_count": int(len(length_change_array)),
        "length_shrink_gt_5_percent_rate": float(shrink_gt_5.mean()) if shrink_gt_5.size else 0.0,
        "clusters_with_length_shrink_gt_5_percent": int(
            sum(
                bool(
                    (
                        group.sort_values("time_s")["posterior_half_length_m"].to_numpy(dtype=float)[1:]
                        < 0.95
                        * group.sort_values("time_s")["posterior_half_length_m"].to_numpy(dtype=float)[:-1]
                    ).any()
                )
                for _, group in clusters.groupby("cluster_id")
            )
        ),
    }
    return history, clusters, {
        "metrics": metrics,
        "config": cfg,
        "monitor_meta": monitor_meta,
        "pressure_meta": pressure_meta,
        "batch_calibration": batch_calibration,
        "calibration_center": calibration_center.tolist(),
        "knowledge_guided_prior": kg_prior,
    }


def plot_results(history: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t = history["time_s"] / 60.0
    split = history.loc[history["phase"].eq("validation"), "time_s"].min() / 60.0
    axes[0, 0].plot(t, history["posterior_liquid_tvd"] * 100, label="液量分配误差")
    axes[0, 0].plot(t, history["posterior_sand_tvd"] * 100, label="砂量分配误差")
    axes[0, 0].axhline(15, color="red", ls=":", label="15%")
    axes[0, 0].set_title("可测分簇分配误差"); axes[0, 0].set_ylabel("TVD / %"); axes[0, 0].legend()
    axes[0, 1].plot(t, history["observed_bottomhole_pressure_mpa"], color="black", label="观测")
    axes[0, 1].plot(t, history["posterior_bottomhole_pressure_mpa"], label="模型")
    axes[0, 1].set_title("井底压力留出验证"); axes[0, 1].set_ylabel("MPa"); axes[0, 1].legend()
    axes[1, 0].plot(t, history["posterior_bhp_relative_error"] * 100, color="#d95d39")
    axes[1, 0].axhline(15, color="red", ls=":"); axes[1, 0].set_title("井底压力相对误差"); axes[1, 0].set_ylabel("误差 / %")
    axes[1, 1].plot(t, history["posterior_eprime_gpa"], label="E' / GPa")
    axes[1, 1].plot(t, history["posterior_min_stress_mpa"], label="σmin / MPa")
    axes[1, 1].set_title("EnKF后验物理参数"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.axvline(split, color="#147d78", ls="--", label="留出验证起点")
        ax.set_xlabel("时间 / min"); ax.grid(alpha=.25)
    fig.suptitle("直接观测空间 PKN-EnKF：70%校准 + 30%留出验证", fontsize=16, weight="bold")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main() -> None:
    configure_font(); args = build_parser().parse_args()
    output = Path(args.run_dir).resolve() / time.strftime("%Y%m%d_%H%M%S"); output.mkdir(parents=True, exist_ok=True)
    history, clusters, result = run(args)
    history.to_csv(output / "direct_observation_history.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(output / "cluster_share_history.csv", index=False, encoding="utf-8-sig")
    plot_results(history, output / "direct_observation_validation.png")
    summary = {
        "demo": "direct_observable_space_pkn_enkf_heldout_validation",
        "scientific_status": "engineering validation prototype",
        "observation_mode": args.observation_mode,
        "state_vector": [
            "E'",
            "C_L",
            "mu",
            "sigma_min",
            "K_IC",
        ],
        "observations": (["bottom-hole pressure converted from wellhead pressure"] if args.observation_mode == "pressure_only" else ["cumulative liquid share by cluster", "cumulative sand share by cluster", "bottom-hole pressure"]),
        "anti_circularity_design": (
            "Fiber cumulative liquid shares drive historical PKN cluster rates and smoothed incremental fiber liquid "
            "allocation drives current-rate/aperture inputs; EnKF does not contain a free cluster growth/intake factor."
        ),
        "allocation_method": "fiber incremental liquid allocation with light EMA smoothing and nonnegative conservation normalization",
        "validation_design": (
            "First 70% calibrates the state. Online validation scores each held-out step before update, then "
            "assimilates the arriving observation and scores the posterior; frozen mode does not update held-out steps."
        ),
        "validation_mode": args.validation_mode,
        "metrics": result["metrics"],
        "batch_calibration": result["batch_calibration"],
        "calibration_center_state": result["calibration_center"],
        "config": asdict(result["config"]),
        "limitations": [
            "Cluster liquid/sand shares constrain intake allocation, not independently measured fracture geometry." if args.observation_mode != "pressure_only" else "DAS is unavailable; cluster liquid/sand shares and cluster geometry are not inferred.",
            "The sand transport observation operator is a cumulative, mean-preserving sublinear q_effective*aperture capacity proxy; it is not a full particle-transport solver.",
            "Pressure friction defaults require client calibration before field interpretation.",
            "The 50% cumulative leakoff cap is an engineering prior and must be recalibrated when formation leakoff measurements are available.",
            "The EnKF state is intentionally limited to E', C_L, mu, sigma_min and K_IC; cluster allocation is measured input, not a free state.",
        ],
        "outputs": {"history": str(output / "direct_observation_history.csv"), "clusters": str(output / "cluster_share_history.csv"), "figure": str(output / "direct_observation_validation.png")},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(output), "metrics": summary["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
