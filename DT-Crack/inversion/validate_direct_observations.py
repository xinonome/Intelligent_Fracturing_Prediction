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
from data_fusion.pressure_schedule_adapter import PressureModelConfig
from inversion import (
    PhysicalEnKFConfig,
    enkf_update,
    physical_values,
    pkn_with_carter_leakoff,
    state_record,
)


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


def predicted_observation(
    forward: dict, n_clusters: int, sand_transport_factors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    liquid_share = normalize_positive(np.asarray(forward["q_effective_m3_s"], dtype=float))
    transport = np.asarray(forward["q_effective_m3_s"], dtype=float) * np.maximum(
        np.asarray(forward["max_aperture_mm"], dtype=float), 1e-6
    )
    if sand_transport_factors is not None:
        transport *= np.asarray(sand_transport_factors, dtype=float)
    sand_share = normalize_positive(transport)
    # The last share in each group is implied by the sum-to-one constraint.
    observation = np.r_[liquid_share[: n_clusters - 1], sand_share[: n_clusters - 1], forward["bottomhole_pressure_mpa"]]
    return observation, np.r_[liquid_share, sand_share]


def clip_augmented_state(state: np.ndarray, n_clusters: int) -> np.ndarray:
    """Clip global physics, dynamic intake factors and sand-transport factors."""
    out = np.asarray(state, dtype=float).copy()
    out[..., 0] = np.clip(out[..., 0], np.log(0.45), np.log(2.2))
    out[..., 1] = np.clip(out[..., 1], np.log(0.1), np.log(8.0))
    out[..., 2] = np.clip(out[..., 2], np.log(0.2), np.log(5.0))
    out[..., 3] = np.clip(out[..., 3], 35.0, 90.0)
    out[..., 4 : 4 + n_clusters] = np.clip(out[..., 4 : 4 + n_clusters], 0.10, 5.0)
    out[..., 4 + n_clusters : 4 + 2 * n_clusters] = np.clip(
        out[..., 4 + n_clusters : 4 + 2 * n_clusters], 0.10, 5.0
    )
    return out


def total_variation(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(predicted) - np.asarray(observed)).sum())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct-observation PKN-EnKF calibration and held-out validation.")
    parser.add_argument("--frac-monitor-text", required=True)
    parser.add_argument("--construction-pressure-xls", required=True)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--calibration-ratio", type=float, default=0.70)
    # 300 members materially reduce seed sensitivity while remaining far below
    # the 15-second online update budget on the current six-cluster problem.
    parser.add_argument("--ensemble-size", type=int, default=300)
    parser.add_argument("--share-noise", type=float, default=0.025)
    parser.add_argument("--bottomhole-pressure-noise-mpa", type=float, default=5.0)
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--base-min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--pressure-proxy-scale", type=float, default=30.0)
    parser.add_argument("--hydraulic-coupling-mode", choices=["legacy", "coupled"], default="coupled")
    parser.add_argument("--conductance-exponent", type=float, default=0.45)
    parser.add_argument("--measured-depth-m", type=float, default=5218.0)
    parser.add_argument("--vertical-depth-m", type=float, default=3196.94)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-mode", choices=["frozen", "online"], default="online")
    parser.add_argument("--run-dir", default=str(DT_ROOT.parent / "outputs" / "dt" / "direct_observation_enkf"))
    return parser


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(args.seed)
    monitor = load_frac_monitor_text(args.frac_monitor_text, default_step_seconds=1.0)
    controls = monitor.controls
    n_clusters = int(controls["cluster_id"].nunique())
    available_steps = np.asarray(sorted(controls["step"].unique()), dtype=int)
    count = min(max(args.max_steps, 2), len(available_steps))
    indices = np.linspace(0, len(available_steps) - 1, count).round().astype(int)
    source_steps = np.unique(available_steps[indices])
    calibration_count = max(1, min(len(source_steps) - 1, int(round(len(source_steps) * args.calibration_ratio))))

    pressure_cfg = PressureModelConfig(min_horizontal_stress_mpa=args.base_min_stress_mpa)
    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls, pressure_cfg, args.measured_depth_m, args.vertical_depth_m
    )
    cfg = PhysicalEnKFConfig(
        base_eprime_pa=args.base_eprime_pa,
        base_leakoff_m_sqrt_s=args.base_leakoff_m_sqrt_s,
        base_viscosity_pa_s=args.base_viscosity_pa_s,
        base_min_stress_mpa=args.base_min_stress_mpa,
        height_m=args.height_m,
        pressure_proxy_scale=args.pressure_proxy_scale,
        hydraulic_coupling_mode=args.hydraulic_coupling_mode,
        conductance_exponent=args.conductance_exponent,
    )
    prior_mean = np.r_[0.0, 0.0, 0.0, args.base_min_stress_mpa, np.ones(n_clusters), np.ones(n_clusters)]
    spread = np.r_[0.18, 0.45, 0.30, 5.0, np.full(n_clusters, 0.30), np.full(n_clusters, 0.30)]
    process = np.r_[0.012, 0.020, 0.015, 0.20, np.full(n_clusters, 0.06), np.full(n_clusters, 0.06)]
    ensemble = clip_augmented_state(
        prior_mean + rng.normal(0.0, spread, size=(args.ensemble_size, len(prior_mean))), n_clusters
    )
    rows: list[dict] = []
    cluster_rows: list[dict] = []

    for sequence_index, source_step in enumerate(source_steps):
        step_started = time.perf_counter()
        phase = "calibration" if sequence_index < calibration_count else "validation"
        step_controls = controls_for_step(controls, int(source_step))
        t_seconds = max(float(source_step), 1.0)
        total_cumulative = float(step_controls["cumulative_liquid_volume_m3"].sum())
        total_rate = max(total_cumulative / t_seconds, 1e-8)
        current_total_rate = max(float(step_controls["flow_rate_m3_min"].sum()) / 60.0, 1e-8)
        # Uniform base allocation is deliberate: observed cluster shares are not fed into the forward input.
        q_base = np.full(n_clusters, total_rate / n_clusters, dtype=float)
        q_current = np.full(n_clusters, current_total_rate / n_clusters, dtype=float)
        observed_liquid, observed_sand = observed_cluster_shares(step_controls)
        observed_bhp = float(pressure_for_step(pressure, int(source_step))["bottomhole_pressure_mpa"])

        should_update = phase == "calibration" or args.validation_mode == "online"
        if should_update:
            ensemble = clip_augmented_state(
                ensemble + rng.normal(0.0, process, size=ensemble.shape), n_clusters
            )
        predictions = [pkn_with_carter_leakoff(member, q_base, t_seconds, cfg, q_current) for member in ensemble]
        predicted_obs_rows = np.asarray([
            predicted_observation(item, n_clusters, member[4 + n_clusters : 4 + 2 * n_clusters])[0]
            for item, member in zip(predictions, ensemble)
        ])
        observed_obs = np.r_[observed_liquid[: n_clusters - 1], observed_sand[: n_clusters - 1], observed_bhp]
        obs_std = np.r_[np.full(2 * (n_clusters - 1), args.share_noise), args.bottomhole_pressure_noise_mpa]

        prior_state = ensemble.mean(axis=0)
        prior = pkn_with_carter_leakoff(prior_state, q_base, t_seconds, cfg, q_current)
        _, prior_shares = predicted_observation(
            prior, n_clusters, prior_state[4 + n_clusters : 4 + 2 * n_clusters]
        )
        gain_mean = 0.0
        if should_update:
            ensemble, gain = enkf_update(ensemble, predicted_obs_rows, observed_obs, obs_std, rng)
            ensemble = clip_augmented_state(ensemble, n_clusters)
            gain_mean = float(np.mean(np.abs(gain)))
        posterior_state = ensemble.mean(axis=0)
        posterior = pkn_with_carter_leakoff(posterior_state, q_base, t_seconds, cfg, q_current)
        _, posterior_shares = predicted_observation(
            posterior, n_clusters, posterior_state[4 + n_clusters : 4 + 2 * n_clusters]
        )

        prior_liquid, prior_sand = prior_shares[:n_clusters], prior_shares[n_clusters:]
        post_liquid, post_sand = posterior_shares[:n_clusters], posterior_shares[n_clusters:]
        bhp_prior_error = abs(float(prior["bottomhole_pressure_mpa"]) - observed_bhp) / max(abs(observed_bhp), 1.0)
        bhp_post_error = abs(float(posterior["bottomhole_pressure_mpa"]) - observed_bhp) / max(abs(observed_bhp), 1.0)
        row = {
            "sequence_index": sequence_index, "source_step": int(source_step), "time_s": t_seconds, "phase": phase,
            "current_total_rate_m3_s": current_total_rate,
            "observed_bottomhole_pressure_mpa": observed_bhp,
            "prior_bottomhole_pressure_mpa": float(prior["bottomhole_pressure_mpa"]),
            "posterior_bottomhole_pressure_mpa": float(posterior["bottomhole_pressure_mpa"]),
            "prior_liquid_tvd": total_variation(prior_liquid, observed_liquid),
            "posterior_liquid_tvd": total_variation(post_liquid, observed_liquid),
            "prior_sand_tvd": total_variation(prior_sand, observed_sand),
            "posterior_sand_tvd": total_variation(post_sand, observed_sand),
            "prior_bhp_relative_error": bhp_prior_error,
            "posterior_bhp_relative_error": bhp_post_error,
            "mean_abs_kalman_gain": gain_mean,
            "posterior_leakoff_fraction": float(posterior["leakoff_fraction"]),
            "posterior_rate_conservation_error": float(posterior["rate_conservation_error"]),
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
                "posterior_half_length_m": float(np.asarray(posterior["half_length_m"])[idx]),
                "posterior_cluster_factor": float(np.asarray(physical_values(posterior_state, cfg, n_clusters)["cluster_factors"])[idx]),
                "posterior_sand_transport_factor": float(posterior_state[4 + n_clusters + idx]),
            })

    history = pd.DataFrame(rows)
    clusters = pd.DataFrame(cluster_rows)
    validation = history[history["phase"].eq("validation")]
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
    }
    return history, clusters, {"metrics": metrics, "config": cfg, "monitor_meta": monitor.meta, "pressure_meta": pressure_meta}


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
        "state_vector": ["E'", "C_L", "mu", "sigma_min", "dynamic cluster intake/growth factors", "dynamic cluster sand-transport factors"],
        "observations": ["cumulative liquid share by cluster", "cumulative sand share by cluster", "bottom-hole pressure"],
        "anti_circularity_design": "PKN receives total rate with uniform base allocation; observed cluster shares are used only in the EnKF residual.",
        "validation_design": (
            "First 70% calibrates the state. Online validation scores each held-out step before update, then "
            "assimilates the arriving observation and scores the posterior; frozen mode does not update held-out steps."
        ),
        "validation_mode": args.validation_mode,
        "metrics": result["metrics"],
        "config": asdict(result["config"]),
        "limitations": [
            "Cluster liquid/sand shares constrain intake allocation, not independently measured fracture geometry.",
            "The sand transport observation operator is a reduced q_effective*aperture proxy.",
            "Pressure friction defaults require client calibration before field interpretation.",
        ],
        "outputs": {"history": str(output / "direct_observation_history.csv"), "clusters": str(output / "cluster_share_history.csv"), "figure": str(output / "direct_observation_validation.png")},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(output), "metrics": summary["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
