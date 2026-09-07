"""Causal-style evaluation for Piggy-Bank balance improvement.

This evaluator refuses to call a single historical replay a causal proof.  It
can calculate an association for any two groups, but it only marks the result
``causal_estimable`` when the input declares an appropriate design (for
example, randomized or matched control stages), contains both policy groups,
and has enough complete stage outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALID_DESIGNS = {"randomized", "matched_control", "difference_in_differences"}


def _bootstrap_effect(treatment: np.ndarray, control: np.ndarray, *, seed: int = 2026, samples: int = 2000) -> tuple[float, float]:
    if len(treatment) == 0 or len(control) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    effects = np.empty(max(int(samples), 100), dtype=float)
    for idx in range(len(effects)):
        effects[idx] = float(rng.choice(treatment, len(treatment), replace=True).mean() - rng.choice(control, len(control), replace=True).mean())
    return float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))


def evaluate_balance_causality(
    stages: pd.DataFrame,
    *,
    treatment_label: str = "piggy_bank",
    control_label: str = "baseline",
    design_type: str = "observational",
    seed: int = 2026,
    bootstrap_samples: int = 2000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"policy", "balance_before", "balance_after"}
    missing = required - set(stages.columns)
    if missing:
        raise ValueError(f"causal evaluation missing columns: {sorted(missing)}")
    frame = stages.copy()
    frame["policy"] = frame["policy"].astype(str)
    frame["balance_before"] = pd.to_numeric(frame["balance_before"], errors="coerce")
    frame["balance_after"] = pd.to_numeric(frame["balance_after"], errors="coerce")
    frame["balance_improvement"] = frame["balance_after"] - frame["balance_before"]
    frame["complete_outcome"] = np.isfinite(frame["balance_improvement"])
    treatment = frame[(frame["policy"] == treatment_label) & frame["complete_outcome"]]["balance_improvement"].to_numpy(dtype=float)
    control = frame[(frame["policy"] == control_label) & frame["complete_outcome"]]["balance_improvement"].to_numpy(dtype=float)
    association = float(treatment.mean() - control.mean()) if len(treatment) and len(control) else float("nan")
    ci_low, ci_high = _bootstrap_effect(treatment, control, seed=seed, samples=bootstrap_samples)
    enough_groups = len(treatment) >= 2 and len(control) >= 2
    design_ok = str(design_type).lower() in VALID_DESIGNS
    status = "causal_estimable" if enough_groups and design_ok else "not_estimable"
    summary: dict[str, Any] = {
        "status": status,
        "design_type": str(design_type),
        "treatment_label": treatment_label,
        "control_label": control_label,
        "treatment_stage_count": int(len(treatment)),
        "control_stage_count": int(len(control)),
        "treatment_mean_balance_improvement": float(treatment.mean()) if len(treatment) else None,
        "control_mean_balance_improvement": float(control.mean()) if len(control) else None,
        "estimated_effect_treatment_minus_control": association if np.isfinite(association) else None,
        "bootstrap_95ci_low": ci_low if np.isfinite(ci_low) else None,
        "bootstrap_95ci_high": ci_high if np.isfinite(ci_high) else None,
        # The evaluator can establish that an effect estimate is computable;
        # it must not turn a CSV declaration into a scientific proof.  Final
        # proof still requires independent design review, adequate sample
        # size, safety non-inferiority and field acceptance.
        "causal_effect_proven": False,
        "causal_effect_estimate_available": status == "causal_estimable",
        "limitations": [
            "A historical replay with no untreated or matched comparison is not causal evidence.",
            "The result measures balance-degree change, not production improvement.",
            "Pressure and safety non-inferiority must be evaluated separately before field use.",
        ],
    }
    return frame, summary


def run_causal_evaluation(input_csv: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    frame = pd.read_csv(input_csv)
    evaluated, summary = evaluate_balance_causality(frame, **kwargs)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output / "balance_causal_stage_metrics.csv", index=False, encoding="utf-8-sig")
    summary["outputs"] = {"stage_metrics": str(output / "balance_causal_stage_metrics.csv")}
    (output / "balance_causal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
