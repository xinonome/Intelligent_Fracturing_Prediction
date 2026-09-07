"""Configuration for the vendored PyFrac reference solver.

The project adapter deliberately keeps these settings outside the PyFrac
source tree.  This makes the third-party code reproducible and keeps all
client-specific data mapping in the project code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class PyFracConfig:
    """Numerical and physical defaults for a single planar fracture."""

    pyfrac_root: str = ""
    young_modulus_pa: float = 3.2e10
    poisson_ratio: float = 0.25
    fracture_toughness_pa_sqrt_m: float = 5.0e5
    leakoff_coefficient_m_sqrt_s: float = 1.0e-6
    viscosity_pa_s: float = 0.1
    min_horizontal_stress_pa: float = 6.0e7
    height_m: float = 30.0
    initial_time_s: float = 1.0
    # Native continuation now starts from the explicit t=1 s initial
    # condition.  Keep this field for compatibility with older callers, but
    # values greater than initial_time_s are a warm start and are rejected by
    # the formal full-process runner.
    native_start_time_s: float = 1.0
    # The first native windows use a smaller maximum time step.  The session
    # can relax it after several successful windows, while retaining the
    # actual value in the checkpoint/audit record.
    initial_step_limit_s: float = 1.0
    # This is an upper reference scale; the adapter also scales the domain
    # from the requested PKN length so small smoke cases are resolved.
    mesh_half_length_m: float = 80.0
    mesh_half_height_m: float = 25.0
    mesh_nx: int = 61
    mesh_ny: int = 41
    max_time_steps: int = 30
    dynamic_step_limit_s: float = 30.0
    # The legacy controller limits a positive-injection step by the fraction
    # of the current fracture volume added in one step. Keep the historical
    # values as the reproducible/default physics path, but expose them so a
    # measured throughput experiment can trade step count against nonlinear
    # robustness explicitly instead of silently editing controller constants.
    injection_volume_step_fraction: float = 0.10
    positive_volume_change_step_fraction: float = 0.12
    negative_volume_change_step_fraction: float = 0.05
    time_step_time_fraction: float = 0.15
    # Multiplier for the legacy velocity/cell traversal criterion.  1.0 is
    # the historical setting. Values above 1.0 are an explicitly labelled
    # throughput experiment: the front may cross more than one cell, so the
    # structural audit and final acceptance checks remain mandatory.
    cell_traversal_fraction: float = 1.0
    # Native PyFrac EHL solver.  ``implicit_Anderson`` is the legacy default;
    # ``RKL2`` is exposed for a measured high-throughput experiment only.
    elastohydr_solver: str = "implicit_Anderson"
    # Solve the native viscous formulation for pressure increments by
    # default.  The absolute-pressure formulation is retained as an
    # explicit diagnostic because it can behave differently after a mesh
    # regrid; every experiment must record which formulation was used.
    solve_delta_p: bool = True
    # Experimental conservative projection for the legacy viscous path.
    # Disabled by default; enabled runs are labelled separately.
    enable_volume_balance_projection: bool = False
    # Maximum fraction of the smallest cell crossed by the fastest tip in one
    # accepted step.  This is applied before legacy front reconstruction.
    front_cfl: float = 0.8
    # Legacy PyFrac also limits a step to a fraction of the current fracture
    # length.  Keep the legacy value available, but expose it so controlled
    # native runs can test the front-CFL guard independently.
    front_length_fraction: float = 0.20
    # Legacy continuous-front reconstruction is fragile during the first
    # native continuation step.  Keep predictor-corrector as the compatibility
    # default, while allowing the formal runner to select the more robust
    # implicit front solve for a controlled experiment.
    front_advancing: str = "predictor-corrector"
    # The legacy continuous-front projection is the current source of the
    # t=1 s continuation failure.  Keep it as the compatibility default, while
    # allowing the formal runner to validate PyFrac's original ILSA projection.
    projection_method: str = "LS_continousfront"
    # Robust native continuation controls.  These are adapter-level settings;
    # the vendored PyFrac source remains unchanged.
    adaptive_mesh_enabled: bool = True
    min_front_cells: int = 8
    boundary_margin_cells: int = 6
    mesh_refinement_factor: float = 1.35
    max_mesh_levels: int = 3
    max_mesh_nx: int = 241
    max_mesh_ny: int = 121
    enable_pyfrac_remeshing: bool = True
    # Native PyFrac can extend the computational domain when the front reaches
    # a boundary.  Keep this opt-in because legacy remeshing changes the
    # numerical grid and must be explicitly recorded in an experiment.  When
    # extension is enabled, all four sides are allowed by default.  The
    # continuous-front solver may hit the vertical boundary even in an
    # otherwise horizontally growing fracture; leaving the vertical sides
    # disabled makes the legacy controller fall back to domain compression,
    # which can destroy the retained front state during continuation.
    mesh_extension_enabled: bool = False
    mesh_extension_all_directions: bool = False
    mesh_extension_factor: float = 1.5
    # When extension is disabled, a boundary hit can be handled by a
    # controlled coarsening/recentering remesh.  Keep the legacy value as the
    # default for compatibility; long-process experiments may choose a
    # smaller factor and must record it in the run configuration.
    remesh_factor: float = 10.0
    # Diagnostic only: skip zero-rate shut-in intervals to the next positive
    # injection event. The skipped interval is written to the native audit;
    # this mode is never considered a fully resolved shut-in validation.
    skip_zero_injection_intervals: bool = False
    # Prevent a native time step from straddling a positive/zero injection
    # transition.  This preserves the continuous PDE solve while avoiding a
    # large nonlinear state jump at shut-in/restart boundaries.
    clip_to_injection_regime_events: bool = True
    # Optional long-run strategy: when a front reaches a boundary, enlarge
    # the physical horizontal domain while keeping the current cell count.
    # This avoids the legacy compression path artificially trapping volume.
    expand_domain_on_boundary: bool = False
    domain_expansion_factor: float = 2.0
    mesh_extension_directions: tuple[bool, bool, bool, bool] = (True, True, True, True)
    # Bound the legacy controller's inner retry/iteration work so the outer
    # checkpoint runner can regain control after a non-convergent step.
    max_solver_iterations: int = 140
    # Relative nonlinear EHL convergence tolerance.  The legacy PyFrac
    # default is intentionally retained; long-run throughput experiments can
    # override it explicitly and must record that choice in their output.
    ehl_tolerance: float = 1.0e-4
    # Anderson damping.  1.0 preserves the legacy undamped iteration;
    # smaller values are available for re-opening/rapid rate-change tests.
    ehl_relaxation: float = 1.0
    max_front_iterations: int = 25
    max_pyfrac_reattempts: int = 8
    checkpoint_enabled: bool = True
    max_retries: int = 2
    retry_time_step_factor: float = 0.5
    min_dynamic_step_s: float = 0.5
    assimilation_relaxation: float = 0.35
    assimilation_max_parameter_step: tuple[float, ...] = (0.08, 0.20, 0.12, 3.0, 0.12)
    convergence_relative_tolerance: float = 0.05
    mass_balance_tolerance: float = 0.10
    allow_dynamic_fallback_to_snapshot: bool = False

    @property
    def e_prime_pa(self) -> float:
        return self.young_modulus_pa / max(1.0 - self.poisson_ratio**2, 1.0e-9)

    def resolved_root(self, project_root: str | Path | None = None) -> Path:
        if self.pyfrac_root:
            return Path(self.pyfrac_root).expanduser().resolve()
        if project_root is not None:
            return Path(project_root).resolve() / "DT-Crack" / "third_party" / "PyFrac"
        return Path(__file__).resolve().parents[1] / "third_party" / "PyFrac"

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["assimilation_max_parameter_step"] = list(self.assimilation_max_parameter_step)
        values["e_prime_pa"] = self.e_prime_pa
        return values

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_pyfrac_config(path: str | Path | None = None) -> PyFracConfig:
    if path is None:
        return PyFracConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {key: value for key, value in payload.items() if key in PyFracConfig.__dataclass_fields__}
    if "mesh_extension_directions" in fields:
        fields["mesh_extension_directions"] = tuple(bool(value) for value in fields["mesh_extension_directions"])
    return PyFracConfig(**fields)
