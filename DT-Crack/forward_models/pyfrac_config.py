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
    # The legacy solver cannot resolve a sub-cell fracture at t=1 s on the
    # coarse reference mesh. Native runs therefore start from a resolved PKN
    # warm state and then time-march with PyFrac's Controller.
    native_start_time_s: float = 120.0
    # This is an upper reference scale; the adapter also scales the domain
    # from the requested PKN length so small smoke cases are resolved.
    mesh_half_length_m: float = 80.0
    mesh_half_height_m: float = 25.0
    mesh_nx: int = 61
    mesh_ny: int = 41
    max_time_steps: int = 30
    dynamic_step_limit_s: float = 30.0
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
        values["e_prime_pa"] = self.e_prime_pa
        return values

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_pyfrac_config(path: str | Path | None = None) -> PyFracConfig:
    if path is None:
        return PyFracConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {key: value for key, value in payload.items() if key in PyFracConfig.__dataclass_fields__}
    return PyFracConfig(**fields)
