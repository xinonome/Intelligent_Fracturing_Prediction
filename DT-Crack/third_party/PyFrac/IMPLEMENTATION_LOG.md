# Multi-stage PyFrac implementation log

## Audit basis

Audited against `C:\Users\xinonome\Downloads\PLAN (1).md` on 2026-08-12. The existing project already had a vendored, unmodified PyFrac 1.1.1 and a project-level single-case adapter, but did not have the planned `multistage` application layer.

## Phase 0 / QG0

- PyFrac source: `src/`; source files were not changed by this implementation.
- Project adapter: `DT-Crack/forward_models/pyfrac_adapter.py`.
- NumPy compatibility remains process-local in the adapter and baseline scripts.
- `python scripts/smoke_pyfrac.py`: PASS.
- `python scripts/reproduce_radial.py`: PASS; finite initialized width and volume.
- `python scripts/reproduce_height_contained.py`: PASS; finite initialized width and volume.
- `python scripts/run_phase0.py`: archives `outputs/baseline/phase0_summary.json`.
- The full upstream regression tests require the same NumPy aliases; `python scripts/run_regression_tests.py` is the supported harness.

## Implemented application layer

- `multistage/config.py`, `schemas.py`, `io.py`, `validation.py`: strict SI config and input contracts, no unresolved `REQUIRED` values.
- `stage_definition.py`, `trajectory.py`, `property_mapping.py`: inclusive stage slicing, E-prime median/statistics, stress interpolation without extrapolation, explicit toughness/leakoff source selection.
- `multistage/pyfrac_adapter.py`: StageSimulationInput, normalized snapshots, current upstream injection history format, memory guard; wraps the existing project adapter.
- `stage_runner.py`, `multistage_runner.py`: config-order stage execution, independent V1 stages, stage-specific files and fail-fast status.
- `global_mapping.py`: local transverse/vertical plane to global X/Y/Z with TVD sign convention and near-vertical guard.
- `diagnostics.py`: geometry metrics, boundary contact, derivative/consecutive-snapshot handover evidence.
- `sensitivity.py`, `cli.py`, `manifest.py`, `export.py`, `plotting.py`, `report.py`: viscosity matrix, explicit toughness refusal, reproducible metadata, exports, plots and report.

## Commands verified

```text
python -m multistage.cli validate --config configs/well_x.yaml       PASS
python -m multistage.cli run --config configs/well_x.yaml            PASS (synthetic demo)
python -m pytest tests/multistage -q                              PASS
python scripts/run_phase0.py                                      PASS
python scripts/compare_adapter_direct.py                         PASS for checked demo input
```

## Known limitations / not yet a scientific acceptance claim

1. The checked-in `data/raw/well_x_*.csv` files are synthetic workflow fixtures, not field Well X logs. Real three-stage QG4/QG8 acceptance remains data-dependent.
2. The current project adapter exposes aggregate scalar width/pressure and does not expose an independent leak-off volume, so mass balance is recorded as `NOT_COMPUTED` rather than fabricated.
3. Native PyFrac time marching is available in the pre-existing adapter but is slow/numerically sensitive; the demo config uses snapshot initialization at requested times.
4. Direct-upstream-vs-adapter comparison, production mesh convergence, and full field-data scientific regression still require the actual field material/trajectory inputs and are not marked PASS by this log.
5. V1 is independent planar stage reconstruction; it does not include stress shadow, stage interaction, proppant transport, reservoir coupling or RL.

## Scientific gate observations

- The checked synthetic convergence run is archived under `outputs/well_x/convergence/`. Medium-vs-fine length difference is 7.14%, so the plan's 5% convergence acceptance criterion is not passed. This is intentionally retained as a failed gate; the workflow does not promote these settings to production sensitivity.
- The same run records volume-balance status as `NOT_COMPUTED` because the current adapter does not expose independent leak-off volume.
