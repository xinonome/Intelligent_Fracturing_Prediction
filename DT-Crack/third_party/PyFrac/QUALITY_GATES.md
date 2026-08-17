# PLAN (1) quality-gate status

Last checked: 2026-08-12.

| Gate | Status | Evidence / boundary |
|---|---|---|
| QG0 | PASS | `scripts/run_phase0.py`, archived `outputs/baseline/phase0_summary.json`, upstream regression harness 11/11 |
| QG1 | PASS | strict config/input layer; `tests/multistage` input tests pass |
| QG2 | PASS | stage boundary inclusion, E-prime median, stress interpolation/no extrapolation and coordinate tests pass |
| QG3 | PASS for the checked demo case | `scripts/compare_adapter_direct.py` compares wrapper and direct PyFrac aggregate outputs with 5% tolerance; must be rerun for production settings |
| QG4 | PASS for synthetic demo | three stages run in config order and create independent outputs; real Well X remains data-dependent |
| QG5 | PASS for synthetic demo | local/global transform round trip and common 3D export verified |
| QG6 | PASS | synthetic barrier and consecutive-snapshot handover tests pass |
| QG7 | PASS for viscosity smoke matrix; conditional for toughness | low/high viscosity summary generated; toughness cases run only when explicit KIC values are placed in matrix |
| QG8 | BLOCKED / FAIL | baseline and memory guard pass, but current snapshot/grid convergence has medium-vs-fine differences above 5% (length 7.14%, volume 0.69%); mass balance is explicitly NOT_COMPUTED because leak-off volume is not exposed |
| QG9 | NOT RUN | no fresh isolated environment was created; current environment and dependency versions are recorded |

The project must not describe the current result as a fully coupled multi-stage fracture-interaction simulator. It is an independent planar-stage workflow with a common global reconstruction.
