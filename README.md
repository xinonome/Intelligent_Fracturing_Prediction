# Intelligent Fracturing Prediction

This repository is a sanitized public code snapshot of the intelligent fracturing prediction project.

The snapshot contains the reusable Python code, model interfaces, EnKF implementation, application skeleton, tests, and configuration templates. Original well data, customer data, trained model artifacts, runtime outputs, reports, screenshots, delivery archives, and extracted knowledge-graph data are intentionally excluded.

## Main modules

- `DT-Crack/`: data adapters, PKN forward model, EnKF inversion, PyFrac adapter, and visualization code.
- `FSL-Expert/`: expert-rule and GNN experiment code. The extracted graph data is not included.
- `HMI-KE/`: decision-engine and simulation environment code.
- `App/`: desktop application skeleton and service layer.
- `tools/`: data and delivery utilities.

## Setup

Use a fresh Python environment and install only the dependencies needed for the module being used:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
```

Optional dependency groups are listed in `requirements-kg.txt`, `requirements-rl.txt`, and `requirements-ui.txt`. The PyFrac multistage component has a separate manifest at `DT-Crack/third_party/PyFrac/requirements-multistage.txt`.

The included registry is intentionally empty of historical runs and points to no customer files. Add an authorized data/artifact package separately and update `App/config/demo_registry.json` for a local replay.

## Important limitation

This public snapshot does not contain the original input data or generated outputs, so the end-to-end replay and application demo require a separately supplied, authorized data package. Do not add customer data, raw well data, trained models, API keys, or local machine paths to a public repository.

The application launcher accepts portable environment overrides:

```powershell
$env:FRACTURING_DATA_ROOT = "C:\path\to\authorized\data"
$env:FRACTURING_ALGORITHM_PYTHON = (Get-Command python).Source
$env:FRACTURING_QT_PYTHON = (Get-Command python).Source
```

## License

The PyFrac third-party component retains its upstream license in `DT-Crack/LICENSE`. Review third-party licensing before publishing a fork or redistributing modified components.
