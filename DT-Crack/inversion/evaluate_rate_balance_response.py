"""CLI for the total-stage-rate versus cluster-balance sensitivity scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inversion.physics import PhysicalEnKFConfig
from inversion.rate_balance_response import (
    RateBalanceResponseConfig,
    rows_to_csv,
    scan_rate_balance_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan stage-total rate impact on predicted cluster balance.")
    parser.add_argument("--state-json", required=True, help="JSON array containing one PKN/EnKF state.")
    parser.add_argument("--total-rates-m3-s", required=True, help="Comma-separated positive total rates.")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-identifiable-range", type=float, default=1.0e-3)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    state = np.asarray(json.loads(Path(args.state_json).read_text(encoding="utf-8")), dtype=float)
    rates = [float(value.strip()) for value in str(args.total_rates_m3_s).split(",") if value.strip()]
    rows, summary = scan_rate_balance_response(
        state,
        rates,
        physics_config=PhysicalEnKFConfig(),
        config=RateBalanceResponseConfig(
            duration_s=float(args.duration_s),
            n_clusters=int(args.n_clusters),
            minimum_identifiable_balance_range=float(args.minimum_identifiable_range),
        ),
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_to_csv(rows, str(output / "rate_balance_response.csv"))
    (output / "rate_balance_response_summary.json").write_text(
        json.dumps(
            {**summary, "outputs": {"response_csv": str(output / "rate_balance_response.csv")}},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
