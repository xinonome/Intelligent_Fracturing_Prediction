from __future__ import annotations

import pandas as pd


def replay_rows(path: str, max_rows: int = 100) -> list[dict]:
    df = pd.read_excel(path, nrows=max_rows) if path.lower().endswith(".xlsx") else pd.read_csv(path, nrows=max_rows)
    return df.fillna("").to_dict(orient="records")
