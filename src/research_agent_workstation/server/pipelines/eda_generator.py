from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_eda_summary(train_path: Path, target: str) -> dict[str, Any]:
    df = pd.read_csv(train_path)
    missing = df.isna().mean().sort_values(ascending=False).head(20)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": target,
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": [col for col in df.columns if col not in df.select_dtypes(include="number").columns],
        "missing_top20": {str(k): round(float(v), 4) for k, v in missing.items()},
    }

