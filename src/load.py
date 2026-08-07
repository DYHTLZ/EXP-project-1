"""Data loading utilities for the pricing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sample_claims.csv"
)

REQUIRED_COLUMNS = {
    "policy_id",
    "age",
    "gender",
    "territory",
    "vehicle_type",
    "annual_miles",
    "driving_experience_years",
    "exposure",
    "claim_count",
    "total_claim_amount",
}


def load_claims(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the claims CSV and validate the expected schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Claims data not found at {path}. Run data/generate_data.py first."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df
