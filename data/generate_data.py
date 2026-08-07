"""Synthetic policy/claim data generator for Project 1.

Produces a policy-level dataset with a claim process that is simple enough
to audit but rich enough to make the fairness question interesting:

* Claim frequency is Poisson with a rate driven by age (U-shaped), gender,
  territory, annual mileage and driving experience.
* Claim severity is Gamma with a mean driven by age, territory and vehicle type.

By construction, protected attributes (gender, territory) genuinely affect
claim rates. That is the point: base rates differ across groups, which is
exactly the setting where fairness metrics can conflict (Chouldechova-style
impossibility) and where "what does fairness cost?" becomes a real question.

Usage:
    python data/generate_data.py --n-policies 100000 --output data/claims_full.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _age_risk(age: np.ndarray) -> np.ndarray:
    """U-shaped age effect: youngest and oldest drivers are riskiest."""
    return 0.35 * np.abs((age - 45.0) / 10.0)


def generate_policies(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate ``n`` synthetic policies with claim outcomes."""
    n = int(n)

    age = rng.integers(18, 81, size=n).astype(float)
    gender = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    territory = rng.choice(["A", "B", "C"], size=n, p=[0.40, 0.35, 0.25])
    vehicle_type = rng.choice(["sedan", "suv", "truck"], size=n, p=[0.50, 0.30, 0.20])
    annual_miles = rng.integers(5_000, 36_000, size=n).astype(float)
    exposure = rng.uniform(0.5, 1.0, size=n)

    # Driving experience is strongly tied to age, with a little noise.
    driving_experience_years = np.clip(
        age - 17 - rng.normal(0, 2, size=n), 0, None
    )
    driving_experience_years = np.round(driving_experience_years, 1)

    # --- Claim frequency (Poisson) --------------------------------------
    gender_log_rate = np.where(gender == "F", -0.20, +0.15)
    territory_log_rate = np.select(
        [territory == "A", territory == "B", territory == "C"],
        [-0.25, 0.00, +0.30],
    )
    miles_effect = 0.10 * (annual_miles - 20_000.0) / 10_000.0
    experience_effect = -0.05 * (driving_experience_years - 20.0) / 10.0

    log_rate = (
        -2.30
        + gender_log_rate
        + territory_log_rate
        + _age_risk(age)
        + miles_effect
        + experience_effect
    )
    claim_rate = np.exp(log_rate) * exposure
    claim_count = rng.poisson(claim_rate).astype(int)

    # --- Claim severity (Gamma) -----------------------------------------
    severity_age_effect = 0.25 * (age - 45.0) / 15.0
    territory_sev_log = np.select(
        [territory == "A", territory == "B", territory == "C"],
        [-0.15, 0.00, +0.20],
    )
    vehicle_sev_log = np.select(
        [vehicle_type == "sedan", vehicle_type == "suv", vehicle_type == "truck"],
        [0.00, 0.10, 0.18],
    )
    mean_severity = 3_000.0 * np.exp(
        severity_age_effect + territory_sev_log + vehicle_sev_log
    )

    gamma_shape = 2.0
    total_claim_amount = np.zeros(n)
    for i in np.flatnonzero(claim_count > 0):
        severity = rng.gamma(
            gamma_shape, mean_severity[i] / gamma_shape, size=claim_count[i]
        )
        total_claim_amount[i] = severity.sum()

    return pd.DataFrame(
        {
            "policy_id": [f"P{i + 1:07d}" for i in range(n)],
            "age": age.astype(int),
            "gender": gender,
            "territory": territory,
            "vehicle_type": vehicle_type,
            "annual_miles": annual_miles.astype(int),
            "driving_experience_years": driving_experience_years,
            "exposure": exposure.round(4),
            "claim_count": claim_count,
            "total_claim_amount": total_claim_amount.round(2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-policies", type=int, default=100_000, help="Number of policies to generate."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/claims_full.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = generate_policies(args.n_policies, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} policies to {args.output}")
    print(f"Overall claim rate: {(df.claim_count > 0).mean():.4f}")
    print(f"Average loss per policy: ${df.total_claim_amount.mean():,.2f}")


if __name__ == "__main__":
    main()
