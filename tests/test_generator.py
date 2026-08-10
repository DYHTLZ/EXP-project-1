"""Audit tests for the synthetic data generator."""

import numpy as np

from data.generate_data import generate_policies


def test_generator_is_deterministic():
    df1 = generate_policies(20_000, np.random.default_rng(42))
    df2 = generate_policies(20_000, np.random.default_rng(42))
    assert df1.equals(df2)


def test_generator_matches_theoretical_frequency():
    rng = np.random.default_rng(42)
    df = generate_policies(50_000, rng)

    age = df["age"].to_numpy(dtype=float)
    gender = df["gender"].to_numpy()
    territory = df["territory"].to_numpy()
    miles = df["annual_miles"].to_numpy(dtype=float)
    exp_years = df["driving_experience_years"].to_numpy(dtype=float)
    exposure = df["exposure"].to_numpy(dtype=float)

    gender_eff = np.where(gender == "F", -0.20, 0.15)
    terr_eff = np.select(
        [territory == "A", territory == "B", territory == "C"],
        [-0.25, 0.00, 0.30],
    )
    age_risk = 0.35 * np.abs((age - 45.0) / 10.0)
    miles_eff = 0.10 * (miles - 20_000.0) / 10_000.0
    exp_eff = -0.05 * (exp_years - 20.0) / 10.0
    theoretical = np.exp(
        -2.30 + gender_eff + terr_eff + age_risk + miles_eff + exp_eff
    ) * exposure

    rel_error = abs(df["claim_count"].mean() - theoretical.mean()) / theoretical.mean()
    assert rel_error < 0.02, f"frequency rel error {rel_error:.4f}"


def test_generator_matches_theoretical_severity():
    rng = np.random.default_rng(42)
    df = generate_policies(50_000, rng)

    age = df["age"].to_numpy(dtype=float)
    territory = df["territory"].to_numpy()
    vehicle = df["vehicle_type"].to_numpy()
    exposure = df["exposure"].to_numpy(dtype=float)

    gender = df["gender"].to_numpy()
    miles = df["annual_miles"].to_numpy(dtype=float)
    exp_years = df["driving_experience_years"].to_numpy(dtype=float)
    gender_eff = np.where(gender == "F", -0.20, 0.15)
    terr_eff = np.select(
        [territory == "A", territory == "B", territory == "C"],
        [-0.25, 0.00, 0.30],
    )
    age_risk = 0.35 * np.abs((age - 45.0) / 10.0)
    miles_eff = 0.10 * (miles - 20_000.0) / 10_000.0
    exp_eff = -0.05 * (exp_years - 20.0) / 10.0
    claim_rate = np.exp(
        -2.30 + gender_eff + terr_eff + age_risk + miles_eff + exp_eff
    ) * exposure

    sev_age = 0.25 * (age - 45.0) / 15.0
    terr_sev = np.select(
        [territory == "A", territory == "B", territory == "C"],
        [-0.15, 0.00, 0.20],
    )
    veh_sev = np.select(
        [vehicle == "sedan", vehicle == "suv", vehicle == "truck"],
        [0.00, 0.10, 0.18],
    )
    mean_severity = 3_000.0 * np.exp(sev_age + terr_sev + veh_sev)
    theoretical_loss = (claim_rate * mean_severity).mean()

    rel_error = abs(df["total_claim_amount"].mean() - theoretical_loss) / theoretical_loss
    assert rel_error < 0.05, f"severity rel error {rel_error:.4f}"
