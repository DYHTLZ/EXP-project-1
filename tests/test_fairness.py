"""Smoke tests for the fairness audit layer."""

import numpy as np

from src.fairness import (
    add_predictions,
    base_rates,
    calibration_by_group,
    demographic_parity,
    equalized_odds,
    individual_fairness,
    premium_shift,
)
from src.load import load_claims
from src.models import fit_frequency_model, fit_severity_model


def test_fairness_metrics_run_on_real_data():
    df = load_claims()
    freq = fit_frequency_model(df)
    sev = fit_severity_model(df)
    scored = add_predictions(df, freq, sev)

    assert base_rates(df).shape[0] == 6
    dp = demographic_parity(scored, "predicted_premium")
    assert "ratio_vs_overall" in dp.columns
    eo = equalized_odds(scored, "predicted_frequency", "has_claim")
    assert {"tpr", "fpr"}.issubset(eo.columns)
    cal = calibration_by_group(scored, "total_claim_amount", "predicted_premium")
    assert "predicted/actual" in cal.columns
    ps = premium_shift(scored, "predicted_premium")
    assert ps["premium_ratio_vs_cheapest"].iloc[0] == 1.0


def test_base_rates_differ_by_gender():
    df = load_claims()
    tab = base_rates(df, group_cols=("gender",))
    assert tab.loc["M", "claim_rate"] > tab.loc["F", "claim_rate"]


def test_calibration_holds_within_10_percent():
    df = load_claims()
    freq = fit_frequency_model(df)
    sev = fit_severity_model(df)
    scored = add_predictions(df, freq, sev)
    cal = calibration_by_group(scored, "total_claim_amount", "predicted_premium")
    assert cal["predicted/actual"].between(0.90, 1.10).all()


def test_equalized_odds_conflict_is_visible():
    """With differing base rates and a common threshold, TPRs must differ."""
    df = load_claims()
    freq = fit_frequency_model(df)
    sev = fit_severity_model(df)
    scored = add_predictions(df, freq, sev)
    eo = equalized_odds(scored, "predicted_frequency", "has_claim", ("gender",))
    assert not np.isclose(eo.loc["F", "tpr"], eo.loc["M", "tpr"], atol=0.02)


def test_individual_fairness_spread_exists():
    """Within the same risk bin, protected groups still pay differently."""
    df = load_claims()
    freq = fit_frequency_model(df)
    sev = fit_severity_model(df)
    scored = add_predictions(df, freq, sev)
    table = individual_fairness(scored, "predicted_frequency", "predicted_premium")
    assert {"risk_bin", "min_premium", "max_premium", "max_min_ratio"}.issubset(
        table.columns
    )
    assert (table["max_min_ratio"] > 1.0).all()
