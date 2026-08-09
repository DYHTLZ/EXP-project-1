"""Tests for constrained pricing and the fairness frontier."""

import numpy as np
import pytest

from src.fairness import add_predictions
from src.frontier import ConstrainedPricing
from src.load import load_claims
from src.models import fit_frequency_model, fit_severity_model


@pytest.fixture(scope="module")
def scored():
    df = load_claims()
    freq = fit_frequency_model(df)
    sev = fit_severity_model(df)
    return add_predictions(df, freq, sev)


def test_full_parity_achieves_zero_gap(scored):
    cp = ConstrainedPricing(scored)
    lam = cp.solve(eps=0.0)
    metrics = cp.metrics(lam)
    assert metrics["fairness_gap"] < 1e-6


def test_budget_neutrality(scored):
    cp = ConstrainedPricing(scored)
    lam = cp.solve(eps=0.05)
    pred = scored["predicted_premium"].to_numpy() * lam[cp.seg]
    assert abs(pred.sum() - scored["predicted_premium"].sum()) < 1e-4


def test_parity_constraint_holds(scored):
    cp = ConstrainedPricing(scored)
    eps = 0.05
    lam = cp.solve(eps=eps)
    mu = cp.overall_mean
    for g in range(cp.n_parity):
        mean_g = (scored["predicted_premium"].to_numpy() * lam[cp.seg])[
            cp.parity == g
        ].mean()
        assert mean_g <= mu * (1 + eps) + 1e-6
        assert mean_g >= mu * (1 - eps) - 1e-6


def test_frontier_cost_is_monotone(scored):
    cp = ConstrainedPricing(scored)
    table = cp.frontier(n_points=8)
    assert table["cost_mse"].is_monotonic_increasing
    assert table["fairness_gap"].iloc[-1] < 1e-6
    assert table["fairness_gap"].iloc[0] > table["fairness_gap"].iloc[-1]


def test_redistribution_table_consistent(scored):
    cp = ConstrainedPricing(scored)
    lam = cp.solve(eps=0.0)
    tab = cp.redistribution_table(lam)
    assert "change" in tab.columns
    # zero-sum: total money moved across segments is 0 (budget neutral)
    total_premium = scored["predicted_premium"].sum()
    assert abs(tab["total_moved"].sum() / total_premium) < 1e-6
    assert (tab["constrained_premium"] > 0).all()
