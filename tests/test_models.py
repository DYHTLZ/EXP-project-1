"""Smoke tests for the pricing baseline."""

import numpy as np
import pandas as pd
import pytest

from src.load import load_claims
from src.models import (
    FREQUENCY_FORMULA_NO_GENDER,
    fit_frequency_model,
    fit_severity_model,
    predict_frequency,
    predict_pure_premium,
    pure_premium_table,
)


@pytest.fixture(scope="module")
def data():
    return load_claims()


def test_frequency_and_severity_models_fit(data):
    freq = fit_frequency_model(data)
    sev = fit_severity_model(data)
    assert np.isfinite(freq.params).all()
    assert np.isfinite(sev.params).all()


def test_pure_premium_matches_empirical_within_10_percent(data):
    freq = fit_frequency_model(data)
    sev = fit_severity_model(data)
    tab = pure_premium_table(data, freq, sev)
    ratio = tab["predicted/empirical"]
    assert ratio.between(0.90, 1.10).all()


def test_predictions_are_positive(data):
    freq = fit_frequency_model(data)
    sev = fit_severity_model(data)
    pred = predict_pure_premium(freq, sev, data)
    assert (pred > 0).all()


def test_frequency_model_calibrated_by_age_band(data):
    """Age splines must fix the linear-age under/over-pricing (regression test)."""
    freq = fit_frequency_model(data)
    out = data.copy()
    out["pred"] = predict_frequency(freq, out)
    out["age_band"] = pd.cut(out["age"], bins=[18, 25, 35, 45, 55, 65, 80], right=True)
    cal = out.groupby("age_band", observed=True).agg(
        actual=("claim_count", "mean"), predicted=("pred", "mean")
    )
    ratio = cal["predicted"] / cal["actual"]
    assert ratio.between(0.90, 1.10).all(), ratio


def test_no_gender_model_improves_demographic_parity(data):
    from src.fairness import demographic_parity

    sev = fit_severity_model(data)
    full = fit_frequency_model(data)
    no_gender = fit_frequency_model(data, formula=FREQUENCY_FORMULA_NO_GENDER)

    scored_full = data.copy()
    scored_full["predicted_premium"] = predict_pure_premium(full, sev, scored_full)
    scored_nog = data.copy()
    scored_nog["predicted_premium"] = predict_pure_premium(no_gender, sev, scored_nog)

    dp_full = demographic_parity(scored_full, "predicted_premium", ("gender",))
    dp_nog = demographic_parity(scored_nog, "predicted_premium", ("gender",))
    dev_full = abs(dp_full.loc["F", "ratio_vs_overall"] - 1.0)
    dev_nog = abs(dp_nog.loc["F", "ratio_vs_overall"] - 1.0)
    assert dev_nog < dev_full
