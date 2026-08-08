"""Smoke tests for the pricing baseline."""

import numpy as np
import pytest

from src.load import load_claims
from src.models import (
    fit_frequency_model,
    fit_severity_model,
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
