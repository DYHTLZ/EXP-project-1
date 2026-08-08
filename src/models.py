"""Pricing baseline: frequency and severity GLMs -> pure premiums.

The classic actuarial decomposition:

    pure premium = expected claim frequency * expected claim severity

* Frequency model: Poisson GLM (log link) with log(exposure) as an offset.
* Severity model: Gamma GLM (log link) on policies with at least one claim.

Everything here uses `statsmodels`, so the summaries, p-values and
diagnostics are available out of the box for the methodology memo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FREQUENCY_FORMULA = (
    "claim_count ~ age + C(gender) + C(territory) "
    "+ annual_miles + driving_experience_years"
)
SEVERITY_FORMULA = (
    "severity_per_claim ~ age + C(territory) + C(vehicle_type)"
)


def fit_frequency_model(
    df: pd.DataFrame, exposure_col: str = "exposure"
):
    """Fit a Poisson GLM for claim counts with a log-exposure offset."""
    offset = np.log(df[exposure_col].clip(lower=1e-12))
    model = smf.glm(
        FREQUENCY_FORMULA,
        data=df,
        family=sm.families.Poisson(),
        offset=offset,
    )
    return model.fit()


def predict_frequency(model, df: pd.DataFrame, exposure_col: str = "exposure") -> pd.Series:
    """Predicted claim counts, re-applying the log-exposure offset."""
    offset = np.log(df[exposure_col].clip(lower=1e-12))
    return model.predict(df, offset=offset)


def fit_severity_model(df: pd.DataFrame):
    """Fit a Gamma GLM (log link) for severity per claim.

    The response is total loss divided by claim count, which keeps the
    decomposition exact: pure premium = frequency * severity per claim.
    """
    claimers = df.loc[df["claim_count"] > 0].copy()
    claimers["severity_per_claim"] = (
        claimers["total_claim_amount"] / claimers["claim_count"]
    )
    model = smf.glm(
        SEVERITY_FORMULA,
        data=claimers,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    )
    return model.fit()


def predict_pure_premium(freq_model, sev_model, df: pd.DataFrame) -> pd.Series:
    """Predicted pure premium = predicted frequency * predicted severity."""
    return predict_frequency(freq_model, df) * sev_model.predict(df)


def pure_premium_table(
    df: pd.DataFrame,
    freq_model,
    sev_model,
    group_cols: tuple[str, ...] = ("gender", "territory"),
) -> pd.DataFrame:
    """Empirical vs modeled pure premium by segment."""
    out = df.copy()
    out["predicted_premium"] = predict_pure_premium(freq_model, sev_model, out)
    tab = out.groupby(list(group_cols), observed=True).agg(
        n_policies=("policy_id", "count"),
        empirical_premium=("total_claim_amount", "mean"),
        predicted_premium=("predicted_premium", "mean"),
    )
    tab["predicted/empirical"] = tab["predicted_premium"] / tab["empirical_premium"]
    return tab.round(2)
