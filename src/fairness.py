"""Fairness audit layer for the pricing baseline.

Measures four families of fairness metrics across protected groups
(default: gender x territory) using baseline model predictions:

* **Base rates** — does true risk differ across groups? (The precondition
  for the whole analysis.)
* **Demographic parity** — are average scores/premiums equal across groups?
* **Equalized odds** — are true-positive and false-positive rates equal
  across groups at a common threshold?
* **Calibration parity** — does the same predicted score mean the same
  risk in every group?
* **Premium shift** — who pays more, and by how much?

These definitions are mutually incompatible when base rates differ
(Chouldechova), which is the mathematical heart of the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import predict_frequency, predict_pure_premium


def add_predictions(
    df: pd.DataFrame, freq_model, sev_model
) -> pd.DataFrame:
    """Attach baseline predictions and the claim outcome to a DataFrame."""
    out = df.copy()
    out["predicted_frequency"] = predict_frequency(freq_model, out)
    out["predicted_premium"] = predict_pure_premium(freq_model, sev_model, out)
    out["has_claim"] = out["claim_count"] > 0
    return out


def base_rates(
    df: pd.DataFrame, group_cols: tuple[str, ...] = ("gender", "territory")
) -> pd.DataFrame:
    """True claim rates and severities by group."""
    tab = df.groupby(list(group_cols), observed=True).agg(
        n_policies=("policy_id", "count"),
        claim_rate=("claim_count", lambda s: (s > 0).mean()),
        claim_frequency=("claim_count", "mean"),
    )
    sev = (
        df.loc[df["claim_count"] > 0]
        .groupby(list(group_cols), observed=True)["total_claim_amount"]
        .mean()
        .rename("avg_severity_given_claim")
    )
    return tab.join(sev).round(4)


def demographic_parity(
    df: pd.DataFrame, score_col: str, group_cols: tuple[str, ...] = ("gender", "territory")
) -> pd.DataFrame:
    """Mean score/premium per group vs the overall mean."""
    overall = df[score_col].mean()
    tab = df.groupby(list(group_cols), observed=True).agg(
        n_policies=("policy_id", "count"),
        mean_score=(score_col, "mean"),
    )
    tab["ratio_vs_overall"] = tab["mean_score"] / overall
    tab["diff_vs_overall"] = tab["mean_score"] - overall
    return tab.round(4)


def equalized_odds(
    df: pd.DataFrame,
    score_col: str,
    outcome_col: str,
    group_cols: tuple[str, ...] = ("gender", "territory"),
    threshold: float | None = None,
) -> pd.DataFrame:
    """TPR/FPR per group at a common score threshold."""
    if threshold is None:
        threshold = df[score_col].mean()
    work = pd.DataFrame(
        {
            **{g: df[g].values for g in group_cols},
            "pred": (df[score_col] > threshold).astype(int).values,
            "y": df[outcome_col].astype(int).values,
        }
    )

    def metrics(g: pd.DataFrame) -> pd.Series:
        tp = ((g["pred"] == 1) & (g["y"] == 1)).sum()
        fn = ((g["pred"] == 0) & (g["y"] == 1)).sum()
        fp = ((g["pred"] == 1) & (g["y"] == 0)).sum()
        tn = ((g["pred"] == 0) & (g["y"] == 0)).sum()
        return pd.Series(
            {
                "n": len(g),
                "tpr": tp / (tp + fn) if tp + fn else np.nan,
                "fpr": fp / (fp + tn) if fp + tn else np.nan,
                "predicted_positive_rate": g["pred"].mean(),
            }
        )

    return work.groupby(list(group_cols), observed=True).apply(
        metrics, include_groups=False
    ).round(4)


def calibration_by_group(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    group_cols: tuple[str, ...] = ("gender", "territory"),
) -> pd.DataFrame:
    """Mean predicted vs mean actual by group."""
    tab = df.groupby(list(group_cols), observed=True).agg(
        n_policies=("policy_id", "count"),
        actual_mean=(actual_col, "mean"),
        predicted_mean=(predicted_col, "mean"),
    )
    tab["predicted/actual"] = tab["predicted_mean"] / tab["actual_mean"]
    return tab.round(4)


def premium_shift(
    df: pd.DataFrame, premium_col: str, group_cols: tuple[str, ...] = ("gender", "territory")
) -> pd.DataFrame:
    """Average premium per group, expressed relative to the cheapest group."""
    tab = (
        df.groupby(list(group_cols), observed=True)[premium_col]
        .mean()
        .sort_values()
        .rename("mean_premium")
        .to_frame()
    )
    tab["premium_ratio_vs_cheapest"] = tab["mean_premium"] / tab["mean_premium"].min()
    tab["premium_gap_vs_cheapest"] = tab["mean_premium"] - tab["mean_premium"].min()
    return tab.round(2)


def threshold_for_tpr(
    df: pd.DataFrame,
    score_col: str,
    outcome_col: str,
    group_cols: tuple[str, ...] = ("gender", "territory"),
    target_tpr: float = 0.5,
) -> pd.DataFrame:
    """Score threshold each group would need to hit a target TPR.

    Used to demonstrate the Chouldechova impossibility: if the thresholds
    differ across groups, the same score means different risk — calibration
    and equalized odds cannot both hold when base rates differ.
    """
    rows = {}
    for keys, g in df.groupby(list(group_cols), observed=True):
        positive_scores = g.loc[g[outcome_col] == 1, score_col]
        threshold = float(np.quantile(positive_scores, 1 - target_tpr))
        rows[keys] = {"needed_threshold": threshold, "base_rate": g[outcome_col].mean()}
    tab = pd.DataFrame(rows).T
    tab.index.names = list(group_cols)
    return tab.round(4)


def individual_fairness(
    df: pd.DataFrame,
    score_col: str,
    premium_col: str,
    group_cols: tuple[str, ...] = ("gender",),
    n_bins: int = 10,
) -> pd.DataFrame:
    """Within-risk-bin premium spread across protected groups.

    Bins policies by predicted risk (``score_col``) and compares mean
    premiums across groups inside each bin. If premiums differ within the
    same risk bin, non-risk factors (e.g., protected attributes) are
    pricing — the individual-fairness lens: *similar risk, similar price*.
    """
    out = df.copy()
    out["risk_bin"] = pd.qcut(df[score_col], q=n_bins, duplicates="drop")
    tab = (
        out.groupby(["risk_bin", *group_cols], observed=True)[premium_col]
        .mean()
        .unstack()
    )
    spread = tab.max(axis=1) / tab.min(axis=1)
    result = pd.DataFrame(
        {
            "risk_bin": tab.index.astype(str),
            "min_premium": tab.min(axis=1),
            "max_premium": tab.max(axis=1),
            "max_min_ratio": spread,
        }
    )
    return result.round(2)
