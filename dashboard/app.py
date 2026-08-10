"""Interactive explorer for the price-of-fairness analysis.

Run from the repository root:

    streamlit run dashboard/app.py

The app reads the precomputed frontier and redistribution tables in
``results/`` (produced by ``notebooks/04_constrained.ipynb``), so it opens
instantly without refitting models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="The Price of Fairness", layout="wide")

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_or_error(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        st.error(f"Missing results/{name} — run notebooks/04_constrained.ipynb first.")
        st.stop()
    return pd.read_csv(path)


frontier_seg = load_or_error("frontier_segment.csv")
frontier_gen = load_or_error("frontier_gender.csv")
seg_summary = load_or_error("segment_summary.csv")
red_full = load_or_error("redistribution_full_parity.csv")
red_gender = load_or_error("redistribution_gender_parity.csv")
calibration = load_or_error("calibration_tradeoff.csv")

st.title("The Price of Fairness")
st.caption(
    "Constrained risk pricing and the accuracy-fairness frontier — "
    "synthetic policy data (10,000 policies), baseline GLM premiums."
)

# ---------------------------------------------------------------------------
# Sidebar: choose constraint and frontier point
# ---------------------------------------------------------------------------

mode = st.sidebar.radio(
    "Parity constraint",
    ["Full segment parity", "Gender parity only"],
    help="Full parity equalizes all six gender x territory means; gender "
    "parity equalizes only the two gender means (EU-style ban).",
)
frontier = frontier_seg if mode.startswith("Full") else frontier_gen
redistribution = red_full if mode.startswith("Full") else red_gender

eps_values = np.round(frontier["eps"].to_numpy(), 4)
position = st.sidebar.select_slider(
    "Allowed parity gap ε",
    options=list(range(len(eps_values))),
    value=len(eps_values) - 1,
    format_func=lambda i: f"ε = {eps_values[i]:.3f}",
)
row = frontier.iloc[position]

st.sidebar.divider()
st.sidebar.metric("Accuracy cost (extra MSE)", f"{row['cost_mse'] * 100:.2f}%")
st.sidebar.metric("Fairness gap", f"{row['fairness_gap']:.2%}")
st.sidebar.metric("Premium volume moved", f"{row['redistribution']:.1%}")
st.sidebar.metric("Aggregate loss ratio", f"{row['loss_ratio']:.3f}")
st.sidebar.metric("Loss-ratio gap", f"{row['loss_ratio_gap']:.2f}")

# ---------------------------------------------------------------------------
# Segment premiums at the selected frontier point
# ---------------------------------------------------------------------------

premium_rows = []
for i in range(len(seg_summary)):
    seg_name = seg_summary["segment"].iloc[i]
    baseline = seg_summary["baseline_premium"].iloc[i]
    constrained = row[f"multiplier_{i}"] * baseline
    premium_rows.append(
        {
            "segment": seg_name,
            "baseline_premium": baseline,
            "constrained_premium": constrained,
            "change_pct": (constrained / baseline - 1) * 100,
        }
    )
premiums = pd.DataFrame(premium_rows)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_frontier, tab_premiums, tab_redistribution, tab_calibration, tab_effects = st.tabs(
    ["Overview", "Frontier", "Premiums", "Redistribution", "Calibration", "Effect sizes"]
)

with tab_overview:
    c1, c2, c3 = st.columns(3)
    baseline_gap = frontier_seg["fairness_gap"].iloc[0]
    c1.metric("Baseline premium spread", f"{baseline_gap:.0%}",
              help="Relative range of segment mean premiums before constraints.")
    c2.metric("Cost of full parity", f"{frontier_seg['cost_mse'].iloc[-1] * 100:.2f}%",
              help="Extra MSE when all six segment means are equal.")
    c3.metric("Cost of gender parity", f"{frontier_gen['cost_mse'].iloc[-1] * 100:.2f}%",
              help="Extra MSE when only gender means are equal (EU-style ban).")

    st.subheader("Segment summary (baseline vs endpoints)")
    st.dataframe(
        seg_summary.style.format(
            {"baseline_premium": "${:,.0f}", "full_parity_premium": "${:,.0f}",
             "gender_parity_premium": "${:,.0f}"}
        ),
        use_container_width=True,
    )

with tab_frontier:
    st.subheader("Accuracy-fairness frontier")
    fig = go.Figure()
    for label, data, color in [
        ("Full segment parity", frontier_seg, "#4C72B0"),
        ("Gender parity only", frontier_gen, "#55A868"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=data["fairness_gap"],
                y=data["cost_mse"] * 100,
                mode="lines+markers",
                name=label,
                line=dict(color=color),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[row["fairness_gap"]],
            y=[row["cost_mse"] * 100],
            mode="markers",
            name="Selected point",
            marker=dict(size=14, color="#C44E52", symbol="star"),
        )
    )
    fig.update_layout(
        xaxis_title="Fairness gap (relative premium range)",
        yaxis_title="Accuracy cost (extra MSE, %)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_premiums:
    st.subheader(f"Segment premiums at ε = {eps_values[position]:.3f}")
    fig = px.bar(
        premiums.melt(id_vars=["segment"], value_vars=["baseline_premium", "constrained_premium"]),
        x="segment",
        y="value",
        color="variable",
        barmode="group",
        labels={"value": "Premium ($)", "variable": ""},
        color_discrete_map={
            "baseline_premium": "#4C72B0",
            "constrained_premium": "#C44E52",
        },
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(premiums.style.format({"baseline_premium": "${:,.2f}",
                                        "constrained_premium": "${:,.2f}",
                                        "change_pct": "{:+.1f}%"}), use_container_width=True)

with tab_redistribution:
    st.subheader("Who pays more, who pays less")
    fig = px.bar(
        redistribution.sort_values("change_pct"),
        x="change_pct",
        y="segment",
        orientation="h",
        color=redistribution.sort_values("change_pct")["is_payer"].map(
            {True: "pays less", False: "pays more"}
        ),
        color_discrete_map={"pays more": "#55A868", "pays less": "#C44E52"},
        labels={"change_pct": "Premium change (%)", "segment": ""},
    )
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        redistribution.style.format(
            {"baseline_premium": "${:,.2f}", "constrained_premium": "${:,.2f}",
             "change": "${:+,.2f}", "change_pct": "{:+.1f}%"}
        ),
        use_container_width=True,
    )

with tab_calibration:
    st.subheader("Calibration vs parity: the impossibility, in dollars")
    st.write(
        "The baseline is calibrated (ratio ≈ 1 in every segment). Imposing "
        "parity forces ratios away from 1 — high-risk segments get "
        "undercharged, low-risk overcharged. Parity and calibration cannot "
        "both hold when base rates differ (Chouldechova)."
    )
    fig = go.Figure()
    x = calibration["segment"].astype(str)
    fig.add_trace(
        go.Bar(x=x, y=calibration["baseline_ratio"], name="Baseline (calibrated)",
               marker_color="#4C72B0")
    )
    fig.add_trace(
        go.Bar(x=x, y=calibration["parity_ratio"], name="Full parity",
               marker_color="#C44E52")
    )
    fig.add_hline(y=1, line_dash="dash", line_color="black")
    fig.update_layout(height=420, yaxis_title="Premium / actual loss ratio")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(calibration, use_container_width=True)

with tab_effects:
    st.subheader("Interpretability: effect sizes (exp(coef) with 95% CI)")
    st.write(
        "Each value is the multiplicative effect of a feature on claim "
        "frequency or severity per claim. Values above 1 increase risk, "
        "below 1 decrease it. Generated by notebooks/02_baseline.ipynb."
    )
    effect_path = RESULTS / "effect_sizes.png"
    if effect_path.exists():
        st.image(str(effect_path), use_container_width=True)
    else:
        st.info("Run notebooks/02_baseline.ipynb to generate results/effect_sizes.png.")

st.caption(
    "Data: synthetic policy/claim generator (seed 42, 10K sample). "
    "Models: Poisson frequency GLM + Gamma severity GLM. "
    "Frontier: convex QP (OSQP) minimizing MSE subject to mean-premium parity."
)
