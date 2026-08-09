# The Price of Fairness: Constrained Risk Pricing

What does fairness actually cost in risk pricing? This project builds a risk-pricing pipeline (frequency and severity GLMs), a fairness audit layer, constrained pricing variants, and an accuracy-fairness trade-off frontier, then explains who pays the price of equality.

**Status:** complete — pricing baseline, fairness audit, constrained
pricing frontier, redistribution analysis, dashboard, and memos.

## Repository structure

- `data/` — synthetic policy/claim data generator + sample output
- `notebooks/` — EDA, baseline pricing, and fairness analysis
- `src/` — pricing pipeline, fairness metrics, constrained pricing, frontier
- `dashboard/` — Streamlit explorer
- `docs/` — methodology and fairness memos
- `tests/` — unit tests for metrics and pricing math

## How to run

Requires Python 3.10+.

1. Create a virtual environment:
   `python -m venv .venv`
2. Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Regenerate the full dataset (100K policies):
   `python data/generate_data.py --n-policies 100000 --output data/claims_full.csv`

   A 10K-row sample ships as `data/sample_claims.csv`, so you can skip this
   step to stay light.
5. Explore the data: `jupyter notebook notebooks/01_eda.ipynb`
6. Pricing baseline: `jupyter notebook notebooks/02_baseline.ipynb`
7. Fairness audit: `jupyter notebook notebooks/03_fairness.ipynb`
8. Constrained pricing and the frontier: `jupyter notebook notebooks/04_constrained.ipynb`
9. Interactive dashboard: `streamlit run dashboard/app.py`
10. Run the tests: `pytest`

## Key findings

* **The baseline is accurate but unfair.** The pure-premium GLM calibrates to
  within ~3% per segment and a 1.06 holdout loss ratio, yet mean premiums
  span from $272 (F/A) to $902 (M/C) — a 3.3× spread driven by base-rate
  differences built into the data.
* **Fairness metrics conflict by mathematics, not by bug.** Calibration
  holds while demographic parity and equalized odds fail; the thresholds
  needed for TPR parity differ across groups (Chouldechova's impossibility).
* **Full demographic parity is expensive in design space, not dollars:**
  equalizing all six segment means costs +2.0% MSE and moves 35% of premium
  volume — a massive cross-subsidy from low-risk to high-risk segments.
* **A single-variable ban is cheap.** Gender-only parity (the EU-style rule)
  costs just +0.18% MSE and moves 12% of premium volume, because territory
  still carries the pricing signal.
* **Parity breaks calibration on purpose.** At full parity the premium/actual
  ratio swings from 0.54 (M/C, undercharged) to 1.83 (F/A, overcharged).

## Results

Precomputed frontier, redistribution, and calibration tables live in
`results/` (also rendered by the dashboard). See
[`docs/methodology.md`](docs/methodology.md) for the full derivation and
[`docs/fairness_memo.md`](docs/fairness_memo.md) for the fairness write-up.

Public-data applications of this pipeline are in
[`test_public_examples/`](test_public_examples/README.md) (marked TEST).
