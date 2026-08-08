# The Price of Fairness: Constrained Risk Pricing

What does fairness actually cost in risk pricing? This project builds a risk-pricing pipeline (frequency and severity GLMs), a fairness audit layer, constrained pricing variants, and an accuracy-fairness trade-off frontier, then explains who pays the price of equality.

**Status:** Weeks 2–4 — pricing baseline (frequency/severity GLMs) done.

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
6. Walk through the pricing baseline: `jupyter notebook notebooks/02_baseline.ipynb`
7. Run the tests: `pytest`
