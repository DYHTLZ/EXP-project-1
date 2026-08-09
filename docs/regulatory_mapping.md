# Regulatory mapping — where this project meets current regulation

The project was built to mirror what pricing teams are actually being asked
to produce. This page maps each regulatory trend (as of August 2026) to the
artifact that addresses it.

## EU AI Act

**What it does:** AI systems used for risk assessment and pricing in life
and health insurance are classified as **high-risk**, bringing requirements
for transparency, documentation, and non-discrimination testing. Some
high-risk obligations are phased in over 2026–2027.

**What the project provides:**

- Model documentation: `docs/model_card.md`, `docs/methodology.md`
- Non-discrimination testing: `src/fairness.py` (demographic parity,
  equalized odds, calibration), `notebooks/03_fairness.ipynb`
- Trade-off quantification: `src/frontier.py`, `notebooks/04_constrained.ipynb`

## Colorado SB 21-169 and Regulation 10-1-1

**What it does:** prohibits insurers from using external consumer data,
algorithms, or predictive models in a way that unfairly discriminates based
on race, color, national or ethnic origin, religion, sex, sexual
orientation, disability, or other protected classes. Auto and health
carriers began submitting **annual compliance reports on July 1, 2026**,
with documented testing of models for unfair discrimination.

**What the project provides:**

- Exactly the testing regime: base rates → parity → equalized odds →
  calibration → constrained pricing (the "what if we remove this variable"
  analysis regulators ask for)
- Reproducible evidence: tests + CI + committed results
- A model card for the documentation exhibit

## NAIC Model Bulletin on the Use of AI by Insurers

**What it does:** principles-based guidance (adopted December 2023)
requiring insurers to establish governance, documentation, and audit
procedures for AI systems — including verification and testing for errors,
bias, and unfair discrimination. The NAIC's Third-Party Data and Models (H)
Working Group has discussed standardized reporting tools, including
**model cards**.

**What the project provides:**

- `docs/model_card.md` — exactly the nutrition-label style documentation
  under discussion
- Governance-ready structure: versioned models, regression tests, CI that
  re-runs tests and notebooks on every push

## Academic and industry trends

* **GLM vs ML fairness** (British Actuarial Journal, 2026): recent research
  evaluates ML pricing models (GBM, Tweedie) on performance *and* fairness
  simultaneously — the same dual evaluation this project implements for
  GLMs. Extension path: add a GBM/Tweedie comparison to the frontier.
* **Causal fairness audit frameworks** are emerging for insurance
  (actuarial conference work, 2026): moving from correlation-based metrics
  to causal definitions. Extension path: add a causal DAG and mediation
  analysis.
* **Individual vs group fairness** is now a standard distinction in the
  literature. This project implements group fairness; individual fairness
  (similar policies → similar premiums) is a documented extension.

## Suggested production extension

1. Replace synthetic data with real policy data.
2. Add a Tweedie/GBM model comparison to the fairness audit.
3. Add SHAP-style explanations for the ML models.
4. Add per-policy constraints and individual-fairness metrics.
5. Generate the compliance report artifacts (model inventory, testing
   results) directly from the pipeline.
