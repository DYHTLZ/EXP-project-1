# Model Card — Price of Fairness Pricing Models

Model cards make the intent, data, and limitations of a model explicit — the
kind of documentation regulators (NAIC, Colorado, EU AI Act) are starting to
require. This card covers the two production models in the project.

## Model details

| Field | Value |
|---|---|
| Developer | Yu Duan (project portfolio) |
| Model date | August 2026 |
| Model versions | v1 (linear age — superseded), v2 (age splines) |
| Model type | Generalized linear models: Poisson (frequency), Gamma (severity) |
| License | MIT |
| Feedback | GitHub issues on this repository |

## Intended use

* **Primary use:** demonstrating a complete actuarial fairness workflow —
  pricing, auditing, constraining, and communicating trade-offs — on
  synthetic data. The *methods* are production-grade; the *numbers* are
  illustrative.
* **Intended users:** actuaries, pricing analysts, risk managers, regulators,
  students.
* **Not intended for:** pricing real policies, making decisions about real
  individuals, or regulatory submission without adaptation to real data.

## Factors

* **Protected groups:** gender (F/M), territory (A/B/C), age bands.
* **Evaluation factors:** group calibration, demographic parity, equalized
  odds, premium shift, accuracy cost, redistribution.

## Metrics

Reported on the committed 10,000-policy sample (seed 42):

| Metric | Value |
|---|---:|
| Holdout loss ratio (predicted/actual) | 1.04 |
| Segment calibration range | 0.94–1.05 |
| Age-band calibration range (frequency, spline model) | 0.95–1.05 |
| Poisson dispersion | 1.02 |
| Demographic parity (F / M mean premium) | $430 / $547 |
| Equalized odds TPR (F / M) | 0.50 / 0.68 |
| Cost of full segment parity | +2.14% MSE |
| Premium volume moved at full parity | 35.3% |
| Cost of gender-only parity | +0.14% MSE |

## Evaluation and training data

Both use the same synthetic dataset: 10,000 policies committed in
`data/sample_claims.csv` (100,000-policy version reproducible with
`data/generate_data.py --n-policies 100000`, seed 42). An 80/20 split with
the same seed separates train/test; no real personal data is used.

## Quantitative analyses

* Accuracy-fairness frontier (`results/frontier_segment.png`): MSE cost vs
  parity gap, segment and gender variants.
* Redistribution (`results/redistribution_full_parity.png`): who pays more,
  who pays less.
* Calibration trade-off (`results/calibration_tradeoff.png`): parity breaks
  calibration by construction (Chouldechova).

## Ethical considerations

* **Synthetic data:** no real individuals are represented; conclusions do
  not describe any real population.
* **Protected attributes are predictors by design** so the fairness
  analysis has a signal to measure. In a real deployment this choice would
  itself be the subject of regulatory review.
* **Fairness is a choice:** the project shows that parity, equalized odds,
  and calibration cannot all hold when base rates differ. Any deployment
  must decide which definition applies and document the trade-off.

## Caveats and recommendations

* Replace synthetic data with real, validated policy data and re-run the
  entire pipeline before any business use.
* Extend the accuracy axis beyond MSE (loss ratio, deviance, Tweedie
  likelihood) and compare against ML models (GBM) as the actuarial
  literature now recommends.
* Add per-policy constraints and individual-fairness checks for production
  use; group-level metrics alone are insufficient.
