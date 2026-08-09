# ⚠️ TEST — Public-data applications of the Project 1 pipeline

> **Status: experimental.** These notebooks apply the Project 1 methods
> (pricing GLMs, fairness audit, calibration checks) to real public datasets
> to demonstrate that the pipeline transfers. They are **not** part of the
> main project analysis and use different data/model choices than the
> synthetic flagship pipeline.

## Datasets

| Example | Dataset | Task | Protected attributes |
|---|---|---|---|
| 1 | UCI Adult (income) | Predict income > 50K with a logistic GLM, then audit fairness | sex, race |
| 2 | freMTPL2 (French motor TPL) | Frequency + severity GLMs → pure premium; audit by age band | age band (proxy) |

Sources:

* UCI Adult: https://archive.ics.uci.edu/ml/datasets/adult (publicly donated benchmark)
* freMTPL2: public academic dataset from the CASdatasets R package; CSV mirror
  hosted at https://huggingface.co/datasets/mabilton/fremtpl2 (originally the
  French Motor Third-Party Liability claims data by Charpentier).

## How to run

1. Download the data (full freMTPL2 file ~36 MB goes to `data/raw/`; a 100K
   sample is committed in `data/`):
   `python data/download_data.py`
2. `jupyter notebook notebooks/01_adult_fairness_test.ipynb`
3. `jupyter notebook notebooks/02_fremtpl2_pricing_test.ipynb`

## Results

Both notebooks execute end-to-end with zero errors; full tables and charts
are in `results/`.

### Example 1 — UCI Adult (fairness audit, 32,561 rows)

* **Calibration holds within every group** (predicted/actual ≈ 1.00 for sex
  and race) — the GLM includes group indicators, so group means are exact.
* **Demographic parity fails.** Mean predicted P(income > 50K):

| Sex | Mean predicted | Ratio vs overall |
|---|---|---|
| Female | 0.110 | 0.45× |
| Male | 0.306 | 1.27× |

  By race: White 0.256, Asian-Pac-Islander 0.266, Black 0.124, Other 0.092.
* **Equalized odds and the impossibility demo** reproduce the Project 1
  pattern: base rates differ, calibration holds, parity metrics fail, and
  the thresholds needed for TPR parity differ by group.

### Example 2 — freMTPL2 (pricing pipeline, 100K-policy sample)

* Poisson frequency GLM (exposure offset) + Gamma severity GLM → pure premium.
* **Age-band audit** (protected proxy):

| Age band | Predicted premium (EUR) | Ratio vs overall | Calibration (pred/actual) |
|---|---|---|---|
| <25 | 126.5 | 1.65× | 0.66 |
| 25–40 | 82.7 | 1.08× | 1.09 |
| 40–60 | 69.0 | 0.90× | 0.94 |
| 60+ | 72.0 | 0.94× | 1.02 |

* **Modeling lesson:** the naive linear-age model *undercharges young
  drivers* (predicted €126 vs actual €192 average loss) — the U-shaped
  young-driver risk needs age bands or splines. Exactly the kind of finding
  the flagship project is built to surface.

### Files

* `notebooks/01_adult_fairness_test.ipynb`, `notebooks/02_fremtpl2_pricing_test.ipynb`
* `results/*.csv` — all audit and calibration tables
* `results/*.png` — charts
* `data/` — committed datasets (`adult.data`, `freMTPL2freq_sample.csv`,
  `freMTPL2sev.csv`); full freMTPL2 file via `data/download_data.py`
