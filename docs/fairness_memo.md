# Fairness memo — The Price of Fairness

## The question

Insurance pricing is one of the few places where a statistical model
directly decides how much people pay. That makes it regulated — and it makes
the word "fair" a technical problem as much as a moral one. This memo
summarizes what the project found, why the findings are mathematical rather
than political, and what they mean for a pricing team.

## What "fair" can mean

There is no single definition of fairness in statistics; there are
incompatible ones. The project measures the three used most in regulation
and ML fairness:

1. **Demographic parity** — every group pays the same on average.
2. **Equalized odds** — a model flags claims equally well in every group
   (same true-positive and false-positive rates).
3. **Calibration** — a premium means the same thing in every group
   (predicted loss = actual loss within each group).

These sound similar. They are not.

## The core result: you cannot have them all

When base rates differ across groups — and in insurance they almost always
do — the definitions conflict. The cleanest statement is Chouldechova's
impossibility result: **calibration and equalized odds cannot both hold**.
Demographic parity conflicts with calibration too.

The project demonstrates this on its own pipeline, not just in theory:

* The baseline is **calibrated**: predicted premiums match actual losses
  within ~1% by gender and 0.94–1.05 by segment.
* The same baseline **violates parity**: F pays $430 on average, M pays
  $547; the full segment spread is 3.4× (F/A $273 vs M/C $919).
* At a common threshold, **equalized odds fail**: TPR 0.50 (F) vs 0.68 (M).
* Matching TPRs would require thresholds 0.131 (F) vs 0.164 (M) — the same
  score would mean different risk in different groups, breaking calibration.

Conclusion: "make the model fair" is not a well-posed instruction. The
real question is *which* fairness you are buying and *what it costs*.

## What it costs — the frontier

Constrained pricing makes the trade-off explicit. Flattening all six segment
means (full demographic parity):

* Accuracy cost: **+2.1% MSE**
* Money moved: **35% of premium volume** — a large cross-subsidy from
  low-risk segments (F/A, M/A) to high-risk ones (M/C, F/C)
* Calibration: premium/actual ratios swing from 0.54 (M/C) to 1.85 (F/A)

Banning a *single* variable — gender, as the EU does — is much cheaper:
**+0.14% MSE, 12% of volume moved**. Territory still does the pricing work,
so parity across gender costs almost nothing in accuracy.

That asymmetry is the memo's headline: **the cost of fairness depends
entirely on which fairness you require, and a well-designed rule can be
nearly free while a blunt one is very expensive.**

## A modeling lesson worth keeping

The first version of the frequency model used a linear age term and badly
mispriced age bands — young drivers undercharged 37%, middle-aged drivers
overcharged ~50%. The fix (age splines) brought every age band within
0.95–1.05 of calibration. Two takeaways:

* Always check calibration on every rating dimension, not just the segments
  in the business brief.
* A model can be perfectly calibrated overall and badly wrong in parts.

## Regulatory context

* **EU** — the Gender Directive (2004/113/EC) bans sex as a rating factor;
  the EU AI Act classifies risk assessment and pricing in life/health
  insurance as high-risk, adding transparency and non-discrimination duties.
* **Colorado SB 21-169** — prohibits insurers from using algorithms or
  predictive models (including external consumer data) that unfairly
  discriminate based on race, sex, religion, and other protected classes;
  auto and health carriers began annual compliance reporting July 1, 2026.
* **NAIC Model Bulletin on AI** (Dec 2023) — principles-based governance
  requiring documentation, verification, and testing for bias and unfair
  discrimination; model cards are emerging as a standardized reporting tool.

A regulator's fairness (don't use protected variables) and a statistician's
fairness (don't let outcomes diverge) are different objectives. The project
quantifies both and shows they cannot be satisfied at once.

## Honest limitations

* The data is synthetic; the *shape* of the trade-off is the result, not the
  dollar amounts.
* We implemented demographic parity on premiums. Equalized odds on claim
  rates, calibration constraints, or individual fairness would trace
  different frontiers.
* Fairness metrics measure groups, not individuals. Two people in the same
  segment can still be treated differently by other pricing features.

## Bottom line

Fairness in pricing is not a checkbox; it is a measured trade-off with a
price tag and identifiable winners and losers. The value of this project is
making that trade-off visible — and showing that smart regulation (ban one
variable, keep the risk signal) can be nearly free, while blunt regulation
(require equal outcomes everywhere) is expensive and misprices risk.
