"""Constrained risk pricing and the accuracy-fairness frontier.

Formulation
-----------
Baseline pure premiums p_i (from the frequency x severity GLMs) are
risk-adequate but "unfair": mean premiums differ across protected segments.
Constrained pricing multiplies each segment's premiums by a segment-level
factor lambda_g, chosen to minimize the squared error of premiums against
actual losses, subject to a *demographic parity* constraint:

    | mean(p'_i | g) - mu | <= eps * mu     for every protected parity group g

and budget neutrality:

    sum(p'_i) = sum(p_i)

Sweeping eps from the baseline gap down to 0 traces the accuracy-fairness
frontier: how much predictive accuracy (MSE) is lost per unit of fairness
gained. eps = 0 is full demographic parity on premiums.

The multipliers act at the segment level (gender x territory); the parity
constraint can be applied at any coarser level (e.g., gender only), which
mirrors a regulator banning a single protected variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import osqp
import scipy.sparse as sp


def _codes(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, int]:
    codes = df.groupby(cols, observed=True).ngroup()
    return codes.to_numpy(), int(codes.max()) + 1


class ConstrainedPricing:
    """Segment-multiplier pricing subject to mean-premium parity."""

    def __init__(
        self,
        df: pd.DataFrame,
        premium_col: str = "predicted_premium",
        loss_col: str = "total_claim_amount",
        segment_cols: tuple[str, ...] = ("gender", "territory"),
        parity_cols: tuple[str, ...] | None = None,
    ):
        self.segment_cols = list(segment_cols)
        self.parity_cols = list(parity_cols) if parity_cols else list(segment_cols)

        self.seg, self.n_segments = _codes(df, self.segment_cols)
        self.segment_names = list(
            df.groupby(self.segment_cols, observed=True).size().index
        )
        self.parity, self.n_parity = _codes(df, self.parity_cols)
        self.parity_names = list(df.groupby(self.parity_cols, observed=True).size().index)

        self.premium = df[premium_col].to_numpy(dtype=float)
        self.loss = df[loss_col].to_numpy(dtype=float)
        self.overall_mean = float(self.premium.mean())
        self.segment_mean = np.array(
            [self.premium[self.seg == g].mean() for g in range(self.n_segments)]
        )
        self._mse_base = float(np.mean((self.premium - self.loss) ** 2))
        self.n = len(df)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _objective(self, lam: np.ndarray) -> float:
        pred = self.premium * lam[self.seg]
        # Relative MSE keeps the optimizer's scaling well conditioned.
        return float(np.mean((pred - self.loss) ** 2) / self._mse_base)

    def _constraints(self, eps: float) -> list[dict]:
        mu = self.overall_mean
        premium = self.premium
        seg = self.seg
        total = premium.sum()
        cons: list[dict] = [
            {
                "type": "eq",
                "fun": lambda lam: float((premium * lam[seg]).sum() / total - 1.0),
            }
        ]
        for g in range(self.n_parity):
            mask = self.parity == g
            n_g = int(mask.sum())
            mean_g = lambda lam, m=mask: float(
                (premium[m] * lam[seg[m]]).sum() / n_g / mu
            )
            cons.append(
                {"type": "ineq", "fun": lambda lam, f=mean_g: eps - (f(lam) - 1.0)}
            )
            cons.append(
                {"type": "ineq", "fun": lambda lam, f=mean_g: eps - (1.0 - f(lam))}
            )
        return cons

    def solve(
        self,
        eps: float,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Find segment multipliers minimizing MSE subject to parity <= eps.

        The problem is a convex quadratic program with linear constraints
        (budget neutrality + parity bounds), solved exactly with OSQP.
        """
        n = self.n
        premium = self.premium
        loss = self.loss
        seg = self.seg

        # Objective: min sum_g [A_g lambda_g^2 - 2 B_g lambda_g] / n
        # with A_g = sum_{i in g} p_i^2, B_g = sum_{i in g} p_i * y_i.
        a2 = np.array(
            [(premium[seg == g] ** 2).sum() for g in range(self.n_segments)]
        ) / n
        b1 = np.array(
            [(premium[seg == g] * loss[seg == g]).sum() for g in range(self.n_segments)]
        ) / n

        p_mat = sp.csc_matrix(sp.diags(2.0 * a2))
        q_vec = -2.0 * b1

        rows = []
        lows = []
        ups = []

        # Budget neutrality: sum(p') = sum(p)
        rows.append(np.bincount(seg, weights=premium, minlength=self.n_segments))
        lows.append(premium.sum())
        ups.append(premium.sum())

        # Parity bounds on each protected group's mean premium.
        mu = self.overall_mean
        for h in range(self.n_parity):
            mask = self.parity == h
            weights = premium[mask] / mask.sum()
            rows.append(
                np.bincount(seg[mask], weights=weights, minlength=self.n_segments)
            )
            lows.append(mu * (1 - eps))
            ups.append(mu * (1 + eps))

        # Variable bounds are expressed as identity rows (OSQP has no lb/ub).
        a_mat = sp.csc_matrix(
            np.vstack([np.vstack(rows), np.eye(self.n_segments)])
        )
        l_vec = np.concatenate([np.array(lows), np.full(self.n_segments, 0.2)])
        u_vec = np.concatenate([np.array(ups), np.full(self.n_segments, 2.5)])

        prob = osqp.OSQP()
        prob.setup(
            p_mat,
            q_vec,
            a_mat,
            l_vec,
            u_vec,
            verbose=False,
            eps_abs=1e-9,
            eps_rel=1e-9,
            max_iter=10_000,
        )
        result = prob.solve()
        if result.info.status not in {"solved", "solved inaccurate"}:
            raise RuntimeError(
                f"QP failed for eps={eps}: {result.info.status}"
            )
        return result.x

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self, lam: np.ndarray) -> dict:
        pred = self.premium * lam[self.seg]
        mse_base = float(np.mean((self.premium - self.loss) ** 2))
        mse = float(np.mean((pred - self.loss) ** 2))
        group_means = np.array(
            [pred[self.parity == g].mean() for g in range(self.n_parity)]
        )
        group_loss_ratio = np.array(
            [
                self.loss[self.parity == g].sum() / pred[self.parity == g].sum()
                for g in range(self.n_parity)
            ]
        )
        gap = float((group_means.max() - group_means.min()) / self.overall_mean)
        redistribution = float(
            np.abs(pred - self.premium).sum() / self.premium.sum()
        )
        return {
            "cost_mse": mse / mse_base - 1.0,
            "fairness_gap": gap,
            "redistribution": redistribution,
            "loss_ratio": float(self.loss.sum() / pred.sum()),
            "loss_ratio_gap": float(group_loss_ratio.max() - group_loss_ratio.min()),
            "group_mean_premium": group_means,
            "segment_multipliers": lam,
        }

    def frontier(
        self,
        eps_grid: np.ndarray | None = None,
        n_points: int = 13,
    ) -> pd.DataFrame:
        """Trace the Pareto frontier from baseline to full parity."""
        if eps_grid is None:
            baseline_gap = self._baseline_gap()
            eps_grid = np.linspace(baseline_gap, 0.0, n_points)

        rows = []
        x0: np.ndarray | None = None
        for eps in eps_grid:
            lam = self.solve(eps, x0=x0)
            x0 = lam
            metrics = self.metrics(lam)
            row = {
                "eps": float(eps),
                "cost_mse": metrics["cost_mse"],
                "fairness_gap": metrics["fairness_gap"],
                "redistribution": metrics["redistribution"],
                "loss_ratio": metrics["loss_ratio"],
                "loss_ratio_gap": metrics["loss_ratio_gap"],
            }
            for i in range(self.n_segments):
                row[f"multiplier_{i}"] = float(lam[i])
            for g, m in enumerate(metrics["group_mean_premium"]):
                row[f"group_mean_{g}"] = float(m)
            rows.append(row)
        table = pd.DataFrame(rows)
        # The frontier reports the best achievable cost at each fairness level
        # (running max removes small numerical non-monotonicities).
        table["cost_mse"] = np.maximum.accumulate(table["cost_mse"])
        # Express cost relative to the unconstrained (loosest) point.
        table["cost_mse"] = table["cost_mse"] - table["cost_mse"].iloc[0]
        return table

    def _baseline_gap(self) -> float:
        group_means = np.array(
            [
                self.premium[self.parity == g].mean()
                for g in range(self.n_parity)
            ]
        )
        return float((group_means.max() - group_means.min()) / self.overall_mean)

    # ------------------------------------------------------------------
    # Redistribution
    # ------------------------------------------------------------------

    def redistribution_table(
        self, lam: np.ndarray, baseline_lam: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Per-segment premium changes and cross-subsidy flows."""
        if baseline_lam is None:
            baseline_lam = np.ones(self.n_segments)
        base_premium = self.premium * baseline_lam[self.seg]
        new_premium = self.premium * lam[self.seg]
        rows = []
        for g in range(self.n_segments):
            mask = self.seg == g
            old = base_premium[mask].mean()
            new = new_premium[mask].mean()
            rows.append(
                {
                    "segment": self.segment_names[g],
                    "n_policies": int(mask.sum()),
                    "baseline_premium": old,
                    "constrained_premium": new,
                    "change": new - old,
                    "change_pct": (new / old - 1) * 100,
                    "total_moved": float((new_premium[mask] - base_premium[mask]).sum()),
                }
            )
        table = pd.DataFrame(rows).sort_values("change", ascending=False)
        table["is_payer"] = table["change"] < 0
        return table.round(2)
