# src/analysis.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, Dict, Any
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

Variant = Literal["A", "B"]


@dataclass
class MetricResult:
    metric_name: str
    variant_a: float
    variant_b: float
    lift: float  # (B - A) / A
    p_value: float
    significant: bool


class AbTestAnalyzer:
    """
    A/B test analyzer for online ads.

    Expects a DataFrame with at least:
      - variant: "A" or "B"
      - clicked: 0/1
      - converted: 0/1
      - revenue: numeric (0 if none)
    """

    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        required_cols = {"variant", "clicked", "converted", "revenue"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.df = df.copy()
        self.alpha = alpha

    def _binary_ztest(self, success_col: str, metric_name: str) -> MetricResult:
        """
        Helper for binary metrics (CTR, conversion rate).
        """
        group = self.df.groupby("variant")[success_col].agg(["sum", "count"])
        if not {"A", "B"} <= set(group.index):
            raise ValueError("Both variants A and B must be present in data.")

        successes = group.loc[["A", "B"], "sum"].to_numpy()
        nobs = group.loc[["A", "B"], "count"].to_numpy()

        stat, p_value = proportions_ztest(successes, nobs)
        rate_a = successes[0] / nobs[0]
        rate_b = successes[1] / nobs[1]
        lift = (rate_b - rate_a) / rate_a if rate_a > 0 else np.nan
        significant = p_value < self.alpha

        return MetricResult(
            metric_name=metric_name,
            variant_a=rate_a,
            variant_b=rate_b,
            lift=lift,
            p_value=p_value,
            significant=significant,
        )

    # ---- Public metrics ----

    def ctr(self) -> MetricResult:
        """
        Click-through rate per impression.
        """
        return self._binary_ztest("clicked", "click_through_rate")

    def conversion_rate(self, denominator: str = "impressions") -> MetricResult:
        """
        Conversion rate, either:
          - per impression (default)
          - per click (set denominator="clicks")
        """
        if denominator == "impressions":
            return self._binary_ztest("converted", "conversion_rate_per_impression")

        if denominator == "clicks":
            df_clicked = self.df[self.df["clicked"] == 1].copy()
            if df_clicked.empty:
                raise ValueError("No clicks in dataset to compute conversion per click.")
            return AbTestAnalyzer(df_clicked, alpha=self.alpha)._binary_ztest(
                "converted", "conversion_rate_per_click"
            )

        raise ValueError("denominator must be 'impressions' or 'clicks'")

    def revenue_per_impression(self) -> MetricResult:
        """
        Average revenue per impression (can be 0 for non-converting users).
        Uses Welch's t-test for mean difference.
        """
        group = self.df.groupby("variant")["revenue"].agg(["mean"])
        if not {"A", "B"} <= set(group.index):
            raise ValueError("Both variants A and B must be present in data.")

        rev_a = self.df[self.df["variant"] == "A"]["revenue"].to_numpy()
        rev_b = self.df[self.df["variant"] == "B"]["revenue"].to_numpy()

        t_stat, p_value = stats.ttest_ind(rev_a, rev_b, equal_var=False)
        mean_a = group.loc["A", "mean"]
        mean_b = group.loc["B", "mean"]
        lift = (mean_b - mean_a) / mean_a if mean_a != 0 else np.nan
        significant = p_value < self.alpha

        return MetricResult(
            metric_name="revenue_per_impression",
            variant_a=mean_a,
            variant_b=mean_b,
            lift=lift,
            p_value=p_value,
            significant=significant,
        )

    def summarize(self) -> Dict[str, Any]:
        """
        Convenience method: return a dictionary of core metrics.
        """
        results = {
            "ctr": self.ctr(),
            "conversion_rate_per_impression": self.conversion_rate("impressions"),
            "revenue_per_impression": self.revenue_per_impression(),
        }

        return {
            name: {
                "metric": r.metric_name,
                "variant_a": r.variant_a,
                "variant_b": r.variant_b,
                "lift": r.lift,
                "p_value": r.p_value,
                "significant": r.significant,
            }
            for name, r in results.items()
        }
