# tests/test_analysis.py

import numpy as np
import pandas as pd
import pytest

from ab_test_online_ads.analysis import AbTestAnalyzer


def _make_simple_df_for_ctr():
    """
    Create a tiny deterministic dataset for CTR testing.

    A: 20 clicks / 100 impressions
    B: 50 clicks / 100 impressions
    """
    n_a = 100
    n_b = 100

    variants = ["A"] * n_a + ["B"] * n_b

    clicked = [1] * 20 + [0] * (n_a - 20) + [1] * 50 + [0] * (n_b - 50)

    # Not important for CTR test, but must exist for analyzer
    converted = [0] * (n_a + n_b)
    revenue = [0.0] * (n_a + n_b)

    return pd.DataFrame(
        {
            "impression_id": np.arange(1, n_a + n_b + 1),
            "user_id": [1] * (n_a + n_b),
            "variant": variants,
            "clicked": clicked,
            "converted": converted,
            "revenue": revenue,
        }
    )


def test_ctr_computation_and_significance():
    """AbTestAnalyzer.ctr should compute correct rates and positive lift."""
    df = _make_simple_df_for_ctr()
    analyzer = AbTestAnalyzer(df, alpha=0.05)

    result = analyzer.ctr()

    # A: 20/100 = 0.2, B: 50/100 = 0.5
    assert pytest.approx(result.variant_a, rel=1e-6) == 0.2
    assert pytest.approx(result.variant_b, rel=1e-6) == 0.5

    # Lift = (0.5 - 0.2) / 0.2 = 1.5 (150%)
    assert pytest.approx(result.lift, rel=1e-6) == 1.5

    # Should be statistically significant with such a large difference
    assert 0 <= result.p_value <= 1
    assert result.significant is np.True_
    assert result.metric_name == "click_through_rate"


def test_conversion_rate_per_impression():
    """Conversion rate per impression should be computed correctly."""
    n_a = 80
    n_b = 120

    # A: 8 conversions / 80 impressions = 0.1
    # B: 24 conversions / 120 impressions = 0.2
    variants = ["A"] * n_a + ["B"] * n_b
    clicked = [1] * (n_a + n_b)  # arbitrary non-zero clicks
    converted = [1] * 8 + [0] * (n_a - 8) + [1] * 24 + [0] * (n_b - 24)
    revenue = [0.0] * (n_a + n_b)

    df = pd.DataFrame(
        {
            "impression_id": np.arange(1, n_a + n_b + 1),
            "user_id": [1] * (n_a + n_b),
            "variant": variants,
            "clicked": clicked,
            "converted": converted,
            "revenue": revenue,
        }
    )

    analyzer = AbTestAnalyzer(df, alpha=0.05)
    result = analyzer.conversion_rate(denominator="impressions")

    assert pytest.approx(result.variant_a, rel=1e-6) == 0.1
    assert pytest.approx(result.variant_b, rel=1e-6) == 0.2
    assert result.lift > 0
    assert 0 <= result.p_value <= 1


def test_revenue_per_impression():
    """Revenue per impression should match the means we expect."""
    # A: 3 impressions, revenues: 0, 10, 20 → mean = 10
    # B: 3 impressions, revenues: 0, 20, 40 → mean = 20
    variants = ["A", "A", "A", "B", "B", "B"]
    clicked = [1, 1, 1, 1, 1, 1]
    converted = [0, 1, 1, 0, 1, 1]
    revenue = [0.0, 10.0, 20.0, 0.0, 20.0, 40.0]

    df = pd.DataFrame(
        {
            "impression_id": np.arange(1, 7),
            "user_id": [1] * 6,
            "variant": variants,
            "clicked": clicked,
            "converted": converted,
            "revenue": revenue,
        }
    )

    analyzer = AbTestAnalyzer(df, alpha=0.05)
    result = analyzer.revenue_per_impression()

    assert pytest.approx(result.variant_a, rel=1e-6) == 10.0
    assert pytest.approx(result.variant_b, rel=1e-6) == 20.0
    assert result.lift > 0
    assert 0 <= result.p_value <= 1
    assert result.metric_name == "revenue_per_impression"


def test_summarize_returns_all_core_metrics():
    """summarize() should return keys and reasonable value structures."""
    df = _make_simple_df_for_ctr()
    analyzer = AbTestAnalyzer(df, alpha=0.05)

    summary = analyzer.summarize()

    # We expect at least these metrics
    assert "ctr" in summary
    assert "conversion_rate_per_impression" in summary
    assert "revenue_per_impression" in summary

    for metric_key, data in summary.items():
        assert "metric" in data
        assert "variant_a" in data
        assert "variant_b" in data
        assert "lift" in data
        assert "p_value" in data
        assert "significant" in data
        assert isinstance(data["significant"], np.bool_)
