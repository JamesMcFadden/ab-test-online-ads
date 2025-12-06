# tests/test_data_generation.py

import pandas as pd

from ab_test_online_ads.data_generation import (AdExperimentConfig,
                                                generate_synthetic_data)


def test_generate_synthetic_data_basic_shape_and_columns():
    """Generated data has expected length and columns."""
    cfg = AdExperimentConfig(n_impressions=1000, seed=123)
    df = generate_synthetic_data(cfg)

    assert len(df) == 1000

    expected_cols = {
        "impression_id",
        "user_id",
        "variant",
        "clicked",
        "converted",
        "revenue",
    }
    assert expected_cols.issubset(df.columns)

    # Basic type checks
    assert df["impression_id"].is_monotonic_increasing
    assert df["variant"].isin(["A", "B"]).all()
    assert set(df["clicked"].unique()).issubset({0, 1})
    assert set(df["converted"].unique()).issubset({0, 1})
    assert (df["revenue"] >= 0).all()


def test_generate_synthetic_data_reproducible_with_seed():
    """Using the same seed and config should yield identical data."""
    cfg1 = AdExperimentConfig(n_impressions=5000, seed=42)
    cfg2 = AdExperimentConfig(n_impressions=5000, seed=42)

    df1 = generate_synthetic_data(cfg1)
    df2 = generate_synthetic_data(cfg2)

    # DataFrames should be exactly equal
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_synthetic_data_different_seed_differs():
    """Different seeds should generally yield different samples."""
    cfg1 = AdExperimentConfig(n_impressions=5000, seed=42)
    cfg2 = AdExperimentConfig(n_impressions=5000, seed=999)

    df1 = generate_synthetic_data(cfg1)
    df2 = generate_synthetic_data(cfg2)

    # Not a strict guarantee, but very unlikely to be identical if seeds differ
    assert not df1.equals(df2)
