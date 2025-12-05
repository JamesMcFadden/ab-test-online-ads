# src/data_generation.py
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AdExperimentConfig:
    """
    Configuration for synthetic online ads A/B experiment.
    Version B has a slightly higher CTR, conversion rate,
    and revenue distribution mean.
    """

    n_impressions: int = 10000

    # CTR (click-through rate) per impression
    ctr_a: float = 0.05
    ctr_b: float = 0.06

    # Conversion rate per impression
    conv_a: float = 0.01
    conv_b: float = 0.012

    # Revenue distribution (only applied when converted == 1)
    rev_mean_a: float = 10.0
    rev_mean_b: float = 11.0
    rev_std: float = 2.0

    # For reproducibility
    seed: int | None = 19


def generate_synthetic_data(config: AdExperimentConfig) -> pd.DataFrame:
    """
    Generate synthetic impression-level data for an ad A/B test.

    Columns:
      - impression_id: unique row ID
      - user_id: simulated user identifier
      - variant: "A" or "B"
      - clicked: 0/1 click indicator
      - converted: 0/1 conversion indicator
      - revenue: revenue from this impression (0 if not converted)
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    n = config.n_impressions

    # Randomly assign impressions to A or B
    variants = np.random.choice(["A", "B"], size=n)

    # CTR per impression
    ctrs = np.where(variants == "A", config.ctr_a, config.ctr_b)
    clicked = (np.random.rand(n) < ctrs).astype(int)

    # Conversion per impression
    convs = np.where(variants == "A", config.conv_a, config.conv_b)
    converted = (np.random.rand(n) < convs).astype(int)

    # Revenue for converted users (0 if not converted)
    means = np.where(variants == "A", config.rev_mean_a, config.rev_mean_b)
    revenue = np.where(
        converted == 1,
        np.random.normal(loc=means, scale=config.rev_std, size=n).clip(min=0),
        0.0,
    )

    df = pd.DataFrame(
        {
            "impression_id": np.arange(1, n + 1),
            "user_id": np.random.randint(1, n // 2 + 1, size=n),
            "variant": variants,
            "clicked": clicked,
            "converted": converted,
            "revenue": revenue,
        }
    )
    return df
