# scripts/run_experiment.py
import argparse
from pathlib import Path

from ab_test_online_ads.data_generation import AdExperimentConfig, generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic A/B test for online ads."
    )
    parser.add_argument(
        "--n-impressions",
        type=int,
        default=10000,
        help="Number of ad impressions to simulate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ads_experiment.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    cfg = AdExperimentConfig(n_impressions=args.n_impressions)
    df = generate_synthetic_data(cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic experiment data to {out_path.resolve()}")


if __name__ == "__main__":
    main()
