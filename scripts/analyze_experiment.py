# scripts/analyze_experiment.py
import argparse

import pandas as pd

from ab_test_online_ads.analysis import AbTestAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze A/B test results for online ads."
    )
    parser.add_argument(
        "input",
        help="Path to CSV with impression-level data (variant, \
        clicked, converted, revenue).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (e.g., 0.05).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    analyzer = AbTestAnalyzer(df, alpha=args.alpha)
    summary = analyzer.summarize()

    print("\n=== A/B Test Results ===")
    for key, res in summary.items():
        print(f"\nMetric: {res['metric']}")
        print(f"  Variant A: {res['variant_a']:.4f}")
        print(f"  Variant B: {res['variant_b']:.4f}")
        print(f"  Lift (B vs A): {res['lift'] * 100:.2f}%")
        print(f"  p-value: {res['p_value']:.4f}")
        print(
            f"  Significant at alpha={args.alpha}? "
            f"{'YES' if res['significant'] else 'NO'}"
        )


if __name__ == "__main__":
    main()
