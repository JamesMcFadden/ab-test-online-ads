# ab-test-online-ads
The repository demonstrates how to perform A/B tests for online ads

cd ab_test_ads
python scripts/run_experiment.py --n-impressions 20000 --output data/ads_experiment.csv

# A/B Testing Toolkit for Online Ads

This repo provides a minimal but complete codebase to:

- Run A/B tests on online ads
- Measure:
  - Click-through rate (CTR)
  - Conversion rate
  - Revenue per impression
- Perform statistical tests for significance

## Setup

```bash
git clone <your-repo-url>
cd ab_test_ads
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate synthetic experiment data
python scripts/run_experiment.py --n-impressions 20000 --output data/ads_experiment.csv

### Analyze the experiment:
python scripts/analyze_experiment.py data/ads_experiment.csv --alpha 0.05
