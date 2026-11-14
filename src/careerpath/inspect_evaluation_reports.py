import json
import pandas as pd
from pathlib import Path

folder = Path("reports/careerpath")

files = [
    "feature_ablation_results.csv",
    "individual_feature_results.csv",
    "feature_novelty_results.csv",
    "sanity_check_results.csv",
    "cross_validation_results.csv",
]

for filename in files:
    path = folder / filename
    df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(filename)
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print(df.head(3).to_string(index=False))

print("\nBASELINE METRICS JSON")
with open(folder / "metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)

print(json.dumps(metrics, indent=2))
