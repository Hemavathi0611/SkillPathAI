import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path("data/processed/career_model_ready.csv")
REPORT_DIR = Path("reports/careerpath")

TARGET = "Recommended Career"

CATEGORICAL = [
    "Education Level",
    "Specialization",
    "Certifications",
    "Score_Type",
]

NUMERIC = ["Score_Value"]

TEXT = [
    "Technical_Skills_Parsed",
    "Soft_Skills_Parsed",
    "Skills_Gained_Parsed",
]

EXPERIMENTS = {
    "A_Education_Specialization": [
        "Education Level",
        "Specialization",
    ],
    "B_Education_Score": [
        "Education Level",
        "Specialization",
        "Score_Type",
        "Score_Value",
    ],
    "C_Skills_Only": TEXT,
    "D_All_Features": CATEGORICAL + NUMERIC + TEXT,
}


def build_pipeline(features):
    transformers = []

    categorical = [c for c in features if c in CATEGORICAL]
    numeric = [c for c in features if c in NUMERIC]
    text = [c for c in features if c in TEXT]

    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ))

    if numeric:
        transformers.append((
            "numeric",
            StandardScaler(),
            numeric,
        ))

    for column in text:
        transformers.append((
            column,
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=3000,
            ),
            column,
        ))

    preprocessing = ColumnTransformer(transformers=transformers)

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42,
        )),
    ])


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    for name, features in EXPERIMENTS.items():
        print(f"\nTraining: {name}")

        model = build_pipeline(features)

        model.fit(X_train[features], y_train)

        predictions = model.predict(X_test[features])

        results.append({
            "Experiment": name,
            "Feature_Count": len(features),
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "Macro_F1": f1_score(
                y_test, predictions, average="macro"
            ),
            "Weighted_F1": f1_score(
                y_test, predictions, average="weighted"
            ),
        })

        print("Completed.")

    results_df = pd.DataFrame(results)

    output_file = REPORT_DIR / "feature_ablation_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\n========== FEATURE ABLATION RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\nSaved:", output_file)
    print("\nFEATURE ABLATION COMPLETED")


if __name__ == "__main__":
    main()
