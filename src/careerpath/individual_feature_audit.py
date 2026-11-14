import json
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

DATA_FILE = Path("data/processed/career_model_ready.csv")
REPORT_DIR = Path("reports/careerpath")

TARGET = "Recommended Career"

FEATURES = {
    "Education Level": "categorical",
    "Specialization": "categorical",
    "Certifications": "categorical",
    "Score_Type": "categorical",
    "Score_Value": "numeric",
    "Technical_Skills_Parsed": "text",
    "Soft_Skills_Parsed": "text",
    "Skills_Gained_Parsed": "text",
}


def build_pipeline(feature, feature_type):

    if feature_type == "categorical":
        transformer = ColumnTransformer([
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                [feature]
            )
        ])

    elif feature_type == "numeric":
        transformer = ColumnTransformer([
            (
                "numeric",
                StandardScaler(),
                [feature]
            )
        ])

    else:
        transformer = ColumnTransformer([
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=10000
                ),
                feature
            )
        ])

    return Pipeline([
        ("preprocessor", transformer),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=42
            )
        )
    ])


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df[list(FEATURES.keys())]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []

    print("CAREERPATH INDIVIDUAL FEATURE AUDIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Target classes:", y.nunique())

    for feature, feature_type in FEATURES.items():

        print(f"\nTesting feature: {feature}")

        model = build_pipeline(feature, feature_type)

        model.fit(X_train[[feature]], y_train)

        predictions = model.predict(X_test[[feature]])

        result = {
            "Feature": feature,
            "Feature_Type": feature_type,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "Macro_F1": f1_score(
                y_test, predictions, average="macro"
            ),
            "Weighted_F1": f1_score(
                y_test, predictions, average="weighted"
            )
        }

        results.append(result)

        print("Accuracy:", round(result["Accuracy"], 4))
        print(
            "Balanced Accuracy:",
            round(result["Balanced_Accuracy"], 4)
        )
        print("Macro F1:", round(result["Macro_F1"], 4))

    results_df = pd.DataFrame(results).sort_values(
        "Macro_F1",
        ascending=False
    )

    output_file = REPORT_DIR / "individual_feature_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\n========== FEATURE COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nSaved report:", output_file)
    print("\nINDIVIDUAL FEATURE AUDIT COMPLETED")


if __name__ == "__main__":
    main()
