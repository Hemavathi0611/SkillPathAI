import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("data/processed/career_model_ready.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/careerpath")

TARGET = "Recommended Career"

CATEGORICAL_FEATURES = [
    "Education Level",
    "Specialization",
    "Certifications",
    "Score_Type",
]

NUMERIC_FEATURES = [
    "Score_Value",
]

TEXT_FEATURES = [
    "Technical_Skills_Parsed",
    "Soft_Skills_Parsed",
    "Skills_Gained_Parsed",
]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if y.isna().any():
        raise ValueError("Missing target values detected.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "score",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "technical_skills",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=3000
                ),
                "Technical_Skills_Parsed"
            ),
            (
                "soft_skills",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=2000
                ),
                "Soft_Skills_Parsed"
            ),
            (
                "skills_gained",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=3000
                ),
                "Skills_Gained_Parsed"
            ),
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    random_state=42
                )
            )
        ]
    )

    print("CareerPath Logistic Regression Training")
    print("---------------------------------------")
    print("Total records:", len(df))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Input features:", X.shape[1])
    print("Target classes:", y.nunique())

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted")
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / "careerpath_logistic_regression.joblib"
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL EVALUATION")
    print("----------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nSaved model:")
    print(MODEL_DIR / "careerpath_logistic_regression.joblib")

    print("\nSaved reports:")
    print(REPORT_DIR)

    print("\nCAREERPATH BASELINE TRAINING COMPLETED")


if __name__ == "__main__":
    main()
