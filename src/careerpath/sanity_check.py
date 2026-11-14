import json
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

DATA_FILE = Path("data/processed/career_model_ready.csv")
REPORT_DIR = Path("reports/careerpath")

TARGET = "Recommended Career"

FEATURES = [
    "Specialization",
    "Technical_Skills_Parsed",
    "Skills_Gained_Parsed",
]


def build_pipeline():

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Specialization"]
        ),
        (
            "technical",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000
            ),
            "Technical_Skills_Parsed"
        ),
        (
            "skills_gained",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000
            ),
            "Skills_Gained_Parsed"
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=42
            )
        )
    ])


def calculate_metrics(actual, predicted):

    return {
        "Accuracy": round(
            accuracy_score(actual, predicted), 4
        ),
        "Balanced_Accuracy": round(
            balanced_accuracy_score(actual, predicted), 4
        ),
        "Macro_F1": round(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            ),
            4
        )
    }


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []

    # Experiment 1: Train with the original labels.
    real_model = build_pipeline()
    real_model.fit(X_train, y_train)

    real_predictions = real_model.predict(X_test)

    real_metrics = calculate_metrics(
        y_test,
        real_predictions
    )

    results.append({
        "Experiment": "Original labels",
        **real_metrics
    })

    # Experiment 2: Train with shuffled labels.
    shuffled_labels = y_train.sample(
        frac=1,
        random_state=42
    ).to_numpy()

    shuffled_model = build_pipeline()
    shuffled_model.fit(X_train, shuffled_labels)

    shuffled_predictions = shuffled_model.predict(X_test)

    shuffled_metrics = calculate_metrics(
        y_test,
        shuffled_predictions
    )

    results.append({
        "Experiment": "Shuffled training labels",
        **shuffled_metrics
    })

    # Experiment 3: Majority-class baseline.
    dummy_model = DummyClassifier(
        strategy="most_frequent"
    )

    dummy_model.fit(X_train, y_train)

    dummy_predictions = dummy_model.predict(X_test)

    dummy_metrics = calculate_metrics(
        y_test,
        dummy_predictions
    )

    results.append({
        "Experiment": "Majority-class baseline",
        **dummy_metrics
    })

    results_df = pd.DataFrame(results)

    print("CAREERPATH MODEL SANITY CHECK")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Number of career classes:", y.nunique())
    print(
        "Approximate random chance level:",
        round(1 / y.nunique(), 4)
    )

    print("\nCOMPARISON RESULTS")
    print(results_df.to_string(index=False))

    output_file = REPORT_DIR / "sanity_check_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\nSaved report:", output_file)

    if (
        shuffled_metrics["Accuracy"] < real_metrics["Accuracy"]
        and dummy_metrics["Accuracy"] < real_metrics["Accuracy"]
    ):
        print("\nSANITY CHECK: PASSED")
    else:
        print("\nSANITY CHECK: NEEDS INVESTIGATION")

    print("\nMODEL SANITY CHECK COMPLETED")


if __name__ == "__main__":
    main()
