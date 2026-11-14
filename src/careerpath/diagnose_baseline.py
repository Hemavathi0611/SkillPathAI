import joblib
import pandas as pd
import re

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_FILE = Path("data/processed/career_model_ready.csv")
MODEL_FILE = Path("models/careerpath_logistic_regression.joblib")
REPORT_DIR = Path("reports/careerpath")

TARGET = "Recommended Career"

TEXT_COLUMNS = [
    "Technical_Skills_Parsed",
    "Soft_Skills_Parsed",
    "Skills_Gained_Parsed",
]

ALL_FEATURES = [
    "Education Level",
    "Specialization",
    "Certifications",
    "Score_Type",
    "Score_Value",
    *TEXT_COLUMNS,
]


def normalize_text(value):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower()
    ).strip()


def main():
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_FILE)
    predictions = model.predict(X_test)

    results = X_test.copy()
    results["Actual_Career"] = y_test
    results["Predicted_Career"] = predictions
    results["Correct"] = (
        results["Actual_Career"] == results["Predicted_Career"]
    )

    errors = results[~results["Correct"]].copy()

    print("\n========== PREDICTION ERROR AUDIT ==========")
    print("Test records:", len(results))
    print("Correct predictions:", int(results["Correct"].sum()))
    print("Incorrect predictions:", len(errors))

    if errors.empty:
        print("No incorrect predictions found.")
    else:
        print("\nIncorrect prediction examples:")
        print(
            errors[
                ["Actual_Career", "Predicted_Career"]
            ].head(15).to_string(index=False)
        )

    print("\n========== TARGET TEXT OVERLAP AUDIT ==========")

    feature_text = X.astype(str).agg(" ".join, axis=1)
    normalized_features = feature_text.map(normalize_text)

    overlap_count = 0
    overlap_examples = []

    for index, target in y.items():
        normalized_target = normalize_text(target)

        if normalized_target and normalized_target in normalized_features.loc[index]:
            overlap_count += 1

            if len(overlap_examples) < 10:
                overlap_examples.append(
                    (target, feature_text.loc[index][:250])
                )

    print("Rows where complete target name appears in features:", overlap_count)

    for target, example in overlap_examples:
        print("\nTarget:", target)
        print("Feature text:", example)

    print("\n========== FEATURE-LEVEL OVERLAP ==========")

    for column in ALL_FEATURES:
        if column not in TEXT_COLUMNS:
            continue

        normalized_column = X[column].map(normalize_text)
        matches = 0

        for index, target in y.items():
            normalized_target = normalize_text(target)

            if normalized_target and normalized_target in normalized_column.loc[index]:
                matches += 1

        print(f"{column}: {matches} complete target-name matches")

    print("\n========== MOST COMMON CONFUSIONS ==========")

    if not errors.empty:
        confusion = (
            errors.groupby(
                ["Actual_Career", "Predicted_Career"]
            )
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        print(confusion.to_string())

    print("\n========== DIAGNOSTIC COMPLETED ==========")


if __name__ == "__main__":
    main()
