import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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


def print_metrics(name, actual, predicted):

    print(f"\n{name}")
    print("Records:", len(actual))

    if len(actual) == 0:
        print("No records available for this group.")
        return

    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print(
        "Balanced Accuracy:",
        round(balanced_accuracy_score(actual, predicted), 4)
    )
    print(
        "Macro F1:",
        round(f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ), 4)
    )


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

    train_keys = set(
        map(tuple, X_train.astype(str).to_numpy())
    )

    test_keys = list(
        map(tuple, X_test.astype(str).to_numpy())
    )

    seen_mask = pd.Series(
        [key in train_keys for key in test_keys],
        index=X_test.index
    )

    unseen_mask = ~seen_mask

    print("CAREERPATH UNSEEN PROFILE AUDIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Unique training combinations:", len(train_keys))
    print("Test combinations seen in training:", int(seen_mask.sum()))
    print("Test combinations unseen in training:", int(unseen_mask.sum()))

    print(
        "Seen combination percentage:",
        round(seen_mask.mean() * 100, 2)
    )

    model = build_pipeline()

    model.fit(X_train, y_train)

    predictions = pd.Series(
        model.predict(X_test),
        index=X_test.index
    )

    print_metrics(
        "OVERALL TEST PERFORMANCE",
        y_test,
        predictions
    )

    print_metrics(
        "SEEN COMBINATIONS",
        y_test[seen_mask],
        predictions[seen_mask]
    )

    print_metrics(
        "UNSEEN COMBINATIONS",
        y_test[unseen_mask],
        predictions[unseen_mask]
    )

    results = X_test.copy()
    results["Actual_Career"] = y_test
    results["Predicted_Career"] = predictions
    results["Combination_Seen_In_Training"] = seen_mask

    output_file = REPORT_DIR / "unseen_profile_predictions.csv"
    results.to_csv(output_file, index=False)

    print("\nSaved predictions:", output_file)
    print("\nUNSEEN PROFILE AUDIT COMPLETED")


if __name__ == "__main__":
    main()
