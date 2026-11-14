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
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42
        ))
    ])


def unseen_mask(train, test, columns):

    train_keys = set(
        map(tuple, train[columns].astype(str).to_numpy())
    )

    test_keys = list(
        map(tuple, test[columns].astype(str).to_numpy())
    )

    return pd.Series(
        [key not in train_keys for key in test_keys],
        index=test.index
    )


def evaluate(name, mask, actual, predicted):

    actual_group = actual[mask]
    predicted_group = predicted[mask]

    print(f"\n{name}")
    print("Records:", len(actual_group))

    if len(actual_group) == 0:
        print("No unseen records in this group.")
        return

    print(
        "Accuracy:",
        round(accuracy_score(actual_group, predicted_group), 4)
    )
    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(actual_group, predicted_group),
            4
        )
    )
    print(
        "Macro F1:",
        round(
            f1_score(
                actual_group,
                predicted_group,
                average="macro",
                zero_division=0
            ),
            4
        )
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

    novelty_groups = {
        "Unseen technical skill values": [
            "Technical_Skills_Parsed"
        ],
        "Unseen skills gained values": [
            "Skills_Gained_Parsed"
        ],
        "Unseen technical + gained pairs": [
            "Technical_Skills_Parsed",
            "Skills_Gained_Parsed"
        ],
        "Unseen specialization + skill combinations": FEATURES,
    }

    print("CAREERPATH FEATURE NOVELTY AUDIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = pd.Series(
        model.predict(X_test),
        index=X_test.index
    )

    print("\nOVERALL ACCURACY:", round(
        accuracy_score(y_test, predictions), 4
    ))

    results = []

    for name, columns in novelty_groups.items():

        mask = unseen_mask(X_train, X_test, columns)

        evaluate(name, mask, y_test, predictions)

        actual_group = y_test[mask]
        predicted_group = predictions[mask]

        results.append({
            "Group": name,
            "Unseen_Records": len(actual_group),
            "Accuracy": (
                accuracy_score(actual_group, predicted_group)
                if len(actual_group) else None
            ),
            "Balanced_Accuracy": (
                balanced_accuracy_score(actual_group, predicted_group)
                if len(actual_group) else None
            ),
            "Macro_F1": (
                f1_score(
                    actual_group,
                    predicted_group,
                    average="macro",
                    zero_division=0
                )
                if len(actual_group) else None
            )
        })

    output_file = REPORT_DIR / "feature_novelty_results.csv"

    pd.DataFrame(results).to_csv(output_file, index=False)

    print("\nSaved report:", output_file)
    print("\nFEATURE NOVELTY AUDIT COMPLETED")


if __name__ == "__main__":
    main()
