import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = build_pipeline()

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    print("CAREERPATH 5-FOLD CROSS-VALIDATION")
    print("Total records:", len(df))
    print("Career classes:", y.nunique())
    print("Number of folds:", cv.get_n_splits())

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1
    )

    fold_results = pd.DataFrame({
        "Fold": range(1, 6),
        "Accuracy": results["test_accuracy"],
        "Balanced_Accuracy": results["test_balanced_accuracy"],
        "Macro_F1": results["test_macro_f1"],
    })

    print("\nFOLD-WISE RESULTS")
    print(fold_results.round(4).to_string(index=False))

    print("\nAVERAGE RESULTS")
    for metric in scoring:
        values = results[f"test_{metric}"]
        print(
            f"{metric}: "
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )

    output_file = REPORT_DIR / "cross_validation_results.csv"
    fold_results.to_csv(output_file, index=False)

    print("\nSaved report:", output_file)
    print("\nCROSS-VALIDATION COMPLETED")


if __name__ == "__main__":
    main()
