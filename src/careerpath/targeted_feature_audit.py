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

FEATURE_TYPES = {
    "Education Level": "categorical",
    "Specialization": "categorical",
    "Technical_Skills_Parsed": "text",
    "Skills_Gained_Parsed": "text",
}

EXPERIMENTS = {
    "A_Education_Specialization_Technical": [
        "Education Level",
        "Specialization",
        "Technical_Skills_Parsed",
    ],
    "B_Education_Specialization_Gained": [
        "Education Level",
        "Specialization",
        "Skills_Gained_Parsed",
    ],
    "C_Education_Technical_Gained": [
        "Education Level",
        "Technical_Skills_Parsed",
        "Skills_Gained_Parsed",
    ],
    "D_Specialization_Technical_Gained": [
        "Specialization",
        "Technical_Skills_Parsed",
        "Skills_Gained_Parsed",
    ],
    "E_All_Four": list(FEATURE_TYPES.keys()),
}


def build_pipeline(features):

    transformers = []

    categorical = [
        feature for feature in features
        if FEATURE_TYPES[feature] == "categorical"
    ]

    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ))

    for feature in features:

        if FEATURE_TYPES[feature] == "text":
            transformers.append((
                feature,
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=10000
                ),
                feature
            ))

    preprocessor = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42
        ))
    ])


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df[list(FEATURE_TYPES.keys())]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []

    print("CAREERPATH TARGETED FEATURE AUDIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    for name, features in EXPERIMENTS.items():

        print(f"\nTesting: {name}")

        model = build_pipeline(features)
        model.fit(X_train[features], y_train)

        predictions = model.predict(X_test[features])

        results.append({
            "Experiment": name,
            "Features": " + ".join(features),
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
        })

    results_df = pd.DataFrame(results).sort_values(
        "Macro_F1",
        ascending=False
    )

    output_file = REPORT_DIR / "targeted_feature_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\n========== TARGETED FEATURE RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\nSaved report:", output_file)
    print("\nTARGETED FEATURE AUDIT COMPLETED")


if __name__ == "__main__":
    main()
