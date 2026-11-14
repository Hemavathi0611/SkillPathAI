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

FEATURE_TYPES = {
    "Education Level": "categorical",
    "Specialization": "categorical",
    "Certifications": "categorical",
    "Score_Type": "categorical",
    "Score_Value": "numeric",
    "Technical_Skills_Parsed": "text",
    "Soft_Skills_Parsed": "text",
    "Skills_Gained_Parsed": "text",
}

EXPERIMENTS = {
    "Technical only": ["Technical_Skills_Parsed"],
    "Skills gained only": ["Skills_Gained_Parsed"],
    "Technical + gained": [
        "Technical_Skills_Parsed",
        "Skills_Gained_Parsed",
    ],
    "All skill fields": [
        "Technical_Skills_Parsed",
        "Soft_Skills_Parsed",
        "Skills_Gained_Parsed",
    ],
    "Education + specialization + skills": [
        "Education Level",
        "Specialization",
        "Technical_Skills_Parsed",
        "Skills_Gained_Parsed",
    ],
    "All features": list(FEATURE_TYPES.keys()),
}


def build_pipeline(features):

    transformers = []

    categorical = [
        f for f in features
        if FEATURE_TYPES[f] == "categorical"
    ]

    numeric = [
        f for f in features
        if FEATURE_TYPES[f] == "numeric"
    ]

    text_features = [
        f for f in features
        if FEATURE_TYPES[f] == "text"
    ]

    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ))

    if numeric:
        transformers.append((
            "numeric",
            StandardScaler(),
            numeric
        ))

    for feature in text_features:
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

    print("CAREERPATH FEATURE COMBINATION AUDIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    for name, features in EXPERIMENTS.items():

        print(f"\nTesting: {name}")

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
            )
        })

    results_df = pd.DataFrame(results).sort_values(
        "Macro_F1",
        ascending=False
    )

    output_file = REPORT_DIR / "feature_combination_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\n========== COMBINATION RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\nSaved report:", output_file)
    print("\nFEATURE COMBINATION AUDIT COMPLETED")


if __name__ == "__main__":
    main()
