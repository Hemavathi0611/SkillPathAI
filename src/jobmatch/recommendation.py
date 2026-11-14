from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "jobmatch"

# Load saved model files
preprocessor = joblib.load(
    MODEL_DIR / "jobmatch_preprocessor.joblib"
)

model = joblib.load(
    MODEL_DIR / "jobmatch_random_forest.joblib"
)

feature_columns = joblib.load(
    MODEL_DIR / "jobmatch_features.joblib"
)


def recommend_jobs(candidate_profile, top_n=5):

    if not isinstance(candidate_profile, dict):
        raise TypeError("Candidate profile must be a dictionary.")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    missing_features = [
        feature for feature in feature_columns
        if feature not in candidate_profile
    ]

    if missing_features:
        raise ValueError(
            f"Missing candidate features: {missing_features}"
        )

    candidate_df = pd.DataFrame(
        [candidate_profile],
        columns=feature_columns
    )

    if candidate_df.isnull().any().any():
        raise ValueError("Candidate profile contains missing values.")

    processed_candidate = preprocessor.transform(candidate_df)

    probabilities = model.predict_proba(processed_candidate)[0]

    recommendations = pd.DataFrame({
        "Recommended_Job": model.classes_,
        "Model_Score": probabilities
    })

    recommendations = recommendations.sort_values(
        by="Model_Score",
        ascending=False
    ).head(min(top_n, len(model.classes_))).reset_index(drop=True)

    return recommendations


def generate_job_report(candidate_profile, top_n=5):

    recommendations = recommend_jobs(candidate_profile, top_n)

    report = [
        "=" * 60,
        "             SKILLPATH AI - JOBMATCH",
        "          PERSONALIZED JOB REPORT",
        "=" * 60,
        f"Qualification: {candidate_profile['Highest_Qualification']}",
        f"Specialization: {candidate_profile['Specialization']}",
        f"Experience: {candidate_profile['Experience_Years']} years",
        f"Field of Interest: {candidate_profile['Fields_of_Interest']}",
        "",
        "TOP JOB RECOMMENDATIONS",
        "-" * 60
    ]

    for index, row in recommendations.iterrows():
        report.append(
            f"{index + 1}. {row['Recommended_Job']} "
            f"(Model score: {row['Model_Score']:.4f})"
        )

    report.extend([
        "=" * 60,
        "Note: Model scores are not guaranteed hiring probabilities."
    ])

    return "\n".join(report)
