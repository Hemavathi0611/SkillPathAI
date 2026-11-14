import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

INPUT_FILE = DATA_DIR / "job_dataset_normalized.csv"

TARGET = "Recommended_Job"

BASE_FEATURES = [
    "Highest_Qualification",
    "Specialization",
    "Technical_Skills_Parsed",
    "Soft_Skills_Parsed",
    "Certifications",
    "Fields_of_Interest",
    "Employment_Type_Interest",
]

EXPERIENCE_FEATURE = "Experience_Years"


def validate_dataset(data, original, name):
    if len(data) != len(original):
        raise ValueError(f"{name}: record count changed.")

    if data.isnull().any().any():
        raise ValueError(f"{name}: missing values found.")

    if data[TARGET].nunique() != original[TARGET].nunique():
        raise ValueError(f"{name}: target classes changed.")

    if TARGET in data.drop(columns=[TARGET]).columns:
        raise ValueError(f"{name}: target leakage detected.")

    print(f"\n{name} validation passed.")
    print("Shape:", data.shape)
    print("Features:", len(data.columns) - 1)
    print("Target classes:", data[TARGET].nunique())
    print("Missing values:", int(data.isnull().sum().sum()))


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    required = BASE_FEATURES + [EXPERIENCE_FEATURE, TARGET]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model_a = df[BASE_FEATURES + [TARGET]].copy()

    model_b = df[
        BASE_FEATURES + [EXPERIENCE_FEATURE, TARGET]
    ].copy()

    validate_dataset(model_a, df, "MODEL A - WITHOUT EXPERIENCE")
    validate_dataset(model_b, df, "MODEL B - WITH EXPERIENCE")

    model_a.to_csv(
        DATA_DIR / "job_model_ready_without_experience.csv",
        index=False
    )

    model_b.to_csv(
        DATA_DIR / "job_model_ready_with_experience.csv",
        index=False
    )

    print("\nJOBMATCH FEATURE PREPARATION COMPLETED")
    print("Model A: 7 input features")
    print("Model B: 8 input features")
    print("Target:", TARGET)
    print("Records in each dataset:", len(df))
    print("\nExcluded: Resume_Summary, expected salary, salary-cap flag")
    print("\nALL JOBMATCH CHECKS PASSED")


if __name__ == "__main__":
    main()
