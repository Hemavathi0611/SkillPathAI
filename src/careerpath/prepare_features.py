import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = DATA_DIR

INPUT_FILE = DATA_DIR / "career_dataset_normalized.csv"
OUTPUT_FILE = OUTPUT_DIR / "career_model_ready.csv"

TARGET = "Recommended Career"

FEATURES = [
    "Education Level",
    "Specialization",
    "Certifications",
    "Score_Type",
    "Score_Value",
    "Technical_Skills_Parsed",
    "Soft_Skills_Parsed",
    "Skills_Gained_Parsed",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_data = df[required_columns].copy()

    if model_data.isnull().any().any():
        raise ValueError("Missing values found in model-ready data.")

    if model_data[TARGET].nunique() != df[TARGET].nunique():
        raise ValueError("Target classes changed during feature preparation.")

    if len(model_data) != len(df):
        raise ValueError("Record count changed during feature preparation.")

    if TARGET in FEATURES:
        raise ValueError("Target leakage: target included in input features.")

    model_data.to_csv(OUTPUT_FILE, index=False)

    print("\nCAREERPATH FEATURE PREPARATION COMPLETED")
    print("Output file:", OUTPUT_FILE)
    print("Model-ready shape:", model_data.shape)
    print("Input features:", len(FEATURES))
    print("Target column:", TARGET)
    print("Target classes:", model_data[TARGET].nunique())
    print("Missing values:", int(model_data.isnull().sum().sum()))
    print("\nFeature columns:")
    for column in FEATURES:
        print("-", column)

    print("\nTarget distribution:")
    print(model_data[TARGET].value_counts().to_string())

    print("\nALL CAREERPATH CHECKS PASSED")


if __name__ == "__main__":
    main()
