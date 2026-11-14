import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def inspect_dataset(file_name, target, dataset_name):
    df = pd.read_csv(DATA / file_name)

    print("\n" + "=" * 65)
    print(dataset_name)
    print("=" * 65)

    print("Dataset shape:", df.shape)
    print("\nColumn inventory:")

    for column in df.columns:
        role = "TARGET" if column == target else "FEATURE / METADATA"

        print(
            f"{column:<35} "
            f"{str(df[column].dtype):<12} "
            f"Unique: {df[column].nunique():<6} "
            f"Role: {role}"
        )

    print("\nTarget distribution:")
    print(df[target].value_counts().to_string())


def main():
    inspect_dataset(
        "career_dataset_normalized.csv",
        "Recommended Career",
        "CAREERPATH AI"
    )

    inspect_dataset(
        "job_dataset_normalized.csv",
        "Recommended_Job",
        "JOBMATCH AI"
    )


if __name__ == "__main__":
    main()
