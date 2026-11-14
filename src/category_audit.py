import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def audit_column(df, column, dataset_name):
    print("\n" + "-" * 65)
    print(f"{dataset_name} | {column}")
    print("-" * 65)

    counts = df[column].value_counts(dropna=False)

    print("Unique values:", df[column].nunique(dropna=False))
    print("Missing values:", df[column].isna().sum())
    print("\nValue frequencies:")

    for value, count in counts.items():
        print(f"{repr(value)} : {count}")


def main():
    career = pd.read_csv(DATA / "career_dataset_structured.csv")
    job = pd.read_csv(DATA / "job_dataset_structured.csv")

    print("\n========== CAREERPATH CATEGORY AUDIT ==========")

    career_columns = [
        "Education Level",
        "Specialization",
        "Certifications",
        "Recommended Career"
    ]

    for column in career_columns:
        audit_column(career, column, "CareerPath")

    print("\n========== JOBMATCH CATEGORY AUDIT ==========")

    job_columns = [
        "Highest_Qualification",
        "Specialization",
        "Employment_Type_Interest",
        "Fields_of_Interest",
        "Certifications",
        "Recommended_Job"
    ]

    for column in job_columns:
        audit_column(job, column, "JobMatch")


if __name__ == "__main__":
    main()
