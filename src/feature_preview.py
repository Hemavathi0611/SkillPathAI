import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def main():
    career = pd.read_csv(DATA / "career_dataset_normalized.csv")
    job = pd.read_csv(DATA / "job_dataset_normalized.csv")

    print("\n========== CAREERPATH FEATURE PREVIEW ==========")

    career_columns = [
        "Technical_Skills_Parsed",
        "Soft_Skills_Parsed",
        "Skills_Gained_Parsed",
        "Score_Type",
        "Score_Value",
        "Recommended Career"
    ]

    for column in career_columns:
        print(f"\n{column}")
        for value in career[column].head(3):
            print(repr(value))

    print("\n========== JOBMATCH FEATURE PREVIEW ==========")

    job_columns = [
        "Technical_Skills_Parsed",
        "Soft_Skills_Parsed",
        "Resume_Summary",
        "Experience_Years",
        "Recommended_Job"
    ]

    for column in job_columns:
        print(f"\n{column}")
        for value in job[column].head(3):
            print(repr(value))

    print("\n========== EXPERIENCE AND JOB LABEL AUDIT ==========")

    experience_audit = (
        job.groupby("Recommended_Job")["Experience_Years"]
        .agg(["count", "min", "max", "nunique"])
        .sort_values(["min", "max"])
    )

    print(experience_audit.to_string())

    print("\n========== FEATURE PREVIEW COMPLETED ==========")


if __name__ == "__main__":
    main()
