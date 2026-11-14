import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def main():
    job = pd.read_csv(DATA / "job_dataset_normalized.csv")
    career = pd.read_csv(DATA / "career_dataset_normalized.csv")

    print("\n========== JOBMATCH RESUME SUMMARY AUDIT ==========")

    summary = job["Resume_Summary"].fillna("").str.lower()

    experience_phrases = (
        job["Experience_Years"].astype(str) + " years of experience"
    ).str.lower()

    matching_experience = sum(
        phrase in text
        for phrase, text in zip(experience_phrases, summary)
    )

    print("Total resume summaries:", len(job))
    print(
        "Summaries containing matching experience phrase:",
        matching_experience
    )

    print("\nExperience wording examples:")

    for years in sorted(job["Experience_Years"].unique()):
        examples = job.loc[
            job["Experience_Years"] == years,
            "Resume_Summary"
        ].head(1)

        if not examples.empty:
            print(f"{years} years:", examples.iloc[0])

    print("\n========== TARGET NAME IN SUMMARY AUDIT ==========")

    target_matches = 0

    for target in job["Recommended_Job"].unique():
        mask = summary.str.contains(
            target.lower(),
            regex=False,
            na=False
        )
        target_matches += int(mask.sum())

    print(
        "Total target-name matches across all labels:",
        target_matches
    )

    print("\n========== CAREERPATH SCORE AUDIT ==========")

    print(
        career.groupby("Score_Type")["Score_Value"]
        .agg(["count", "min", "max", "mean"])
        .to_string()
    )

    print("\n========== LEAKAGE AUDIT COMPLETED ==========")


if __name__ == "__main__":
    main()
