import pandas as pd

from pathlib import Path

DATA_FILE = Path("data/processed/career_model_ready.csv")

TARGET = "Recommended Career"

PROFILE_FEATURES = [
    "Education Level",
    "Specialization",
    "Technical_Skills_Parsed",
    "Soft_Skills_Parsed",
    "Skills_Gained_Parsed",
]


def audit_group(df, columns, name):
    grouped = df.groupby(columns, dropna=False)[TARGET].agg(
        records="size",
        unique_careers="nunique"
    )

    repeated = grouped[grouped["records"] > 1]

    if repeated.empty:
        consistency = 0.0
    else:
        consistency = (
            (repeated["records"] - 1).sum()
            / (repeated["records"].sum() - len(repeated))
            if repeated["records"].sum() > len(repeated)
            else 0.0
        )

    print(f"\n{name}")
    print("Unique feature groups:", len(grouped))
    print("Repeated feature groups:", len(repeated))
    print(
        "Repeated groups with conflicting careers:",
        int((repeated["unique_careers"] > 1).sum())
    )
    print("Repeated-group consistency:", round(consistency, 4))


def main():
    df = pd.read_csv(DATA_FILE)

    print("CAREERPATH DATASET PATTERN AUDIT")
    print("Records:", len(df))

    audit_group(
        df,
        PROFILE_FEATURES,
        "COMPLETE PROFILE PATTERNS"
    )

    audit_group(
        df,
        ["Technical_Skills_Parsed"],
        "TECHNICAL SKILL PATTERNS"
    )

    audit_group(
        df,
        ["Specialization"],
        "SPECIALIZATION PATTERNS"
    )

    print("\n========== SPECIALIZATION / CAREER DISTRIBUTION ==========")

    distribution = pd.crosstab(
        df["Specialization"],
        df[TARGET],
        normalize="index"
    )

    for specialization in distribution.index:
        top_career = distribution.loc[specialization].idxmax()
        top_percentage = distribution.loc[specialization].max() * 100

        print(
            f"{specialization}: "
            f"{top_career} ({top_percentage:.1f}%)"
        )

    print("\nPATTERN AUDIT COMPLETED")


if __name__ == "__main__":
    main()
