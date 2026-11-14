import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/career_model_ready.csv")
TARGET = "Recommended Career"

GROUPS = {
    "Complete Profile": [
        "Education Level",
        "Specialization",
        "Technical_Skills_Parsed",
        "Soft_Skills_Parsed",
        "Skills_Gained_Parsed",
    ],
    "Technical Skills": [
        "Technical_Skills_Parsed",
    ],
    "Education + Specialization": [
        "Education Level",
        "Specialization",
    ],
    "Specialization": [
        "Specialization",
    ],
}


def audit_group(df, columns, name):
    counts = (
        df.groupby(columns + [TARGET], dropna=False)
        .size()
        .unstack(fill_value=0)
    )

    group_sizes = counts.sum(axis=1)
    repeated = counts[group_sizes > 1]

    if repeated.empty:
        consistency = None
        conflicts = 0
    else:
        consistency = (
            repeated.max(axis=1).sum()
            / repeated.sum().sum()
        )
        conflicts = int(
            (repeated.gt(0).sum(axis=1) > 1).sum()
        )

    print(f"\n========== {name} ==========")
    print("Total feature groups:", len(counts))
    print("Repeated feature groups:", len(repeated))
    print("Conflicting groups:", conflicts)

    if consistency is not None:
        print(
            "Majority-label consistency:",
            f"{consistency:.4f} ({consistency * 100:.2f}%)"
        )
    else:
        print("Majority-label consistency: Not applicable")

    if not repeated.empty:
        conflicting = repeated[
            repeated.gt(0).sum(axis=1) > 1
        ].copy()

        if not conflicting.empty:
            conflicting["Total Records"] = conflicting.sum(axis=1)
            conflicting["Most Common Career"] = (
                conflicting.drop(columns=["Total Records"])
                .idxmax(axis=1)
            )

            print("\nTop 5 conflicting groups:")
            print(
                conflicting.sort_values(
                    "Total Records",
                    ascending=False
                ).head(5).to_string()
            )


def main():
    df = pd.read_csv(DATA_FILE)

    print("CAREERPATH LABEL CONSISTENCY AUDIT")
    print("Records:", len(df))

    for name, columns in GROUPS.items():
        audit_group(df, columns, name)

    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
