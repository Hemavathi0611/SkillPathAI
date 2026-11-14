import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def split_list(value):
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def validate_careerpath():
    print("\n========== CAREERPATH SKILL VALIDATION ==========")

    df = pd.read_csv(DATA / "career_dataset_structured.csv")

    assert len(df) == 9000, "Incorrect CareerPath row count!"

    source_columns = [
        "Skills",
        "Technical_Skills_Parsed",
        "Soft_Skills_Parsed",
        "Skills_Gained_Parsed"
    ]

    assert df[source_columns].notna().all().all(), \
        "Missing values detected in skill columns!"

    for index, row in df.iterrows():
        sections = row["Skills"].split("|")

        expected = [
            split_list(sections[0].split(":", 1)[1]),
            split_list(sections[1].split(":", 1)[1]),
            split_list(sections[2].split(":", 1)[1])
        ]

        actual = [
            split_list(row["Technical_Skills_Parsed"]),
            split_list(row["Soft_Skills_Parsed"]),
            split_list(row["Skills_Gained_Parsed"])
        ]

        if expected != actual:
            raise ValueError(
                f"Skill extraction mismatch at row {index}"
            )

    print("Rows verified:", len(df))
    print("Technical skill extraction: PASSED")
    print("Soft skill extraction: PASSED")
    print("Skills gained extraction: PASSED")


def validate_jobmatch():
    print("\n========== JOBMATCH SKILL VALIDATION ==========")

    df = pd.read_csv(DATA / "job_dataset_structured.csv")

    assert len(df) == 6000, "Incorrect JobMatch row count!"

    for index, row in df.iterrows():
        expected_technical = split_list(row["Technical_Skills"])
        expected_soft = split_list(row["Soft_Skills"])

        actual_technical = split_list(
            row["Technical_Skills_Parsed"]
        )
        actual_soft = split_list(
            row["Soft_Skills_Parsed"]
        )

        if expected_technical != actual_technical:
            raise ValueError(
                f"Technical skill mismatch at row {index}"
            )

        if expected_soft != actual_soft:
            raise ValueError(
                f"Soft skill mismatch at row {index}"
            )

        if len(expected_technical) != row["Technical_Skill_Count"]:
            raise ValueError(
                f"Technical skill count mismatch at row {index}"
            )

        if len(expected_soft) != row["Soft_Skill_Count"]:
            raise ValueError(
                f"Soft skill count mismatch at row {index}"
            )

    print("Rows verified:", len(df))
    print("Technical skill extraction: PASSED")
    print("Soft skill extraction: PASSED")
    print("Skill counts: PASSED")


if __name__ == "__main__":
    validate_careerpath()
    validate_jobmatch()
    print("\nALL STRUCTURED SKILL VALIDATIONS PASSED")
