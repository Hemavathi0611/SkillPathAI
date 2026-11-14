import pandas as pd
import re
from pathlib import Path

DATA = Path("data/processed")


def split_skill_list(value):
    """Split semicolon-separated skills without changing skill names."""
    if pd.isna(value):
        return []

    return [
        re.sub(r"\s+", " ", skill).strip()
        for skill in str(value).split(";")
        if skill.strip()
    ]


def structure_careerpath():
    print("\n========== STRUCTURING CAREERPATH ==========")

    path = DATA / "career_dataset_cleaned.csv"
    df = pd.read_csv(path)

    original_rows = len(df)

    technical_list = []
    soft_list = []
    gained_list = []

    for index, value in df["Skills"].items():
        sections = str(value).split("|")

        if len(sections) != 3:
            raise ValueError(
                f"Invalid section count at row {index}"
            )

        prefixes = [
            "Technical:",
            "Soft:",
            "Skills gained:"
        ]

        parsed = []

        for section, prefix in zip(sections, prefixes):
            section = section.strip()

            if not section.startswith(prefix):
                raise ValueError(
                    f"Invalid section prefix at row {index}: {section}"
                )

            content = section[len(prefix):].strip()
            parsed.append(split_skill_list(content))

        technical_list.append(parsed[0])
        soft_list.append(parsed[1])
        gained_list.append(parsed[2])

    # Preserve the original Skills column.
    df["Technical_Skills_Parsed"] = [
        "; ".join(skills) for skills in technical_list
    ]

    df["Soft_Skills_Parsed"] = [
        "; ".join(skills) for skills in soft_list
    ]

    df["Skills_Gained_Parsed"] = [
        "; ".join(skills) for skills in gained_list
    ]

    df["Technical_Skill_Count"] = [
        len(skills) for skills in technical_list
    ]

    df["Soft_Skill_Count"] = [
        len(skills) for skills in soft_list
    ]

    df["Skills_Gained_Count"] = [
        len(skills) for skills in gained_list
    ]

    assert len(df) == original_rows
    assert df[
        [
            "Technical_Skills_Parsed",
            "Soft_Skills_Parsed",
            "Skills_Gained_Parsed"
        ]
    ].notna().all().all()

    output = DATA / "career_dataset_structured.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print("Rows preserved:", len(df))
    print("Technical skills extracted:", len(technical_list))
    print("Soft skills extracted:", len(soft_list))
    print("Skills gained extracted:", len(gained_list))
    print("Saved:", output)

    print("\nSample structured record:")
    print(df[
        [
            "Technical_Skills_Parsed",
            "Soft_Skills_Parsed",
            "Skills_Gained_Parsed"
        ]
    ].iloc[0].to_string())


def structure_jobmatch():
    print("\n========== STRUCTURING JOBMATCH ==========")

    path = DATA / "job_dataset_cleaned.csv"
    df = pd.read_csv(path)

    original_rows = len(df)

    technical = df["Technical_Skills"].apply(split_skill_list)
    soft = df["Soft_Skills"].apply(split_skill_list)

    # Keep original columns unchanged.
    df["Technical_Skills_Parsed"] = technical.apply(
        lambda skills: "; ".join(skills)
    )

    df["Soft_Skills_Parsed"] = soft.apply(
        lambda skills: "; ".join(skills)
    )

    df["Technical_Skill_Count"] = technical.apply(len)
    df["Soft_Skill_Count"] = soft.apply(len)

    assert len(df) == original_rows
    assert df["Technical_Skills_Parsed"].notna().all()
    assert df["Soft_Skills_Parsed"].notna().all()

    output = DATA / "job_dataset_structured.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print("Rows preserved:", len(df))
    print("Technical skill lists extracted:", len(technical))
    print("Soft skill lists extracted:", len(soft))
    print("Saved:", output)

    print("\nSample structured record:")
    print(df[
        [
            "Technical_Skills_Parsed",
            "Soft_Skills_Parsed",
            "Technical_Skill_Count",
            "Soft_Skill_Count"
        ]
    ].iloc[0].to_string())


if __name__ == "__main__":
    structure_careerpath()
    structure_jobmatch()

    print("\nSKILL STRUCTURING COMPLETED")
