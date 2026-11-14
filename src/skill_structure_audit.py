import pandas as pd
from pathlib import Path
from collections import Counter

DATA = Path("data/processed")


def audit_careerpath():
    print("\n========== CAREERPATH SKILL STRUCTURE ==========")

    df = pd.read_csv(DATA / "career_dataset_cleaned.csv")

    section_counts = Counter()
    malformed_rows = []

    for index, value in df["Skills"].items():
        parts = str(value).split("|")

        section_counts[len(parts)] += 1

        if len(parts) != 3:
            malformed_rows.append((index, value))
            continue

        prefixes = [
            parts[0].strip().startswith("Technical:"),
            parts[1].strip().startswith("Soft:"),
            parts[2].strip().startswith("Skills gained:")
        ]

        if not all(prefixes):
            malformed_rows.append((index, value))

    print("Rows by number of pipe-separated sections:")
    for count, rows in sorted(section_counts.items()):
        print(f"{count} sections: {rows} rows")

    print("Malformed rows:", len(malformed_rows))

    if malformed_rows:
        print("\nFirst 10 malformed examples:")
        for index, value in malformed_rows[:10]:
            print(f"Row {index}: {value}")

    print("\nSample section extraction:")
    for value in df["Skills"].head(5):
        parts = str(value).split("|")

        if len(parts) == 3:
            print("\nTechnical:", parts[0].strip())
            print("Soft:", parts[1].strip())
            print("Skills gained:", parts[2].strip())


def audit_jobmatch():
    print("\n========== JOBMATCH SKILL VOCABULARY ==========")

    df = pd.read_csv(DATA / "job_dataset_cleaned.csv")

    for column in ["Technical_Skills", "Soft_Skills"]:
        vocabulary = set()

        for value in df[column].dropna():
            skills = str(value).split(";")

            for skill in skills:
                skill = skill.strip()
                if skill:
                    vocabulary.add(skill)

        print(f"\n{column}")
        print("Unique skill tokens:", len(vocabulary))
        print("First 40 skill tokens:")

        for skill in sorted(vocabulary)[:40]:
            print("-", skill)


if __name__ == "__main__":
    audit_careerpath()
    audit_jobmatch()
