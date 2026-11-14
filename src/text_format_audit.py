import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def inspect_column(df, column, title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    values = df[column].dropna().astype(str)

    print("Total records:", len(values))
    print("Unique complete values:", values.nunique())

    print("\nMost common complete values:")
    print(values.value_counts().head(15).to_string())

    separators = ["|", ",", ";", "/", "&"]

    print("\nSeparator frequency:")
    for separator in separators:
        count = values.str.count(
            __import__("re").escape(separator)
        ).sum()
        print(repr(separator), ":", int(count))

    print("\nSample values:")
    for value in values.drop_duplicates().head(10):
        print("-", value)


def main():
    career = pd.read_csv(DATA / "career_dataset_cleaned.csv")
    job = pd.read_csv(DATA / "job_dataset_cleaned.csv")

    print("\nCAREERPATH TEXT AUDIT")

    inspect_column(
        career, "Skills", "CareerPath Skills"
    )

    inspect_column(
        career, "Specialization", "CareerPath Specialization"
    )

    inspect_column(
        career, "Certifications", "CareerPath Certifications"
    )

    print("\nJOBMATCH TEXT AUDIT")

    for column in [
        "Technical_Skills",
        "Soft_Skills",
        "Fields_of_Interest",
        "Certifications"
    ]:
        inspect_column(job, column, "JobMatch " + column)


if __name__ == "__main__":
    main()
