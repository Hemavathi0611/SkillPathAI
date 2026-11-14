import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text_columns(df, columns):
    """Normalize whitespace without changing the meaning of text."""
    for column in columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


def clean_careerpath():
    print("\n" + "=" * 65)
    print("CLEANING CAREERPATH AI")
    print("=" * 65)

    file_path = RAW_DIR / "career_dataset.csv"
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    original_rows = len(df)

    required_columns = [
        "Education Level",
        "Specialization",
        "Skills",
        "Certifications",
        "CGPA/Percentage",
        "Recommended Career"
    ]

    if list(df.columns) != required_columns:
        raise ValueError("CareerPath column names do not match the audited dataset.")

    text_columns = [
        "Education Level",
        "Specialization",
        "Skills",
        "Certifications",
        "Recommended Career"
    ]

    df = clean_text_columns(df, text_columns)

    # Identify missing certifications before filling them.
    missing_cert = df["Certifications"].isna()

    df["Certification_Status"] = missing_cert.map({
        True: "Not specified",
        False: "Provided"
    })

    # Use an explicit value for missing certifications.
    df["Certifications"] = df["Certifications"].fillna("Not specified")

    # Preserve the original score and identify its scale.
    df["CGPA/Percentage"] = pd.to_numeric(
        df["CGPA/Percentage"], errors="raise"
    )

    school_levels = ["Class 10", "Class 12"]

    df["Score_Type"] = df["Education Level"].map(
        lambda x: "Percentage" if x in school_levels else "CGPA"
    )

    df["Score_Value"] = df["CGPA/Percentage"]

    invalid_percentage = (
        (df["Score_Type"] == "Percentage") &
        (~df["Score_Value"].between(0, 100))
    )

    invalid_cgpa = (
        (df["Score_Type"] == "CGPA") &
        (~df["Score_Value"].between(0, 10))
    )

    if invalid_percentage.any() or invalid_cgpa.any():
        raise ValueError(
            "Invalid score values detected. Cleaning stopped for investigation."
        )

    # Do not delete records or modify recommendation labels.
    output_path = PROCESSED_DIR / "career_dataset_cleaned.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Original records:", original_rows)
    print("Cleaned records:", len(df))
    print("Missing certifications identified:", int(missing_cert.sum()))
    print("Score types:")
    print(df["Score_Type"].value_counts().to_string())
    print("Duplicate rows after normalization:", int(df.duplicated().sum()))
    print("Saved:", output_path)


def clean_jobmatch():
    print("\n" + "=" * 65)
    print("CLEANING JOBMATCH AI")
    print("=" * 65)

    file_path = RAW_DIR / "job_dataset.csv"
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    original_rows = len(df)

    required_columns = [
        "Highest_Qualification",
        "Specialization",
        "Experience_Years",
        "Technical_Skills",
        "Soft_Skills",
        "Certifications",
        "Fields_of_Interest",
        "Employment_Type_Interest",
        "Expected_Salary_INR_Annual",
        "Resume_Summary",
        "Recommended_Job"
    ]

    if list(df.columns) != required_columns:
        raise ValueError("JobMatch column names do not match the audited dataset.")

    text_columns = [
        "Highest_Qualification",
        "Specialization",
        "Technical_Skills",
        "Soft_Skills",
        "Certifications",
        "Fields_of_Interest",
        "Employment_Type_Interest",
        "Resume_Summary",
        "Recommended_Job"
    ]

    df = clean_text_columns(df, text_columns)

    # Validate numerical columns.
    df["Experience_Years"] = pd.to_numeric(
        df["Experience_Years"], errors="raise"
    )

    df["Expected_Salary_INR_Annual"] = pd.to_numeric(
        df["Expected_Salary_INR_Annual"], errors="raise"
    )

    if (df["Experience_Years"] < 0).any():
        raise ValueError("Negative experience values detected.")

    if (df["Expected_Salary_INR_Annual"] <= 0).any():
        raise ValueError("Invalid salary values detected.")

    # Flag the observed salary cap; do not modify the salary.
    df["Salary_At_Observed_Cap"] = (
        df["Expected_Salary_INR_Annual"] == 500000
    )

    # Preserve original recommendation labels and all records.
    output_path = PROCESSED_DIR / "job_dataset_cleaned.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Original records:", original_rows)
    print("Cleaned records:", len(df))
    print(
        "Salary values at observed cap:",
        int(df["Salary_At_Observed_Cap"].sum())
    )
    print(
        "Duplicate rows after normalization:",
        int(df.duplicated().sum())
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    clean_careerpath()
    clean_jobmatch()

    print("\n" + "=" * 65)
    print("FIRST CLEANING PASS COMPLETED")
    print("Original CSV files were not modified.")
    print("=" * 65)
