import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def validate_careerpath():
    print("\n========== CAREERPATH VALIDATION ==========")

    raw = pd.read_csv(RAW / "career_dataset.csv")
    clean = pd.read_csv(PROCESSED / "career_dataset_cleaned.csv")

    assert len(raw) == len(clean), "CareerPath row count changed!"
    assert raw["Recommended Career"].equals(clean["Recommended Career"]), \
        "Career recommendation labels changed!"

    assert clean["Certifications"].notna().all(), \
        "Missing certifications remain!"

    assert clean["Score_Type"].notna().all(), \
        "Missing score types!"

    assert clean["Score_Value"].notna().all(), \
        "Missing score values!"

    assert (clean.loc[
        clean["Score_Type"] == "Percentage", "Score_Value"
    ].between(0, 100)).all(), "Invalid percentage values!"

    assert (clean.loc[
        clean["Score_Type"] == "CGPA", "Score_Value"
    ].between(0, 10)).all(), "Invalid CGPA values!"

    print("Rows preserved:", len(clean))
    print("Recommendation labels unchanged: YES")
    print("Missing certifications remaining:",
          clean["Certifications"].isna().sum())
    print("Score type counts:")
    print(clean["Score_Type"].value_counts().to_string())
    print("CareerPath validation: PASSED")


def validate_jobmatch():
    print("\n========== JOBMATCH VALIDATION ==========")

    raw = pd.read_csv(RAW / "job_dataset.csv")
    clean = pd.read_csv(PROCESSED / "job_dataset_cleaned.csv")

    assert len(raw) == len(clean), "JobMatch row count changed!"
    assert raw["Recommended_Job"].equals(clean["Recommended_Job"]), \
        "Job recommendation labels changed!"

    assert clean["Experience_Years"].notna().all(), \
        "Missing experience values!"

    assert clean["Expected_Salary_INR_Annual"].notna().all(), \
        "Missing salary values!"

    expected_flag = (
        clean["Expected_Salary_INR_Annual"] == 500000
    )

    assert clean["Salary_At_Observed_Cap"].equals(expected_flag), \
        "Salary cap flag is incorrect!"

    print("Rows preserved:", len(clean))
    print("Recommendation labels unchanged: YES")
    print("Missing values by column:")
    print(clean.isna().sum().to_string())
    print("Salary cap flags:", int(expected_flag.sum()))
    print("JobMatch validation: PASSED")


if __name__ == "__main__":
    validate_careerpath()
    validate_jobmatch()
    print("\nALL VALIDATION CHECKS PASSED")
