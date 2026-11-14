import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

career = pd.read_csv(DATA_DIR / "career_dataset.csv", encoding="utf-8-sig")
job = pd.read_csv(DATA_DIR / "job_dataset.csv", encoding="utf-8-sig")

def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

# CAREERPATH AUDIT
section("1. CAREERPATH - EDUCATION LEVELS")
print(career["Education Level"].value_counts(dropna=False).to_string())

section("2. CAREERPATH - SCORE STATISTICS BY EDUCATION")
print(
    career.groupby("Education Level")["CGPA/Percentage"]
    .agg(["count", "min", "max", "mean", "median"])
    .to_string()
)

section("3. CAREERPATH - SPECIALIZATIONS")
print(career["Specialization"].value_counts(dropna=False).to_string())

section("4. CAREERPATH - CERTIFICATION MISSING VALUES BY EDUCATION")
print(
    career.assign(Certification_Missing=career["Certifications"].isna())
    .groupby("Education Level")["Certification_Missing"]
    .agg(["sum", "count"])
    .to_string()
)

section("5. CAREERPATH - RECOMMENDED CAREER LABEL COUNTS")
print(career["Recommended Career"].value_counts().to_string())

# JOBMATCH AUDIT
section("6. JOBMATCH - QUALIFICATIONS")
print(job["Highest_Qualification"].value_counts().to_string())

section("7. JOBMATCH - RECOMMENDED JOB LABEL COUNTS")
print(job["Recommended_Job"].value_counts().to_string())

section("8. JOBMATCH - EXPERIENCE STATISTICS")
print(job["Experience_Years"].describe().to_string())

section("9. JOBMATCH - SALARY STATISTICS")
print(job["Expected_Salary_INR_Annual"].describe().to_string())

section("10. JOBMATCH - SALARY VALUE COUNTS")
print(job["Expected_Salary_INR_Annual"].value_counts().head(20).to_string())

# BOTH DATASETS: TEXT QUALITY
for name, df in [("CAREERPATH", career), ("JOBMATCH", job)]:
    section(f"{name} - TEXT QUALITY CHECK")

    text_columns = df.select_dtypes(include=["object", "str"]).columns

    for column in text_columns:
        values = df[column].dropna().astype(str)

        blank_count = values.str.strip().eq("").sum()
        whitespace_count = values.ne(values.str.strip()).sum()

        print(
            f"{column}: "
            f"blank={blank_count}, "
            f"leading/trailing whitespace={whitespace_count}"
        )

print("\nDETAILED AUDIT COMPLETED - ORIGINAL DATASETS NOT MODIFIED")
