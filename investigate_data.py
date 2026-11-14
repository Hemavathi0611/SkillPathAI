import pandas as pd
from pathlib import Path

career = pd.read_csv("data/raw/career_dataset.csv", encoding="utf-8-sig")
job = pd.read_csv("data/raw/job_dataset.csv", encoding="utf-8-sig")

print("\n" + "=" * 65)
print("1. CAREERPATH - CERTIFICATION VALUE DISTRIBUTION")
print("=" * 65)

print(career["Certifications"].value_counts(dropna=False).to_string())

print("\nMissing certifications by education level:")
print(
    career[career["Certifications"].isna()]
    ["Education Level"]
    .value_counts()
    .to_string()
)

print("\nSample rows with missing certifications:")
print(
    career.loc[
        career["Certifications"].isna(),
        ["Education Level", "Specialization", "Skills",
         "CGPA/Percentage", "Recommended Career"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 65)
print("2. JOBMATCH - SALARY CAP INVESTIGATION")
print("=" * 65)

job["Salary_Capped"] = job["Expected_Salary_INR_Annual"].eq(500000)

print("\nSalary capped vs not capped:")
print(job["Salary_Capped"].value_counts().to_string())

print("\nSalary cap by recommended job:")
print(
    job.groupby("Recommended_Job")["Salary_Capped"]
    .agg(["sum", "count", "mean"])
    .sort_values("sum", ascending=False)
    .to_string()
)

print("\nSalary cap by experience:")
print(
    job.groupby("Experience_Years")["Salary_Capped"]
    .agg(["sum", "count", "mean"])
    .to_string()
)

print("\n" + "=" * 65)
print("3. JOBMATCH - EXPERIENCE BY RECOMMENDED JOB")
print("=" * 65)

print(
    job.groupby("Recommended_Job")["Experience_Years"]
    .agg(["count", "min", "median", "max"])
    .sort_values("median")
    .to_string()
)

print("\nINVESTIGATION COMPLETED - NO FILES MODIFIED")
