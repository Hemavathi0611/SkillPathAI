import pandas as pd

career = pd.read_csv("data/raw/career_dataset.csv", encoding="utf-8-sig")
job = pd.read_csv("data/raw/job_dataset.csv", encoding="utf-8-sig")

def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

# CAREERPATH
section("1. CAREERPATH - CAREER RECOMMENDATIONS BY EDUCATION")

for education in sorted(career["Education Level"].unique()):
    subset = career[career["Education Level"] == education]

    print(f"\n{education} ({len(subset)} records)")
    print(subset["Recommended Career"].value_counts().head(8).to_string())

section("2. CAREERPATH - RECOMMENDATIONS BY SPECIALIZATION")

for specialization in sorted(career["Specialization"].unique()):
    subset = career[career["Specialization"] == specialization]

    print(f"\n{specialization} ({len(subset)} records)")
    print(subset["Recommended Career"].value_counts().head(5).to_string())

career_features = [
    "Education Level", "Specialization", "Skills",
    "Certifications", "CGPA/Percentage"
]

career_profile_counts = career.groupby(career_features, dropna=False)[
    "Recommended Career"
].nunique()

print("\nRepeated CareerPath profiles:")
print("Profiles appearing more than once:",
      int((career.groupby(career_features, dropna=False).size() > 1).sum()))

print("Repeated profiles with conflicting career labels:",
      int((career_profile_counts > 1).sum()))

# JOBMATCH
section("3. JOBMATCH - EXPERIENCE DISTRIBUTION BY RECOMMENDED JOB")

experience_table = pd.crosstab(
    job["Recommended_Job"],
    job["Experience_Years"]
)

print(experience_table.to_string())

section("4. JOBMATCH - REPEATED CANDIDATE PROFILES")

job_features = [
    "Highest_Qualification",
    "Specialization",
    "Experience_Years",
    "Technical_Skills",
    "Soft_Skills",
    "Certifications",
    "Fields_of_Interest",
    "Employment_Type_Interest",
    "Expected_Salary_INR_Annual",
    "Resume_Summary"
]

job_profile_counts = job.groupby(job_features, dropna=False)[
    "Recommended_Job"
].nunique()

print("Repeated JobMatch profiles:",
      int((job.groupby(job_features, dropna=False).size() > 1).sum()))

print("Repeated profiles with conflicting job labels:",
      int((job_profile_counts > 1).sum()))

section("5. FINAL AUDIT SUMMARY")
print("CareerPath records:", len(career))
print("JobMatch records:", len(job))
print("No source files were modified.")

