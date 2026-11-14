import pandas as pd
from pathlib import Path

DATA = Path("data/processed")

CAREER_INPUT = DATA / "career_dataset_structured.csv"
JOB_INPUT = DATA / "job_dataset_structured.csv"

CAREER_OUTPUT = DATA / "career_dataset_normalized.csv"
JOB_OUTPUT = DATA / "job_dataset_normalized.csv"


def main():
    career = pd.read_csv(CAREER_INPUT)
    job = pd.read_csv(JOB_INPUT)

    career["Recommended_Career_Original"] = career["Recommended Career"]

    career["Recommended Career"] = career["Recommended Career"].replace({
        "M.Sc / postgraduate specialization in a relevant life-science or healthcarefield":
        "M.Sc / postgraduate specialization in a relevant life-science or healthcare field"
    })

    career.to_csv(CAREER_OUTPUT, index=False)
    job.to_csv(JOB_OUTPUT, index=False)

    assert len(career) == 9000
    assert len(job) == 6000

    assert career["Recommended Career"].nunique() == 30
    assert job["Recommended_Job"].nunique() == 34

    assert career["Recommended_Career_Original"].nunique() == 30

    assert (
        career["Recommended Career"]
        == "M.Sc / postgraduate specialization in a relevant life-science or healthcare field"
    ).sum() == 351

    print("CATEGORY NORMALIZATION COMPLETED")
    print("CareerPath rows:", len(career))
    print("JobMatch rows:", len(job))
    print("CareerPath unique labels:", career["Recommended Career"].nunique())
    print("JobMatch unique labels:", job["Recommended_Job"].nunique())
    print("Corrected career label count:", (
        career["Recommended Career"]
        == "M.Sc / postgraduate specialization in a relevant life-science or healthcare field"
    ).sum())
    print("Original career labels preserved: YES")
    print("JobMatch labels unchanged: YES")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
