import pandas as pd
from pathlib import Path

DATA = Path("data/processed")


def main():
    career_source = pd.read_csv(DATA / "career_dataset_structured.csv")
    career_normalized = pd.read_csv(DATA / "career_dataset_normalized.csv")

    job_source = pd.read_csv(DATA / "job_dataset_structured.csv")
    job_normalized = pd.read_csv(DATA / "job_dataset_normalized.csv")

    # 1. Verify record counts.
    assert len(career_source) == len(career_normalized) == 9000
    assert len(job_source) == len(job_normalized) == 6000

    # 2. Verify original career labels were preserved.
    assert (
        career_normalized["Recommended_Career_Original"]
        == career_source["Recommended Career"]
    ).all()

    # 3. Verify career labels were not unintentionally changed.
    assert (
        career_normalized["Recommended Career"]
        == career_source["Recommended Career"]
    ).all()

    # 4. Verify all original CareerPath columns are unchanged.
    for column in career_source.columns:
        if column != "Recommended Career":
            assert career_normalized[column].equals(
                career_source[column]
            ), f"CareerPath column changed: {column}"

    # 5. Verify JobMatch is completely unchanged.
    assert job_source.equals(job_normalized)

    # 6. Verify category counts.
    assert career_normalized["Recommended Career"].nunique() == 30
    assert job_normalized["Recommended_Job"].nunique() == 34

    print("NORMALIZED DATA VALIDATION PASSED")
    print("CareerPath records: 9000")
    print("JobMatch records: 6000")
    print("CareerPath unique labels: 30")
    print("JobMatch unique labels: 34")
    print("Original career labels preserved: YES")
    print("CareerPath labels unchanged: YES")
    print("JobMatch dataset unchanged: YES")
    print("No unnecessary label correction: CONFIRMED")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
