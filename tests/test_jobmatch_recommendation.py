
import pytest
import pandas as pd

from src.jobmatch.recommendation import recommend_jobs


@pytest.fixture
def valid_candidate():
    return {
        "Highest_Qualification": "B.Tech",
        "Specialization": "Computer Science",
        "Experience_Years": 0,
        "Technical_Skills_Parsed": "Python; SQL; Excel; Power BI; Pandas",
        "Soft_Skills_Parsed": "Communication; Problem Solving; Teamwork",
        "Certifications": "Google Data Analytics",
        "Employment_Type_Interest": "Full-time",
        "Technical_Skill_Count": 5,
        "Soft_Skill_Count": 3,
        "Fields_of_Interest": "Data Analytics"
    }


def test_valid_job_recommendations(valid_candidate):
    result = recommend_jobs(valid_candidate, top_n=5)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
    assert "Recommended_Job" in result.columns
    assert "Model_Score" in result.columns


def test_recommendations_are_sorted(valid_candidate):
    result = recommend_jobs(valid_candidate, top_n=5)

    scores = result["Model_Score"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_missing_features():
    invalid_candidate = {
        "Highest_Qualification": "B.Tech",
        "Specialization": "Computer Science"
    }

    with pytest.raises(ValueError, match="Missing candidate features"):
        recommend_jobs(invalid_candidate)


def test_invalid_candidate_type():
    with pytest.raises(TypeError, match="dictionary"):
        recommend_jobs("invalid candidate")


def test_invalid_top_n(valid_candidate):
    with pytest.raises(ValueError, match="positive integer"):
        recommend_jobs(valid_candidate, top_n=0)


def test_negative_top_n(valid_candidate):
    with pytest.raises(ValueError, match="positive integer"):
        recommend_jobs(valid_candidate, top_n=-1)
