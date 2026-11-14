
import pandas as pd

from src.careerpath.recommendation import (
    career_model,
    recommend_eligible_careers
)


def create_student_profile(education, specialization):
    return pd.DataFrame([{
        "Education Level": education,
        "Specialization": specialization,
        "Certifications": "Python certification",
        "Score_Type": (
            "Percentage"
            if education in ["Class 10", "Class 12"]
            else "CGPA"
        ),
        "Score_Value": 90.0 if education in ["Class 10", "Class 12"] else 8.5,
        "Technical_Skills_Parsed": "Python | SQL | Data analysis",
        "Soft_Skills_Parsed": "Communication | Problem-solving",
        "Skills_Gained_Parsed": "Programming | Logical reasoning"
    }])


def test_model_loaded():
    assert len(career_model.classes_) == 30


# CLASS 10 TESTS

def test_class10_mpc_recommendations():
    profile = create_student_profile("Class 10", "MPC")

    results = recommend_eligible_careers(profile)

    assert not results.empty
    assert len(results) == 2
    assert results["Mapping Basis"].iloc[0] == "Class 10 + Interest"


def test_class10_bipc_recommendations():
    profile = create_student_profile("Class 10", "BiPC")

    results = recommend_eligible_careers(profile)

    assert not results.empty
    assert len(results) == 3
    assert results["Mapping Basis"].iloc[0] == "Class 10 + Interest"


def test_class10_commerce_recommendations():
    profile = create_student_profile("Class 10", "Commerce")

    results = recommend_eligible_careers(profile)

    assert not results.empty
    assert results["Mapping Basis"].iloc[0] == "Class 10 + Interest"


def test_class10_humanities_recommendations():
    profile = create_student_profile("Class 10", "Humanities")

    results = recommend_eligible_careers(profile)

    assert not results.empty
    assert results["Mapping Basis"].iloc[0] == "Class 10 + Interest"


def test_class10_unknown_interest():
    profile = create_student_profile("Class 10", "Not sure")

    results = recommend_eligible_careers(profile)

    assert not results.empty
    assert results["Mapping Basis"].iloc[0] == "Class 10 + Interest"


# EXISTING CLASS 12 TESTS

def test_class12_mpc_recommendations():
    profile = create_student_profile("Class 12", "MPC")
    results = recommend_eligible_careers(profile)

    assert len(results) == 2
    assert results["Mapping Basis"].iloc[0] == "Class 12 + Stream"


def test_class12_bipc_recommendations():
    profile = create_student_profile("Class 12", "BiPC")
    results = recommend_eligible_careers(profile)

    assert len(results) == 2


# DEGREE-LEVEL TEST

def test_btech_recommendations():
    profile = create_student_profile("B.Tech", "Computer Applications")
    results = recommend_eligible_careers(profile)

    assert len(results) == 2
    assert results["Mapping Basis"].iloc[0] == "Education + Specialization"


# VALIDATION TESTS

def test_unsupported_profile():
    profile = create_student_profile("Class 12", "Unknown Stream")
    results = recommend_eligible_careers(profile)

    assert "Message" in results.columns


def test_invalid_profile():
    try:
        recommend_eligible_careers(pd.DataFrame())
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for invalid profile"
