from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.careerpath.recommendation import recommend_eligible_careers


app = FastAPI(
    title="SkillPath AI - CareerPath API",
    description="Personalized career pathway recommendation API",
    version="1.0.0"
)


class StudentProfile(BaseModel):
    education_level: str
    specialization: str
    certifications: str = "Not specified"
    score_type: str
    score_value: float = Field(ge=0)
    technical_skills: str = ""
    soft_skills: str = ""
    skills_gained: str = ""


@app.get("/")
def home():
    return {
        "message": "Welcome to SkillPath AI CareerPath API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "module": "CareerPath"
    }


@app.post("/careerpath/recommend")
def recommend_career(profile: StudentProfile):

    student_data = pd.DataFrame([{
        "Education Level": profile.education_level,
        "Specialization": profile.specialization,
        "Certifications": profile.certifications,
        "Score_Type": profile.score_type,
        "Score_Value": profile.score_value,
        "Technical_Skills_Parsed": profile.technical_skills,
        "Soft_Skills_Parsed": profile.soft_skills,
        "Skills_Gained_Parsed": profile.skills_gained
    }])

    try:
        recommendations = recommend_eligible_careers(
            student_data,
            top_n=5
        )

        if "Career" not in recommendations.columns:
            return {
                "status": "no_mapping",
                "message": recommendations.iloc[0]["Message"],
                "recommendations": []
            }

        return {
            "status": "success",
            "education_level": profile.education_level,
            "specialization": profile.specialization,
            "mapping_basis": recommendations["Mapping Basis"].iloc[0],
            "recommendations": recommendations.reset_index().to_dict(
                orient="records"
            ),
            "disclaimer": (
                "These are preliminary suggestions. "
                "Model scores are not admission or eligibility probabilities. "
                "Verify course requirements with the relevant institution."
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
