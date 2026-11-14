
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.jobmatch.recommendation import recommend_jobs


app = FastAPI(
    title="SkillPath AI - JobMatch API",
    description="AI-powered job recommendation system",
    version="1.0.0"
)


class CandidateProfile(BaseModel):
    highest_qualification: str
    specialization: str
    experience_years: int = Field(ge=0)

    technical_skills: str
    soft_skills: str
    certifications: str

    employment_type_interest: str
    technical_skill_count: int = Field(ge=0)
    soft_skill_count: int = Field(ge=0)

    fields_of_interest: str


@app.get("/")
def home():
    return {
        "message": "Welcome to SkillPath AI JobMatch API",
        "module": "JobMatch"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "module": "JobMatch"
    }


@app.post("/jobmatch/recommend")
def job_recommendation(candidate: CandidateProfile):

    candidate_profile = {
        "Highest_Qualification": candidate.highest_qualification,
        "Specialization": candidate.specialization,
        "Experience_Years": candidate.experience_years,
        "Technical_Skills_Parsed": candidate.technical_skills,
        "Soft_Skills_Parsed": candidate.soft_skills,
        "Certifications": candidate.certifications,
        "Employment_Type_Interest": candidate.employment_type_interest,
        "Technical_Skill_Count": candidate.technical_skill_count,
        "Soft_Skill_Count": candidate.soft_skill_count,
        "Fields_of_Interest": candidate.fields_of_interest
    }

    try:
        recommendations = recommend_jobs(
            candidate_profile,
            top_n=5
        )

        return {
            "status": "success",
            "module": "JobMatch",
            "recommendations": recommendations.to_dict(
                orient="records"
            ),
            "disclaimer": (
                "Model scores are not guaranteed hiring probabilities."
            )
        }

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
