
from fastapi import FastAPI

from backend.careerpath_api import (
    StudentProfile,
    recommend_career
)

from backend.jobmatch_api import (
    CandidateProfile,
    job_recommendation
)


app = FastAPI(
    title="SkillPath AI",
    description=(
        "AI-powered Career Pathway and Job Recommendation System"
    ),
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to SkillPath AI",
        "modules": [
            "CareerPath",
            "JobMatch"
        ],
        "documentation": "/docs",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "SkillPath AI",
        "modules": {
            "CareerPath": "available",
            "JobMatch": "available"
        }
    }


@app.post("/careerpath/recommend")
def careerpath_endpoint(profile: StudentProfile):
    return recommend_career(profile)


@app.post("/jobmatch/recommend")
def jobmatch_endpoint(candidate: CandidateProfile):
    return job_recommendation(candidate)
