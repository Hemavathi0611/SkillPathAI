# SKILLPATH AI
## Career Pathway and Job Recommendation System

**Final Year Project – Technical Report**

**Team:**  Harini Sarva, Hemavathi N, Vishnu Vardhan, Sathwik Reddy, Kartikeya
**Degree:** Bachelor of Technology  
**Branch:** Computer Science and Engineering  
**Academic Year:** 2026–2027

---

## 1. Abstract

SkillPath AI is an AI-powered Career Pathway and Job Recommendation System designed to help students explore suitable educational pathways and help job seekers identify relevant employment opportunities.

The system consists of two modules: CareerPath and JobMatch.

CareerPath analyzes academic qualifications, specialization, marks, certifications and skills to suggest suitable educational pathways. JobMatch analyzes qualifications, work experience, technical skills, soft skills, certifications and career interests to recommend relevant job roles.

The application uses Machine Learning and Natural Language Processing techniques, with a FastAPI backend and Streamlit frontend.

The objective is to provide an accessible, interactive and data-driven career guidance platform.

## 2. Problem Statement

Students and fresh graduates often face difficulties when choosing educational pathways and identifying suitable job opportunities.

Traditional career guidance may not sufficiently consider an individual's academic background, technical skills, certifications and interests together.

SkillPath AI addresses this problem by developing an integrated recommendation platform that analyzes user profiles and provides personalized suggestions.

## 3. Project Objectives

1. Develop an intelligent educational pathway recommendation system.
2. Recommend suitable job roles based on candidate profiles.
3. Analyze academic qualifications and skill information.
4. Apply Machine Learning and NLP techniques.
5. Develop an interactive web application.
6. Integrate recommendation models through REST APIs.
7. Evaluate model performance using standard classification metrics.

## 4. System Modules

### 4.1 CareerPath

CareerPath recommends suitable educational pathways based on:

- Education level
- Academic specialization or stream
- CGPA or percentage
- Technical skills
- Soft skills
- Certifications
- Skills gained

**Dataset:** 9,000 records  
**Target categories:** 30  
**Algorithm:** Logistic Regression

### 4.2 JobMatch

JobMatch recommends relevant job roles based on:

- Highest qualification
- Specialization
- Experience in years
- Technical skills
- Soft skills
- Certifications
- Employment type preference
- Technical and soft skill counts
- Fields of interest

**Dataset:** 6,000 records  
**Target categories:** 34  
**Algorithm:** Random Forest Classifier

## 5. Dataset Description

Two datasets were used.

| Dataset | Records | Target classes |
|---|---:|---:|
| CareerPath | 9,000 | 30 |
| JobMatch | 6,000 | 34 |
| Total | 15,000 | — |

Data preprocessing included:

- Missing-value handling
- Duplicate and blank-value checks
- Text normalization
- Skill parsing
- Feature extraction
- Categorical encoding
- Numerical feature scaling
- Train-test splitting

The datasets used in this project are synthetic datasets.

## 6. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the datasets and identify patterns.

The analysis included:

- Missing-value analysis
- Duplicate-record analysis
- Target-class distribution
- Education-level distribution
- Academic score analysis
- Experience distribution
- Skill distribution
- Field-of-interest analysis
- Feature relationships

Python libraries used included Pandas, NumPy, Matplotlib and Seaborn.

## 7. Feature Engineering

### CareerPath

The following features were used:

- Education Level
- Specialization
- Certifications
- Score Type
- Score Value
- Technical Skills
- Soft Skills
- Skills Gained

TF-IDF was applied to textual skill features. One-Hot Encoding and numerical scaling were used for other features.

### JobMatch

The model uses qualification, specialization, experience, skill information, certifications, employment preferences and fields of interest.

Feature engineering included:

- TF-IDF text vectorization
- One-Hot Encoding
- Standard Scaling
- Skill-count features
- Experience features

## 8. Machine Learning Methodology

### CareerPath Model

Logistic Regression was selected as the final model.

Evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 99.67% |
| Balanced Accuracy | 99.68% |
| Macro F1-score | 99.68% |

The model was evaluated using an 80:20 stratified train-test split and cross-validation.

### JobMatch Model

Random Forest Classifier was selected as the final model.

Evaluation results using experience and fields of interest:

| Metric | Score |
|---|---:|
| Accuracy | 99.75% |
| Balanced Accuracy | 99.32% |
| Macro F1-score | 99.50% |
| Weighted F1-score | 99.74% |

Five-fold cross-validation was also performed on the Random Forest pipeline.

**Important evaluation limitation:** These high scores reflect patterns in the synthetic datasets. They should not be interpreted as evidence of equivalent performance on real-world student or recruitment data.

## 9. System Architecture

The application follows a modular architecture.

**Frontend:** Streamlit  
**Backend:** FastAPI  
**Machine Learning:** Scikit-learn  
**Data Processing:** Pandas and NumPy  
**Model Persistence:** Joblib

Application workflow:

1. User opens the Streamlit application.
2. User selects CareerPath or JobMatch.
3. User enters their profile information.
4. Streamlit sends a request to the FastAPI backend.
5. The backend validates the input.
6. The recommendation module loads and uses the trained model.
7. The model generates ranked recommendations.
8. Results are returned to the frontend.
9. Streamlit displays the recommendation report.

## 10. API Development

The combined FastAPI application provides the following endpoints:

| HTTP Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application information |
| GET | `/health` | Health check |
| POST | `/careerpath/recommend` | Career recommendations |
| POST | `/jobmatch/recommend` | Job recommendations |

Swagger UI is available at:

`http://127.0.0.1:8000/docs`

## 11. Frontend Development

The frontend was developed using Streamlit.

The application includes:

- Home page
- CareerPath page
- JobMatch page
- About page
- Backend connection status
- Interactive input forms
- Recommendation reports
- Model-score displays

The interface uses a premium blue-themed design.

## 12. Testing and Validation

Automated testing was performed using Pytest.

| Module | Tests | Result |
|---|---:|---|
| CareerPath | 6 | Passed |
| JobMatch | 6 | Passed |
| Total | 12 | Passed |

Additional validation included:

- Model loading
- Recommendation generation
- Invalid-input handling
- Missing-feature handling
- API endpoint testing
- Frontend integration testing
- Python syntax compilation

**Final automated test result: 12/12 passed.**

## 13. Project Screenshots

Application screenshots are available in:

`docs/screenshots/`

The screenshots include:

1. Home dashboard
2. CareerPath recommendation
3. JobMatch recommendation
4. FastAPI Swagger documentation

## 14. Limitations

- The current datasets are synthetic.
- Real-world generalization has not yet been established.
- Career recommendations do not replace official admission eligibility checks.
- Job recommendations do not guarantee employment.
- Model scores are not admission or hiring probabilities.
- Live job-market integration is not currently implemented.

## 15. Future Enhancements

1. Resume parsing and analysis.
2. Advanced NLP and Deep Learning integration.
3. Real-time job-market data integration.
4. Explainable AI recommendations.
5. User authentication and profile management.
6. Cloud deployment.
7. Integration with verified educational and recruitment datasets.
8. Continuous model evaluation using real-world feedback.

## 16. Conclusion

SkillPath AI demonstrates the application of Machine Learning, NLP, data preprocessing, REST API development and interactive web technologies to career guidance.

The integration of CareerPath and JobMatch provides a unified platform for exploring educational pathways and job opportunities.

The project successfully implements two recommendation modules, a combined FastAPI backend and a Streamlit frontend. All 12 automated tests passed during final verification.

The project provides a foundation for further development using larger, verified real-world datasets and advanced recommendation techniques.

---

**Project:** SkillPath AI – Career Pathway & Job Recommendation System
**Developer:** Harini Sarva, Hemavathi N, Vishnu Vardhan, Sathwik Reddy, Kartikeya
**Department:** Computer Science and Engineering  
**Academic Year:** 2026–2027