
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND MODEL LOADING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "careerpath_logistic_regression.ipynb.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"CareerPath model not found: {MODEL_PATH}"
    )

career_model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. EDUCATION-LEVEL PATHWAY MAPPING
# --------------------------------------------------

eligibility_mapping = {
    "B.Tech": [
        "M.Tech in Engineering Specialization",
        "M.Tech Computer Science / M.S. in Computer Science",
        "MBA / M.Com / postgraduate specialization in Management"
    ],

    "BCA": [
        "M.Tech Computer Science / M.S. in Computer Science",
        "MBA / M.Com / postgraduate specialization in Management"
    ],

    "B.Com": [
        "M.Com / MBA Finance / professional finance qualification pathway",
        "MBA / M.Com / postgraduate specialization in Management"
    ],

    "BBA": [
        "MBA / M.Com / postgraduate specialization in Management",
        "M.Com / MBA Finance / professional finance qualification pathway"
    ],

    "B.Sc": [
        "M.Sc / postgraduate specialization in a relevant life-science or healthcare field",
        "M.Sc Biotechnology / Microbiology / Life Sciences"
    ],

    "B.Pharm": [
        "M.Sc / postgraduate specialization in a relevant life-science or healthcare field",
        "M.Sc Biotechnology / Microbiology / Life Sciences"
    ],

    "B.Ag": [
        "M.Sc Agriculture / Horticulture / Food Science specialization"
    ],

    "B.Des": [
        "M.Des / postgraduate specialization in Design or Media"
    ],

    "BA": [
        "MA in a relevant humanities or social-science specialization"
    ]
}


# --------------------------------------------------
# 3. EDUCATION + SPECIALIZATION MAPPING
# --------------------------------------------------

specialization_mapping = {
    ("B.Tech", "Computer Applications"): [
        "M.Tech Computer Science / M.S. in Computer Science",
        "MBA / M.Com / postgraduate specialization in Management"
    ],

    ("B.Tech", "Engineering"): [
        "M.Tech in Engineering Specialization",
        "M.Tech Computer Science / M.S. in Computer Science",
        "MBA / M.Com / postgraduate specialization in Management"
    ],

    ("BCA", "Computer Applications"): [
        "M.Tech Computer Science / M.S. in Computer Science",
        "MBA / M.Com / postgraduate specialization in Management"
    ]
}


# --------------------------------------------------
# 4. CLASS 10 INTEREST-BASED PATHWAY MAPPING
# --------------------------------------------------

def get_class10_pathways(specialization):

    interest = str(specialization).strip().lower()

    # Mathematics / Engineering interest
    if any(word in interest for word in [
        "mpc", "math", "mathematics", "engineering", "physics"
    ]):
        return [
            "MPC → Intermediate (MPC) → B.Tech Computer Science and Engineering",
            "MPC → Intermediate (MPC) → B.Tech Mechanical/Electrical Engineering"
        ]

    # Biology / Healthcare interest
    elif any(word in interest for word in [
        "bipc", "biology", "medical", "healthcare", "medicine"
    ]):
        return [
            "BiPC → Intermediate (BiPC) → Healthcare-related degree pathway",
            "BiPC → Intermediate (BiPC) → B.Sc Life Sciences / Biotechnology",
            "BiPC / relevant vocational or agriculture pathway"
        ]

    # Commerce / Finance interest
    elif any(word in interest for word in [
        "cec", "mec", "commerce", "finance", "accounting", "business"
    ]):
        return [
            "CEC / MEC → Intermediate → B.Com / BBA pathway",
            "CEC / MEC → Intermediate → B.Com / Finance pathway"
        ]

    # Humanities interest
    elif any(word in interest for word in [
        "hec", "humanities", "history", "political science", "social science"
    ]):
        return [
            "HEC / Humanities → Intermediate → BA pathway"
        ]

    # Arts / Design interest
    elif any(word in interest for word in [
        "arts", "design", "creative", "drawing", "media"
    ]):
        return [
            "Arts / Humanities or a relevant design-oriented vocational pathway"
        ]

    # Agriculture interest
    elif "agriculture" in interest:
        return [
            "BiPC / relevant vocational or agriculture pathway"
        ]

    # General guidance when the student is unsure
    else:
        return [
            "MPC → Intermediate (MPC) → B.Tech Computer Science and Engineering",
            "MPC → Intermediate (MPC) → B.Tech Mechanical/Electrical Engineering",
            "BiPC → Intermediate (BiPC) → Healthcare-related degree pathway",
            "BiPC → Intermediate (BiPC) → B.Sc Life Sciences / Biotechnology",
            "CEC / MEC → Intermediate → B.Com / BBA pathway",
            "CEC / MEC → Intermediate → B.Com / Finance pathway",
            "HEC / Humanities → Intermediate → BA pathway",
            "Arts / Humanities or a relevant design-oriented vocational pathway"
        ]



# --------------------------------------------------
# 4. CLASS 12 STREAM MAPPING
# --------------------------------------------------

class12_mapping = {
    ("Class 12", "MPC"): [
        "MPC → Intermediate (MPC) → B.Tech Computer Science and Engineering",
        "MPC → Intermediate (MPC) → B.Tech Mechanical/Electrical Engineering"
    ],

    ("Class 12", "BiPC"): [
        "BiPC → Intermediate (BiPC) → Healthcare-related degree pathway",
        "BiPC → Intermediate (BiPC) → B.Sc Life Sciences / Biotechnology"
    ],

    ("Class 12", "CEC"): [
        "CEC / MEC → Intermediate → B.Com / Finance pathway"
    ],

    ("Class 12", "MEC"): [
        "CEC / MEC → Intermediate → B.Com / Finance pathway"
    ],

    ("Class 12", "HEC / Humanities"): [
        "HEC / Humanities → Intermediate → BA pathway"
    ],

    ("Class 12", "Arts/Humanities"): [
        "Arts / Humanities or a relevant design-oriented vocational pathway"
    ]
}


# --------------------------------------------------
# 5. RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_eligible_careers(student_profile, top_n=5):

    required_columns = [
        "Education Level",
        "Specialization",
        "Certifications",
        "Score_Type",
        "Score_Value",
        "Technical_Skills_Parsed",
        "Soft_Skills_Parsed",
        "Skills_Gained_Parsed"
    ]

    if not isinstance(student_profile, pd.DataFrame):
        raise TypeError("Student profile must be a pandas DataFrame.")

    if len(student_profile) != 1:
        raise ValueError("Provide exactly one student profile.")

    missing_columns = [
        column for column in required_columns
        if column not in student_profile.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing profile columns: {missing_columns}"
        )

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    student_profile = student_profile[required_columns].copy()

    education = student_profile.iloc[0]["Education Level"]
    specialization = student_profile.iloc[0]["Specialization"]

    profile_key = (education, specialization)

    
    if str(education).strip().lower() in ["class 10", "10", "10th"]:
        eligible_pathways = get_class10_pathways(specialization)
        mapping_type = "Class 10 + Interest"

    elif profile_key in class12_mapping:
        eligible_pathways = class12_mapping[profile_key]
        mapping_type = "Class 12 + Stream"

    elif profile_key in specialization_mapping:
        eligible_pathways = specialization_mapping[profile_key]
        mapping_type = "Education + Specialization"

    elif education in eligibility_mapping:
        eligible_pathways = eligibility_mapping[education]
        mapping_type = "Education only"

    else:
        return pd.DataFrame({
            "Message": [
                "No preliminary pathway mapping is configured for this profile."
            ]
        })


    probabilities = career_model.predict_proba(student_profile)[0]

    predictions = pd.DataFrame({
        "Career": career_model.classes_,
        "Confidence (%)": probabilities * 100
    })

    recommendations = predictions[
        predictions["Career"].isin(eligible_pathways)
    ].copy()

    recommendations = recommendations.sort_values(
        by="Confidence (%)",
        ascending=False
    ).head(top_n).reset_index(drop=True)

    recommendations.index += 1
    recommendations.index.name = "Rank"

    recommendations["Mapping Basis"] = mapping_type

    return recommendations


# --------------------------------------------------
# 6. PERSONALIZED REPORT FUNCTION
# --------------------------------------------------

def generate_recommendation_report(student_profile, recommendations):

    education = student_profile.iloc[0]["Education Level"]
    specialization = student_profile.iloc[0]["Specialization"]

    print("=" * 70)
    print("                       SKILLPATH AI")
    print("              PERSONALIZED CAREER REPORT")
    print("=" * 70)

    print(f"\nEducation Level : {education}")
    print(f"Specialization  : {specialization}")

    print("\nRECOMMENDED CAREER PATHWAYS")
    print("-" * 70)

    if "Career" not in recommendations.columns or recommendations.empty:
        print("No preliminary pathway recommendations are available.")

    else:
        for rank, (_, row) in enumerate(
            recommendations.iterrows(), start=1
        ):
            print(f"\n{rank}. {row['Career']}")
            print(f"   Model score: {row['Confidence (%)']:.4f}%")

        if "Mapping Basis" in recommendations.columns:
            print(
                "\nMapping Basis:",
                recommendations["Mapping Basis"].iloc[0]
            )

    print("\n" + "=" * 70)
    print("IMPORTANT INFORMATION")
    print("-" * 70)
    print("These recommendations are preliminary suggestions.")
    print("Model scores are not admission or eligibility probabilities.")
    print("Course eligibility depends on institutional requirements.")
    print("Verify prerequisites and admission rules before applying.")
    print("The dataset requires further real-world validation.")
    print("=" * 70)
