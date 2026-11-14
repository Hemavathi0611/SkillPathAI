
import streamlit as st
import requests
import pandas as pd
from textwrap import dedent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkillPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "Home",
    "career_results": None,
    "career_profile": None,
    "job_results": None,
    "job_profile": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PREMIUM BLUE THEME
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #F3F6FC;
    color: #172B4D;
}

[data-testid="stHeader"] {
    background-color: #F3F6FC;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #102A43 !important;
    font-weight: 750 !important;
}

h1 {
    font-size: 2.3rem !important;
}

h2 {
    font-size: 1.7rem !important;
}

h3 {
    font-size: 1.25rem !important;
}

p, li {
    color: #334B68;
    font-size: 15px;
    line-height: 1.65;
}

[data-testid="stMarkdownContainer"] p {
    color: #334B68;
}

[data-testid="stCaptionContainer"] {
    color: #52677F !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #102A43;
    border-right: 1px solid #244968;
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #DCE8F5 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Widget labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #183B63 !important;
    font-weight: 700 !important;
}

/* Text inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #A9BCD3 !important;
    border-radius: 9px !important;
    font-size: 15px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #A9BCD3 !important;
    border-radius: 9px !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #172B4D !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #FFFFFF !important;
}

[role="option"] {
    color: #172B4D !important;
    background-color: #FFFFFF !important;
}

[role="option"]:hover {
    background-color: #E5EEFF !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #1769E0 !important;
    color: #FFFFFF !important;
    border: 1px solid #1769E0 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 43px;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #0F52BA !important;
    border-color: #0F52BA !important;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #D7E2F0;
    border-radius: 13px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 14px;
    border: 1px solid #D7E2F0;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #52677F !important;
}

[data-testid="stMetricValue"] {
    color: #124A9C !important;
    font-weight: 800 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 10px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #D7E2F0 !important;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_backend():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


def section_heading(title, description=None):
    st.markdown(f"## {title}")

    if description:
        st.markdown(
            f"<p style='color:#52677F;'>{description}</p>",
            unsafe_allow_html=True
        )


def show_footer():
    footer_html = dedent("""
        <div style="
            text-align:center;
            padding:18px 0;
            color:#52677F;
            font-size:13px;
        ">
            <strong style="color:#164A8A;">SkillPath AI</strong>
            &nbsp; | &nbsp;
            AI-Powered Career Intelligence Platform
            <br>
            Built with Python, Machine Learning, FastAPI and Streamlit
        </div>
    """)

    st.divider()
    st.markdown(footer_html, unsafe_allow_html=True)


def display_error(error):
    st.error(
        "Unable to connect to the recommendation service. "
        "Please check whether FastAPI is running."
    )
    st.caption(f"Technical details: {error}")


def show_recommendations(data, name_column, score_column):

    if not data:
        st.info("No recommendations are available.")
        return

    results_df = pd.DataFrame(data)

    if name_column not in results_df.columns:
        st.warning("Unexpected recommendation format.")
        st.dataframe(results_df, use_container_width=True)
        return

    st.markdown("### Recommended Options")

    for index, row in results_df.iterrows():

        with st.container(border=True):

            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(
                    f"**{index + 1}. {row[name_column]}**"
                )

            with col2:

                if score_column in results_df.columns:

                    score = float(row[score_column])

                    if score > 1:
                        score_text = f"{score:.2f}"
                    else:
                        score_text = f"{score * 100:.2f}%"

                    score_html = dedent(f"""
                        <div style="
                            background:#E5EEFF;
                            color:#124A9C;
                            padding:8px;
                            border-radius:8px;
                            text-align:center;
                            font-weight:750;
                        ">
                            {score_text}
                        </div>
                    """)

                    st.markdown(
                        score_html,
                        unsafe_allow_html=True
                    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    sidebar_html = dedent("""
        <div style="padding:15px 0 22px 0;">
            <div style="
                font-size:29px;
                font-weight:850;
                color:#FFFFFF;
                letter-spacing:-1px;
            ">
                SkillPath <span style="color:#60A5FA;">AI</span>
            </div>
            <div style="
                color:#C5D8ED;
                font-size:12px;
                letter-spacing:1.2px;
            ">
                CAREER INTELLIGENCE PLATFORM
            </div>
        </div>
    """)

    st.markdown(sidebar_html, unsafe_allow_html=True)

    st.divider()

    pages = ["Home", "CareerPath", "JobMatch", "About"]

    selected_page = st.radio(
        "NAVIGATION",
        pages,
        index=pages.index(st.session_state.page)
    )

    st.session_state.page = selected_page

    st.divider()

    st.markdown("### System Status")

    if check_backend():
        st.success("Backend Connected")
    else:
        st.error("Backend Disconnected")

    st.caption("SkillPath AI Version 1.0.0")


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    hero_html = dedent("""
        <div style="
            background:linear-gradient(120deg,#102A43,#1769E0);
            padding:38px 35px;
            border-radius:17px;
            margin-bottom:25px;
        ">

            <div style="
                color:#DBEAFE;
                font-size:13px;
                font-weight:700;
                letter-spacing:2px;
            ">
                YOUR FUTURE STARTS HERE
            </div>

            <div style="
                color:#FFFFFF;
                font-size:36px;
                font-weight:850;
                margin-top:14px;
                line-height:1.4;
            ">
                Discover Your Next<br>
                Career Opportunity.
            </div>

            <div style="
                color:#FFFFFF;
                font-size:16px;
                margin-top:15px;
                line-height:1.7;
            ">
                Explore personalized career pathways and discover
                job opportunities powered by AI.
            </div>

        </div>
    """)

    st.html(hero_html)

    section_heading(
        "Explore SkillPath AI",
        "Choose the service that matches your current career stage."
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### 🎓 CareerPath")

            st.write(
                "Explore suitable education and career pathways "
                "based on your academic background, specialization "
                "and skills."
            )

            st.markdown(
                "**Ideal for:** Class 10, Class 12 and degree students."
            )

            if st.button(
                "Explore CareerPath →",
                use_container_width=True,
                key="home_career"
            ):
                st.session_state.page = "CareerPath"
                st.rerun()

    with col2:

        with st.container(border=True):

            st.markdown("### 💼 JobMatch")

            st.write(
                "Discover suitable job roles using your qualification, "
                "work experience, technical skills and career interests."
            )

            st.markdown(
                "**Ideal for:** Freshers and experienced candidates."
            )

            if st.button(
                "Explore JobMatch →",
                use_container_width=True,
                key="home_job"
            ):
                st.session_state.page = "JobMatch"
                st.rerun()

    st.markdown("")

    section_heading("How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        with st.container(border=True):
            st.markdown("### 01")
            st.markdown("**Enter Your Profile**")
            st.write("Provide your education, skills and interests.")

    with step2:
        with st.container(border=True):
            st.markdown("### 02")
            st.markdown("**AI Analysis**")
            st.write("Our trained machine learning models analyze your profile.")

    with step3:
        with st.container(border=True):
            st.markdown("### 03")
            st.markdown("**Get Recommendations**")
            st.write("Explore personalized career or job suggestions.")

    show_footer()


# ============================================================
# CAREERPATH PAGE
# ============================================================

elif st.session_state.page == "CareerPath":

    section_heading(
        "CareerPath Recommendation",
        "Find suitable education and career pathways based on your profile."
    )

    with st.container(border=True):

        st.markdown("### Academic Information")

        with st.form("careerpath_form"):

            col1, col2 = st.columns(2)

            with col1:

                education_level = st.selectbox(
                    "Education Level",
                    [
                        "Class 10", "Class 12",
                        "B.Tech", "B.E", "B.Sc",
                        "B.Com", "BBA", "BCA", "BA",
                        "M.Tech", "M.Sc", "MBA",
                        "MCA", "M.Com", "Other"
                    ]
                )

                specialization = st.text_input(
                    "Specialization / Stream",
                    placeholder="Example: MPC, BiPC, Computer Science"
                )

                score_type = st.selectbox(
                    "Score Type",
                    ["Percentage", "CGPA"]
                )

            with col2:

                score_value = st.number_input(
                    "Percentage / CGPA",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.1
                )

                certifications = st.text_input(
                    "Certifications",
                    placeholder="Example: Google Data Analytics"
                )

            st.markdown("### Skills and Knowledge")

            technical_skills = st.text_area(
                "Technical Skills",
                placeholder="Example: Python, SQL, Excel, Power BI",
                height=90
            )

            soft_skills = st.text_area(
                "Soft Skills",
                placeholder="Example: Communication, Teamwork, Leadership",
                height=80
            )

            skills_gained = st.text_area(
                "Skills Gained",
                placeholder="Example: Data Analysis, Problem Solving",
                height=80
            )

            submitted = st.form_submit_button(
                "Generate Career Recommendations",
                use_container_width=True
            )

    if submitted:

        if not specialization.strip():
            st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 14px;
        font-weight: 600;
        margin: 10px 0;
    ">
        ⚠ Please enter your specialization or stream.
    </div>
    """,
    unsafe_allow_html=True
)

        elif score_type == "Percentage" and score_value > 100:
            st.warning("Percentage cannot exceed 100.")

        elif score_type == "CGPA" and score_value > 10:
            st.warning("CGPA cannot exceed 10.")

        else:

            payload = {
                "education_level": education_level,
                "specialization": specialization.strip(),
                "certifications": certifications.strip() or "Not specified",
                "score_type": score_type,
                "score_value": score_value,
                "technical_skills": technical_skills.strip(),
                "soft_skills": soft_skills.strip(),
                "skills_gained": skills_gained.strip()
            }

            try:

                with st.spinner("Analyzing your academic profile..."):

                    response = requests.post(
                        f"{API_URL}/careerpath/recommend",
                        json=payload,
                        timeout=30
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state.career_profile = payload
                    st.session_state.career_results = result

                    st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 15px;
        font-weight: 600;
        margin: 12px 0;
    ">
        ✓ Career analysis completed!
    </div>
    """,
    unsafe_allow_html=True
)

                else:
                    st.error(f"Recommendation failed: {response.text}")

            except requests.RequestException as error:
                display_error(error)

    if st.session_state.career_results is not None:

        result = st.session_state.career_results
        profile = st.session_state.career_profile

        st.divider()

        section_heading("Your Career Report")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric("Education", profile["education_level"])

        with metric2:
            st.metric("Specialization", profile["specialization"])

        with metric3:
            st.metric(
                "Score",
                f"{profile['score_value']} {profile['score_type']}"
            )

        if result.get("status") == "success":

            st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 15px;
        font-weight: 600;
        margin: 12px 0;
    ">
        ✓ Your career recommendations are ready.
    </div>
    """,
    unsafe_allow_html=True
)

            if result.get("mapping_basis"):
                st.markdown(
    f"""
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 14px;
        font-weight: 600;
        margin: 10px 0;
    ">
        ℹ️ Recommendation basis: {result['mapping_basis']}
    </div>
    """,
    unsafe_allow_html=True
)

            show_recommendations(
                result.get("recommendations", []),
                "Career",
                "Confidence"
            )

            if result.get("disclaimer"):
                st.caption(result["disclaimer"])

        else:

            st.warning(
                result.get(
                    "message",
                    "No preliminary pathway mapping is configured."
                )
            )

    show_footer()


# ============================================================
# JOBMATCH PAGE
# ============================================================

elif st.session_state.page == "JobMatch":

    section_heading(
        "JobMatch Recommendation",
        "Discover job roles that match your qualifications, skills and interests."
    )

    with st.container(border=True):

        st.markdown("### Candidate Information")

        with st.form("jobmatch_form"):

            col1, col2 = st.columns(2)

            with col1:

                highest_qualification = st.selectbox(
                    "Highest Qualification",
                    [
                        "B.Tech", "B.E", "B.Sc", "B.Com",
                        "BBA", "BCA", "BA", "M.Tech",
                        "M.Sc", "MBA", "MCA", "M.Com", "Other"
                    ]
                )

                specialization = st.text_input(
                    "Specialization",
                    placeholder="Example: Computer Science"
                )

                experience_years = st.number_input(
                    "Experience (Years)",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1
                )

                employment_type_interest = st.selectbox(
                    "Employment Type Interest",
                    [
                        "Full-time",
                        "Part-time",
                        "Internship",
                        "Contract",
                        "Remote"
                    ]
                )

            with col2:

                fields_of_interest = st.selectbox(
                    "Field of Interest",
                    [
                        "Data Analytics",
                        "Software Development",
                        "AI/ML",
                        "Web Development",
                        "Cloud Computing",
                        "Cybersecurity",
                        "Business Analysis",
                        "Quality Assurance",
                        "UI/UX Design",
                        "Finance",
                        "Digital Marketing",
                        "Human Resources"
                    ]
                )

                certifications = st.text_input(
                    "Certifications",
                    placeholder="Example: Google Data Analytics"
                )

                technical_skill_count = st.number_input(
                    "Number of Technical Skills",
                    min_value=0,
                    max_value=100,
                    value=4,
                    step=1
                )

                soft_skill_count = st.number_input(
                    "Number of Soft Skills",
                    min_value=0,
                    max_value=100,
                    value=2,
                    step=1
                )

            st.markdown("### Skills Profile")

            technical_skills = st.text_area(
                "Technical Skills",
                placeholder="Example: Python, SQL, Excel, Power BI",
                height=90
            )

            soft_skills = st.text_area(
                "Soft Skills",
                placeholder="Example: Communication, Problem Solving, Teamwork",
                height=80
            )

            submitted = st.form_submit_button(
                "Find Matching Jobs",
                use_container_width=True
            )

    if submitted:

        if not specialization.strip():
            st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 14px;
        font-weight: 600;
        margin: 10px 0;
    ">
        ⚠ Please enter your specialization.
    </div>
    """,
    unsafe_allow_html=True
)

        elif not technical_skills.strip():
            st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 14px;
        font-weight: 600;
        margin: 10px 0;
    ">
        ⚠ Please enter your technical skills.
    </div>
    """,
    unsafe_allow_html=True
)

        else:

            payload = {
                "highest_qualification": highest_qualification,
                "specialization": specialization.strip(),
                "experience_years": experience_years,
                "technical_skills": technical_skills.strip(),
                "soft_skills": soft_skills.strip(),
                "certifications": certifications.strip() or "Not specified",
                "employment_type_interest": employment_type_interest,
                "technical_skill_count": technical_skill_count,
                "soft_skill_count": soft_skill_count,
                "fields_of_interest": fields_of_interest
            }

            try:

                with st.spinner("Finding suitable job opportunities..."):

                    response = requests.post(
                        f"{API_URL}/jobmatch/recommend",
                        json=payload,
                        timeout=30
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state.job_profile = payload
                    st.session_state.job_results = result

                    st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B2A4A, #164E8A);
        color: #FFFFFF;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
        font-size: 15px;
        font-weight: 600;
        margin: 12px 0;
    ">
        ✓ Job recommendations generated!
    </div>
    """,
    unsafe_allow_html=True
)

                else:
                    st.error(f"Recommendation failed: {response.text}")

            except requests.RequestException as error:
                display_error(error)

    if st.session_state.job_results is not None:

        result = st.session_state.job_results
        profile = st.session_state.job_profile

        st.divider()

        section_heading("Your JobMatch Report")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Qualification",
                profile["highest_qualification"]
            )

        with metric2:
            st.metric(
                "Experience",
                f"{profile['experience_years']} years"
            )

        with metric3:
            st.metric(
                "Interest",
                profile["fields_of_interest"]
            )

        show_recommendations(
            result.get("recommendations", []),
            "Recommended_Job",
            "Model_Score"
        )

        if result.get("disclaimer"):
            st.caption(result["disclaimer"])

    show_footer()


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    section_heading(
        "About SkillPath AI",
        "An AI-powered platform designed to support education and career decisions."
    )

    with st.container(border=True):

        st.markdown("### Our Vision")

        st.write(
            "SkillPath AI aims to help students and job seekers "
            "make informed career decisions through personalized "
            "recommendations powered by machine learning."
        )

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### 🎓 CareerPath")

            st.write(
                "Provides preliminary education and career pathway "
                "suggestions based on academic qualifications, "
                "specialization, scores and skills."
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 💼 JobMatch")

            st.write(
                "Recommends suitable job roles based on qualifications, "
                "experience, technical skills, soft skills and interests."
            )

    st.markdown("")

    with st.container(border=True):

        st.markdown("### Technology Stack")

        tech1, tech2, tech3 = st.columns(3)

        with tech1:
            st.markdown("**Programming**")
            st.write("Python")
            st.write("Pandas")
            st.write("NumPy")

        with tech2:
            st.markdown("**Machine Learning**")
            st.write("Scikit-learn")
            st.write("TF-IDF")
            st.write("Random Forest")

        with tech3:
            st.markdown("**Application**")
            st.write("FastAPI")
            st.write("Streamlit")
            st.write("Joblib")

    st.markdown("")

    with st.container(border=True):

        st.markdown("### Important Information")

        st.write(
            "The current recommendation models were trained using "
            "synthetic datasets. Their evaluation scores reflect "
            "performance on those datasets and do not guarantee "
            "real-world career suitability, admission eligibility "
            "or hiring outcomes."
        )

        st.info(
            "Use recommendations as preliminary guidance. Always "
            "verify education requirements and job criteria with "
            "the relevant institutions or employers."
        )

    show_footer()
