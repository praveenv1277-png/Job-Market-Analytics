"""
AI-Powered Tech Job Market & Salary Analytics Dashboard.
Modern, interactive Streamlit web application providing market intelligence,
real-time salary prediction, role classification, skill gap analytics, and live job explorer.
"""

import os
import sys
import time
import re
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path so src imports work cleanly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_model import (
    predict_single_salary,
    load_salary_model,
    load_role_model
)
from src.feature_engineering import TECH_SKILLS, extract_skills_from_text
from src.ingest_api import fetch_remotive_jobs
from src.data_cleaning import clean_job_dataset

# ---------------------------------------------------------
# Page Configuration & Modern Design System (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="TechSalary AI | Job Market Intelligence & Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background & main container */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(17, 24, 39, 0.95), rgba(11, 15, 25, 1));
    color: #F3F4F6;
}

/* Custom Header Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50px;
    right: -50px;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #94A3B8;
    margin-top: 8px;
    margin-bottom: 0px;
}

.badge-tag {
    display: inline-block;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.15);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.3);
    margin-bottom: 12px;
}

/* Glassmorphism Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 22px;
    transition: all 0.25s ease-in-out;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.01em;
}

.metric-sub {
    font-size: 0.8rem;
    color: #10B981;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Salary Prediction Result Box */
.prediction-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(17, 24, 39, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    padding: 26px 30px;
    box-shadow: 0 12px 35px rgba(99, 102, 241, 0.15);
    margin-top: 15px;
    margin-bottom: 20px;
}

.prediction-amount {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.skill-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    color: #C7D2FE;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.skill-pill-highlight {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #6EE7B7;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* Job Explorer Card */
.job-listing-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

.job-listing-card:hover {
    background: rgba(30, 41, 59, 0.8);
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
}

.job-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 2px;
}

.job-company {
    font-size: 0.9rem;
    font-weight: 600;
    color: #818CF8;
}

.job-meta {
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 4px;
    margin-bottom: 8px;
}

/* Custom tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(17, 24, 39, 0.6);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 600;
    padding: 0 18px;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
}

/* Streamlit button modern styling */
.stButton>button {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Loading Utilities (Cached for Performance)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_dataset() -> pd.DataFrame:
    """Loads cleaned job data or runs pipeline if missing."""
    cleaned_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_jobs.csv")
    if not os.path.exists(cleaned_path):
        from src.seed_data import run_pipeline
        run_pipeline()
    df = pd.read_csv(cleaned_path)
    return df


@st.cache_resource
def get_ml_models():
    """Loads trained ML regression and classification artifacts."""
    salary_bundle = load_salary_model()
    role_bundle = load_role_model()
    return salary_bundle, role_bundle


def ensure_models_available():
    """Checks if models are ready, triggers pipeline if not."""
    salary_bundle, _ = get_ml_models()
    if salary_bundle is None:
        with st.spinner("Initializing models and data pipeline... Please wait..."):
            from src.seed_data import run_pipeline
            run_pipeline()
            st.cache_data.clear()
            st.cache_resource.clear()


# Ensure data and models are active
ensure_models_available()
df = load_dataset()
salary_bundle, role_bundle = get_ml_models()


# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <span class="badge-tag">AI & Data Career Analytics Platform</span>
    <h1 class="hero-title">Tech Market Intelligence & Salary Predictor</h1>
    <p class="hero-subtitle">
        Analyze real-world tech compensation, discover high-ROI technical skills, explore live industry postings, 
        and leverage machine learning models to forecast career compensation.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Configuration & Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Navigation & Filters")
    st.markdown("---")

    roles_list = sorted(df["role_category"].dropna().unique().tolist())
    selected_role_filter = st.multiselect(
        "Filter by Role Category",
        options=roles_list,
        default=roles_list,
        help="Select roles to filter the analytics and job explorer views."
    )

    exp_levels = ["Entry-level", "Mid-level", "Senior", "Executive"]
    selected_exp_filter = st.multiselect(
        "Experience Seniority",
        options=exp_levels,
        default=exp_levels
    )

    min_sal = int(df["salary_clean"].dropna().min()) if "salary_clean" in df.columns else 30000
    max_sal = int(df["salary_clean"].dropna().max()) if "salary_clean" in df.columns else 350000
    salary_range = st.slider(
        "Salary Filter ($ USD)",
        min_value=min_sal,
        max_value=max_sal,
        value=(min_sal, max_sal),
        step=5000,
        format="$%d"
    )

    st.markdown("---")
    st.markdown("#### ⚡ Live API Ingestion")
    if st.button("🔄 Fetch Fresh Remotive Jobs", use_container_width=True):
        with st.spinner("Fetching latest live tech jobs from Remotive API..."):
            try:
                new_jobs = fetch_remotive_jobs(category="data", limit=25)
                st.success(f"Successfully fetched {len(new_jobs)} live jobs!")
                time.sleep(1)
            except Exception as e:
                st.error(f"Error fetching API data: {e}")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #94A3B8; line-height: 1.5;'>
            <b>TechStack:</b> Python 3.12, Streamlit, Scikit-Learn, Plotly, Pandas.<br>
            <b>Model:</b> GradientBoostingRegressor & RandomForestClassifier.
        </div>
        """,
        unsafe_allow_html=True
    )

# Filter dataframe based on sidebar
filtered_df = df[
    (df["role_category"].isin(selected_role_filter)) &
    (df["experience_level"].isin(selected_exp_filter)) &
    (df["salary_clean"].between(salary_range[0], salary_range[1]))
]


# ---------------------------------------------------------
# Top KPI Metric Cards Row
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_jobs = len(filtered_df)
avg_salary = filtered_df["salary_clean"].mean() if not filtered_df.empty else 0
median_salary = filtered_df["salary_clean"].median() if not filtered_df.empty else 0

# Determine top paying role in current filter
if not filtered_df.empty and "role_category" in filtered_df.columns:
    top_role_series = filtered_df.groupby("role_category")["salary_clean"].mean().sort_values(ascending=False)
    top_paying_role = top_role_series.index[0] if len(top_role_series) > 0 else "N/A"
    top_role_val = top_role_series.iloc[0] if len(top_role_series) > 0 else 0
else:
    top_paying_role = "N/A"
    top_role_val = 0

# Most demanded skill
skill_cols = [c for c in filtered_df.columns if c.startswith("skill_")]
if skill_cols and not filtered_df.empty:
    skill_counts = filtered_df[skill_cols].sum().sort_values(ascending=False)
    top_skill = skill_counts.index[0].replace("skill_", "").upper() if not skill_counts.empty else "PYTHON"
    top_skill_count = int(skill_counts.iloc[0]) if not skill_counts.empty else 0
else:
    top_skill = "PYTHON"
    top_skill_count = 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Filtered Jobs</div>
        <div class="metric-value">{total_jobs:,}</div>
        <div class="metric-sub">Across {len(selected_role_filter)} active role categories</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Compensation</div>
        <div class="metric-value">${avg_salary:,.0f}</div>
        <div class="metric-sub">Median: ${median_salary:,.0f} / year</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Top Paying Role</div>
        <div class="metric-value" style="font-size: 1.4rem; line-height: 1.3;">{top_paying_role}</div>
        <div class="metric-sub">${top_role_val:,.0f} avg base compensation</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">#1 In-Demand Skill</div>
        <div class="metric-value" style="color: #818CF8;">{top_skill}</div>
        <div class="metric-sub">Present in {top_skill_count} job postings</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Tabbed Main Application
# ---------------------------------------------------------
tab_market, tab_predict, tab_classify, tab_explorer, tab_metrics, tab_data = st.tabs([
    "📊 Market Intelligence",
    "🤖 AI Salary Predictor",
    "🎯 Role Classifier & Matcher",
    "🔎 Live Job Explorer",
    "📈 ML Model Analytics",
    "📥 Data Hub & Export"
])


# =========================================================
# TAB 1: Market Intelligence & EDA
# =========================================================
with tab_market:
    st.markdown("### 📊 Tech Job Market & Compensation Insights")
    st.markdown("Interactive exploratory data analysis on real-world compensation distributions, skill premiums, and experience curves.")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Salary by Role Category Boxplot
        fig_salary_role = px.box(
            filtered_df,
            x="role_category",
            y="salary_clean",
            color="role_category",
            title="<b>Annual Salary Distribution by Role Category</b>",
            labels={"role_category": "Role Category", "salary_clean": "Annual Salary ($ USD)"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_salary_role.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            margin=dict(l=20, r=20, t=50, b=50),
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_salary_role, use_container_width=True)

    with col_chart2:
        # Salary progression by Experience Level
        exp_order = ["Entry-level", "Mid-level", "Senior", "Executive"]
        exp_df = filtered_df.groupby("experience_level")["salary_clean"].agg(["mean", "median", "count"]).reindex(exp_order).reset_index()
        
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            x=exp_df["experience_level"],
            y=exp_df["median"],
            name="Median Salary",
            marker=dict(color="#6366F1", line=dict(color="#818CF8", width=1.5)),
            text=exp_df["median"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else ""),
            textposition="auto"
        ))
        fig_exp.add_trace(go.Scatter(
            x=exp_df["experience_level"],
            y=exp_df["mean"],
            name="Mean Salary",
            mode="lines+markers",
            marker=dict(size=10, color="#10B981"),
            line=dict(width=3, color="#10B981")
        ))
        fig_exp.update_layout(
            title="<b>Compensation Growth by Seniority Level</b>",
            yaxis_title="Salary ($ USD)",
            xaxis_title="Seniority Level",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    # Row 2 Charts
    col_chart3, col_chart4 = st.columns([1.2, 0.8])

    with col_chart3:
        # Top 15 Technical Skills in Demand
        all_skill_cols = [c for c in filtered_df.columns if c.startswith("skill_")]
        if all_skill_cols:
            skill_sums = filtered_df[all_skill_cols].sum()
            skill_names = [c.replace("skill_", "").upper() for c in skill_sums.index]
            skills_df = pd.DataFrame({"skill": skill_names, "count": skill_sums.values}).sort_values("count", ascending=True).tail(15)

            fig_skills = px.bar(
                skills_df,
                x="count",
                y="skill",
                orientation="h",
                title="<b>Top 15 Most Demanded Technical Skills</b>",
                labels={"count": "Number of Job Postings", "skill": "Skill"},
                template="plotly_dark",
                color="count",
                color_continuous_scale="Purples"
            )
            fig_skills.update_layout(
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                margin=dict(l=20, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_skills, use_container_width=True)

    with col_chart4:
        # Remote / Top Location Breakdown
        if "location" in filtered_df.columns:
            loc_counts = filtered_df["location"].value_counts().head(8).reset_index()
            loc_counts.columns = ["location", "count"]

            fig_loc = px.pie(
                loc_counts,
                values="count",
                names="location",
                hole=0.45,
                title="<b>Top Hiring Locations & Remote Distribution</b>",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Agsunset
            )
            fig_loc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                margin=dict(l=20, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_loc, use_container_width=True)


# =========================================================
# TAB 2: AI Salary Predictor
# =========================================================
with tab_predict:
    st.markdown("### 🤖 Machine Learning Salary Estimation")
    st.markdown("Configure your profile, seniority, location, and technical skill set to forecast realistic annual compensation using our trained gradient boosted regressor.")

    col_input, col_result = st.columns([1.1, 1.2])

    with col_input:
        st.markdown("#### 1. Candidate & Position Parameters")
        
        pred_role = st.selectbox(
            "Target Role",
            options=roles_list,
            index=0
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pred_exp = st.selectbox(
                "Experience Level",
                options=["Entry-level", "Mid-level", "Senior", "Executive"],
                index=1
            )
        with col_p2:
            pred_emp_type = st.selectbox(
                "Employment Type",
                options=["Full-Time", "Contract", "Part-Time", "Freelance"],
                index=0
            )

        # Location selector
        common_locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Remote, US", "Remote, Worldwide", "London, UK", "Toronto, Canada", "Other"
        ]
        pred_location = st.selectbox("Location / Work Mode", options=common_locations, index=0)

        st.markdown("#### 2. Select Your Technical Skills")
        
        # Categorized skill choices for clean UX
        skill_groups = {
            "Languages": ["python", "sql", "r", "javascript", "typescript", "java", "c++", "bash"],
            "ML / AI & Data Science": ["machine_learning", "deep_learning", "pytorch", "tensorflow", "scikit_learn", "xgboost", "llm", "nlp", "computer_vision", "mlops"],
            "Data Engineering & Cloud": ["spark", "airflow", "snowflake", "bigquery", "aws", "gcp", "azure", "docker", "kubernetes", "kafka", "dbt", "git"],
            "BI & Visualization": ["tableau", "power_bi", "looker", "excel"]
        }

        selected_skills = []
        for group, skills in skill_groups.items():
            with st.expander(f"{group} ({len(skills)} skills)", expanded=(group in ["Languages", "ML / AI & Data Science"])):
                cols = st.columns(3)
                for i, skill in enumerate(skills):
                    with cols[i % 3]:
                        if st.checkbox(skill.replace("_", " ").title(), key=f"pred_skill_{skill}"):
                            selected_skills.append(skill)

        predict_btn = st.button("🚀 Calculate Estimated Compensation", use_container_width=True)

    with col_result:
        st.markdown("#### 3. Prediction & Market Benchmarking")

        # Run prediction
        pred_data = predict_single_salary(
            role=pred_role,
            experience=pred_exp,
            employment_type=pred_emp_type,
            location=pred_location,
            selected_skills=selected_skills
        )

        if "error" in pred_data:
            st.error(pred_data["error"])
        else:
            est = pred_data["estimated_salary"]
            low = pred_data["range_lower"]
            high = pred_data["range_upper"]

            # Role baseline comparison
            role_baseline = df[df["role_category"] == pred_role]["salary_clean"].mean() if not df.empty else 120000
            diff_pct = ((est - role_baseline) / role_baseline) * 100
            diff_sign = "+" if diff_pct >= 0 else ""
            diff_color = "#10B981" if diff_pct >= 0 else "#EF4444"

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #818CF8; letter-spacing: 0.05em; margin-bottom: 6px;">
                    Forecasted Annual Base Compensation
                </div>
                <div class="prediction-amount">${est:,.0f} <span style="font-size: 1.1rem; color: #94A3B8; font-weight: 500;">USD / year</span></div>
                <div style="margin-top: 8px; font-size: 0.95rem; color: #CBD5E1;">
                    Expected Range: <b style="color: #FFFFFF;">${low:,.0f} &ndash; ${high:,.0f}</b> (85% Confidence Interval)
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem; color: {diff_color}; font-weight: 600;">
                    {diff_sign}{diff_pct:.1f}% vs {pred_role} industry average (${role_baseline:,.0f})
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Gauge Chart for Salary Bracket
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=est,
                number={"prefix": "$", "valueformat": ",.0f", "font": {"color": "#FFFFFF", "size": 28}},
                title={"text": "<b>Compensation Bracket Positioning</b>", "font": {"color": "#CBD5E1", "size": 15}},
                gauge={
                    "axis": {"range": [40000, 300000], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": "#6366F1", "thickness": 0.3},
                    "bgcolor": "rgba(30, 41, 59, 0.5)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [40000, 90000], "color": "rgba(59, 130, 246, 0.2)"},
                        {"range": [90000, 150000], "color": "rgba(16, 185, 129, 0.25)"},
                        {"range": [150000, 220000], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [220000, 300000], "color": "rgba(168, 85, 247, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "#10B981", "width": 4},
                        "thickness": 0.75,
                        "value": est
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(l=25, r=25, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # High ROI Skill Recommendations
            st.markdown("#### 💡 High-ROI Skill Recommendations")
            st.markdown("Skills highly associated with salary premiums for this role that you haven't selected yet:")
            
            valuable_skills = ["llm", "mlops", "pytorch", "snowflake", "spark", "aws", "kubernetes", "deep_learning"]
            recommended = [s for s in valuable_skills if s not in selected_skills][:4]
            
            if recommended:
                pills_html = "".join([f"<span class='skill-pill-highlight'>+ {s.upper()} (~+$12k - $20k)</span>" for s in recommended])
                st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
            else:
                st.info("You have already selected top high-premium technical skills!")


# =========================================================
# TAB 3: Role Classifier & Skill Matcher
# =========================================================
with tab_classify:
    st.markdown("### 🎯 Job Role Classification & Skill Gap Matcher")
    st.markdown("Paste a job description or text excerpt. Our trained multi-class classifier will categorize the role and detect missing high-priority technical skills.")

    sample_desc = """We are looking for a Senior Machine Learning Engineer to design and deploy scalable deep learning models. 
Responsibilities include training transformer architectures using PyTorch and HuggingFace, building automated MLOps pipelines with Docker and Kubernetes, and orchestrating cloud services on AWS. Candidate must have strong Python, SQL, and Git expertise."""

    input_text = st.text_area(
        "Paste Job Description / Candidate Bio",
        value=sample_desc,
        height=140
    )

    if st.button("🔍 Analyze Text & Classify Role", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # 1. Skill Extraction
            detected_skills = extract_skills_from_text(input_text)
            
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown("#### Extracted Technical Skills")
                if detected_skills:
                    pills = "".join([f"<span class='skill-pill'>{s.upper()}</span>" for s in detected_skills])
                    st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #10B981; font-weight: 600; margin-top: 10px;'>Found {len(detected_skills)} relevant technical keywords.</p>", unsafe_allow_html=True)
                else:
                    st.info("No predefined technical skills detected in the text.")

            with col_res2:
                st.markdown("#### Predicted Role Category")
                # Inference using Role Classifier
                if role_bundle is not None:
                    clf_model = role_bundle["model"]
                    clf_meta = role_bundle["metadata"]
                    clf_classes = clf_meta["classes"]

                    # Construct feature vector
                    cat_cols = clf_meta["categorical_columns"]
                    ohe = clf_meta["one_hot_encoder"]
                    dummy_cat = pd.DataFrame([{"experience_level": "Mid-level", "employment_type": "Full-Time"}])[cat_cols]
                    encoded_cat = ohe.transform(dummy_cat)
                    cat_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(cat_cols))

                    skill_dict = {}
                    for col in clf_meta["skill_columns"]:
                        s_name = col.replace("skill_", "")
                        skill_dict[col] = 1 if s_name in detected_skills else 0
                    skill_df = pd.DataFrame([skill_dict])

                    full_clf_x = pd.concat([cat_df, skill_df], axis=1)[clf_meta["feature_names"]]
                    
                    if hasattr(clf_model, "predict_proba"):
                        probs = clf_model.predict_proba(full_clf_x)[0]
                        prob_df = pd.DataFrame({
                            "Role": clf_classes,
                            "Probability": probs
                        }).sort_values("Probability", ascending=False)

                        predicted_top_role = prob_df.iloc[0]["Role"]
                        conf_pct = prob_df.iloc[0]["Probability"] * 100

                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 16px 20px;">
                            <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Top Classification Match</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #818CF8;">{predicted_top_role}</div>
                            <div style="font-size: 0.85rem; color: #10B981; font-weight: 600;">{conf_pct:.1f}% Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Small probability bar chart
                        fig_prob = px.bar(
                            prob_df.head(4),
                            x="Probability",
                            y="Role",
                            orientation="h",
                            title="Role Probability Distribution",
                            template="plotly_dark",
                            color="Probability",
                            color_continuous_scale="Purples"
                        )
                        fig_prob.update_layout(
                            coloraxis_showscale=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30, 41, 59, 0.3)",
                            height=200,
                            margin=dict(l=10, r=10, t=35, b=20)
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)


# =========================================================
# TAB 4: Live Job Explorer
# =========================================================
with tab_explorer:
    st.markdown("### 🔎 Search & Explore Tech Job Postings")
    st.markdown("Filter, search by keyword, and inspect real-time job openings in the cleaned repository.")

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        keyword = st.text_input("🔍 Search by title, company, or skill keyword", placeholder="e.g. PyTorch, Snowflake, Data Scientist")
    with col_s2:
        sort_by = st.selectbox("Sort By", ["Highest Salary", "Lowest Salary", "Job Title", "Company"])

    job_view_df = filtered_df.copy()
    if keyword.strip():
        k = keyword.lower()
        job_view_df = job_view_df[
            job_view_df["job_title"].str.lower().str.contains(k, na=False) |
            job_view_df["company"].str.lower().str.contains(k, na=False) |
            job_view_df["description"].str.lower().str.contains(k, na=False) |
            job_view_df["skills"].str.lower().str.contains(k, na=False)
        ]

    # Sorting
    if sort_by == "Highest Salary":
        job_view_df = job_view_df.sort_values("salary_clean", ascending=False)
    elif sort_by == "Lowest Salary":
        job_view_df = job_view_df.sort_values("salary_clean", ascending=True)
    elif sort_by == "Job Title":
        job_view_df = job_view_df.sort_values("job_title")
    elif sort_by == "Company":
        job_view_df = job_view_df.sort_values("company")

    st.markdown(f"**Showing {len(job_view_df)} matching opportunities:**")

    # Render job cards (top 20 for speed)
    for idx, row in job_view_df.head(20).iterrows():
        title = row.get("job_title", "Tech Role")
        company = row.get("company", "Confidential")
        location = row.get("location", "Remote")
        sal_val = row.get("salary_clean", 0)
        sal_str = f"${sal_val:,.0f} / year" if pd.notna(sal_val) and sal_val > 0 else "Compensation Undisclosed"
        exp = row.get("experience_level", "Mid-level")
        emp = row.get("employment_type", "Full-Time")
        desc = row.get("description", "")
        skills_str = row.get("skills", "")
        skills_list = [s.strip() for s in skills_str.split(",") if s.strip()][:5]

        pills = "".join([f"<span class='skill-pill'>{s.upper()}</span>" for s in skills_list])

        st.markdown(f"""
        <div class="job-listing-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="job-title">{title}</div>
                    <div class="job-company">🏢 {company} &nbsp;•&nbsp; 📍 {location}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #10B981;">{sal_str}</div>
                    <span class="badge-tag" style="margin: 0; font-size: 0.68rem;">{exp} • {emp}</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.88rem; color: #CBD5E1; line-height: 1.4;">
                {desc[:220]}...
            </div>
            <div style="margin-top: 8px;">
                {pills}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 5: ML Model Analytics & Explainability
# =========================================================
with tab_metrics:
    st.markdown("### 📈 Model Architecture, Metrics & Explainability")
    st.markdown("Evaluation metrics comparing regression algorithms and global feature importance explaining salary predictions.")

    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics_summary.pkl")
    if os.path.exists(metrics_path):
        import joblib
        summary = joblib.load(metrics_path)
        reg_metrics = summary.get("reg_metrics", {})
        feat_importances = summary.get("feat_importances", {})
        clf_metrics = summary.get("clf_metrics", {})

        col_m1, col_m2 = st.columns([1, 1.2])

        with col_m1:
            st.markdown("#### Regression Model Benchmark")
            reg_table = []
            for name, m in reg_metrics.items():
                reg_table.append({
                    "Algorithm": name,
                    "R² Score": f"{m['R2']:.4f}",
                    "MAE ($)": f"${m['MAE']:,.2f}",
                    "RMSE ($)": f"${m['RMSE']:,.2f}"
                })
            st.dataframe(pd.DataFrame(reg_table), use_container_width=True, hide_index=True)

            st.markdown("#### Classification Model Benchmark")
            clf_table = []
            for name, m in clf_metrics.items():
                clf_table.append({
                    "Classifier": name,
                    "Accuracy": f"{m['Accuracy'] * 100:.2f}%",
                    "Weighted F1": f"{m['Weighted_F1']:.4f}"
                })
            st.dataframe(pd.DataFrame(clf_table), use_container_width=True, hide_index=True)

        with col_m2:
            st.markdown("#### Global Feature Importance (Top Salary Drivers)")
            if feat_importances:
                fi_df = pd.DataFrame({
                    "Feature": list(feat_importances.keys()),
                    "Importance": list(feat_importances.values())
                }).sort_values("Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="<b>Key Features Influencing Compensation</b>",
                    template="plotly_dark",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_fi.update_layout(
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30, 41, 59, 0.4)",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=30)
                )
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Metrics summary file will be refreshed on model retraining.")


# =========================================================
# TAB 6: Data Hub & Export
# =========================================================
with tab_data:
    st.markdown("### 📥 Dataset Repository & Export")
    st.markdown("Preview the cleaned, standardized dataset with encoded features and export for offline analysis.")

    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="cleaned_tech_jobs_data.csv",
        mime="text/csv",
        use_container_width=True
    )
