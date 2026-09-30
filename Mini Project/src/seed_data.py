"""
Data Pipeline and Model Seeding Script.
Fetches real jobs from Remotive API and supplements with statistical tech market data,
cleans the dataset, extracts skill features, trains ML models, and saves artifacts.
"""

import os
import sys
import random
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.ingest_api import fetch_remotive_jobs
from src.data_cleaning import clean_job_dataset
from src.feature_engineering import add_skill_features, prepare_regression_dataset, prepare_classification_dataset
from src.train_model import train_salary_regressors, train_role_classifiers, save_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/job_postings_raw.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_jobs.csv"


def generate_benchmark_jobs(n_samples: int = 1500) -> pd.DataFrame:
    """
    Generates realistic tech and data industry benchmark job postings
    to ensure full statistical coverage for training and analytics.
    """
    roles_config = {
        "Data Scientist": {
            "titles": ["Data Scientist", "Senior Data Scientist", "Lead Data Scientist", "Staff Data Scientist", "Principal Data Scientist", "Junior Data Scientist"],
            "base_salary": (95000, 195000),
            "typical_skills": ["python", "sql", "machine_learning", "scikit_learn", "deep_learning", "pytorch", "r", "tableau", "aws", "docker"],
            "exp_weights": [0.15, 0.45, 0.35, 0.05]
        },
        "Machine Learning Engineer": {
            "titles": ["Machine Learning Engineer", "Senior ML Engineer", "AI Engineer", "MLOps Engineer", "Deep Learning Specialist", "Lead AI Engineer"],
            "base_salary": (115000, 225000),
            "typical_skills": ["python", "pytorch", "tensorflow", "docker", "kubernetes", "mlops", "aws", "gcp", "ci_cd", "llm", "git", "scikit_learn"],
            "exp_weights": [0.10, 0.40, 0.45, 0.05]
        },
        "Data Engineer": {
            "titles": ["Data Engineer", "Senior Data Engineer", "Big Data Engineer", "Lead Data Engineer", "Analytics Engineer", "Cloud Data Architect"],
            "base_salary": (100000, 200000),
            "typical_skills": ["python", "sql", "spark", "airflow", "snowflake", "bigquery", "aws", "kafka", "databricks", "dbt", "docker"],
            "exp_weights": [0.12, 0.48, 0.35, 0.05]
        },
        "Data Analyst": {
            "titles": ["Data Analyst", "Senior Data Analyst", "Business Data Analyst", "Quantitative Analyst", "Marketing Data Analyst"],
            "base_salary": (65000, 130000),
            "typical_skills": ["sql", "tableau", "power_bi", "excel", "python", "r", "looker"],
            "exp_weights": [0.30, 0.50, 0.18, 0.02]
        },
        "Business Intelligence": {
            "titles": ["BI Developer", "Senior BI Analyst", "Business Intelligence Engineer", "Tableau Developer", "Power BI Specialist"],
            "base_salary": (75000, 145000),
            "typical_skills": ["power_bi", "tableau", "sql", "excel", "data_warehouse", "snowflake", "dbt", "python"],
            "exp_weights": [0.20, 0.55, 0.22, 0.03]
        },
        "Software Engineer": {
            "titles": ["Software Engineer", "Senior Backend Engineer", "Full Stack Developer", "Cloud Engineer", "Python Backend Engineer"],
            "base_salary": (90000, 185000),
            "typical_skills": ["python", "javascript", "typescript", "docker", "kubernetes", "aws", "git", "ci_cd", "postgresql", "bash"],
            "exp_weights": [0.20, 0.45, 0.30, 0.05]
        },
        "Research / AI": {
            "titles": ["Research Scientist", "AI Research Engineer", "NLP Scientist", "Computer Vision Specialist", "GenAI Researcher"],
            "base_salary": (130000, 260000),
            "typical_skills": ["python", "pytorch", "deep_learning", "nlp", "computer_vision", "llm", "huggingface", "tensorflow"],
            "exp_weights": [0.10, 0.35, 0.45, 0.10]
        }
    }

    companies = [
        "Google", "Microsoft", "Amazon AWS", "Meta", "Apple", "Netflix", "Databricks",
        "Snowflake", "OpenAI", "Anthropic", "Stripe", "Spotify", "Uber", "Airbnb",
        "Coinbase", "Palantir", "NVIDIA", "Scale AI", "ScaleOps", "Cloudflare",
        "Healthcare AI Labs", "FinTech Nexus", "Quantum Analytics", "Vanguard",
        "JPMorgan Chase", "Capital One", "Target Tech", "Walmart Global Tech"
    ]

    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA",
        "Remote, US", "Chicago, IL", "Denver, CO", "Los Angeles, CA", "London, UK",
        "Toronto, Canada", "Berlin, Germany", "Remote, Worldwide"
    ]

    exp_levels = ["Entry-level", "Mid-level", "Senior", "Executive"]
    exp_mult = {"Entry-level": 0.72, "Mid-level": 1.0, "Senior": 1.35, "Executive": 1.85}

    records = []
    np.random.seed(42)
    random.seed(42)

    for i in range(n_samples):
        role_cat = random.choice(list(roles_config.keys()))
        cfg = roles_config[role_cat]
        title = random.choice(cfg["titles"])
        company = random.choice(companies)
        location = random.choice(locations)
        exp = np.random.choice(exp_levels, p=cfg["exp_weights"])

        # Base range calculation
        low, high = cfg["base_salary"]
        base = np.random.uniform(low, high)
        adjusted_salary = base * exp_mult[exp]

        # Location premium
        if "San Francisco" in location or "New York" in location:
            adjusted_salary *= 1.15
        elif "Remote, Worldwide" in location:
            adjusted_salary *= 0.90

        # Sample skills for this role
        num_skills = random.randint(3, 7)
        chosen_skills = random.sample(cfg["typical_skills"], min(num_skills, len(cfg["typical_skills"])))
        
        # Skill bonus
        if "llm" in chosen_skills or "mlops" in chosen_skills:
            adjusted_salary *= 1.08

        # Create description snippet
        desc = f"We are seeking an exceptional {title} to join {company}. The ideal candidate has deep expertise in {', '.join(chosen_skills)}. Role requires hands-on experience in modern technology workflows, collaboration, and high-impact delivery."

        records.append({
            "job_title": title,
            "company": company,
            "location": location,
            "salary": f"${int(adjusted_salary):,}",
            "experience": exp,
            "employment_type": "Full-Time" if random.random() > 0.1 else "Contract",
            "description": desc,
            "skills": ", ".join(chosen_skills)
        })

    return pd.DataFrame(records)


def run_pipeline():
    """Runs end-to-end ingestion, preprocessing, skill extraction, and model training."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    logger.info("Step 1: Ingesting jobs from Remotive API...")
    remotive_jobs = []
    try:
        remotive_data = fetch_remotive_jobs(category="data", limit=60)
        remotive_jobs.extend(remotive_data)
        remotive_dev = fetch_remotive_jobs(category="software-dev", limit=40)
        remotive_jobs.extend(remotive_dev)
    except Exception as e:
        logger.warning(f"Live Remotive API call had issue: {e}")

    api_records = []
    if remotive_jobs:
        for rj in remotive_jobs:
            api_records.append({
                "job_title": rj.get("title", ""),
                "company": rj.get("company_name", ""),
                "location": rj.get("candidate_required_location", "Remote"),
                "salary": rj.get("salary", ""),
                "experience": "Mid-level",
                "employment_type": rj.get("job_type", "Full-Time"),
                "description": rj.get("description", ""),
                "skills": ", ".join(rj.get("tags", []))
            })
        logger.info(f"Collected {len(api_records)} real-time jobs from Remotive API.")

    logger.info("Step 2: Generating benchmark industry market data...")
    benchmark_df = generate_benchmark_jobs(n_samples=1600)

    if api_records:
        api_df = pd.DataFrame(api_records)
        raw_combined = pd.concat([benchmark_df, api_df], ignore_index=True)
    else:
        raw_combined = benchmark_df

    raw_combined.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Raw dataset saved to {RAW_DATA_PATH} ({len(raw_combined)} records).")

    logger.info("Step 3: Cleaning dataset...")
    cleaned_df = clean_job_dataset(raw_combined)

    logger.info("Step 4: Extracting skill features...")
    df_with_skills, skill_cols = add_skill_features(cleaned_df)
    df_with_skills.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned data saved to {CLEANED_DATA_PATH} ({len(df_with_skills)} records).")

    logger.info("Step 5: Training Salary Regression Models...")
    X_reg, y_reg, reg_features, reg_meta = prepare_regression_dataset(df_with_skills)
    best_reg_model, reg_metrics, feat_importances = train_salary_regressors(X_reg, y_reg)

    logger.info("Step 6: Training Role Classification Models...")
    X_clf, y_clf, clf_features, clf_meta = prepare_classification_dataset(df_with_skills, target_col="role_category")
    best_clf_model, clf_metrics, clf_report = train_role_classifiers(X_clf, y_clf, target_names=clf_meta["classes"])

    logger.info("Step 7: Saving Model Artifacts...")
    save_artifacts(best_reg_model, reg_meta, best_clf_model, clf_meta)

    # Save performance summary for dashboard display
    metrics_summary = {
        "reg_metrics": reg_metrics,
        "feat_importances": feat_importances,
        "clf_metrics": clf_metrics,
        "clf_classes": clf_meta["classes"]
    }
    import joblib
    joblib.dump(metrics_summary, "models/metrics_summary.pkl")

    logger.info("Pipeline executed successfully! Ready for Streamlit App.")


if __name__ == "__main__":
    run_pipeline()
