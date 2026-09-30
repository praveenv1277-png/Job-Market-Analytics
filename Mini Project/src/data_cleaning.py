"""
Data Cleaning and Preprocessing Module.
Standardizes schema, cleans salaries, standardizes job titles and experience levels,
handles missing values, and removes duplicates.
"""

import re
import logging
from typing import Optional, Union, List, Dict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard experience level mapping
EXPERIENCE_MAPPING = {
    "en": "Entry-level",
    "entry": "Entry-level",
    "junior": "Entry-level",
    "jr": "Entry-level",
    "intern": "Entry-level",
    "trainee": "Entry-level",
    "mi": "Mid-level",
    "mid": "Mid-level",
    "intermediate": "Mid-level",
    "associate": "Mid-level",
    "se": "Senior",
    "senior": "Senior",
    "sr": "Senior",
    "lead": "Senior",
    "principal": "Senior",
    "staff": "Senior",
    "ex": "Executive",
    "executive": "Executive",
    "director": "Executive",
    "head": "Executive",
    "vp": "Executive",
    "chief": "Executive"
}

# Standard role classification rules
ROLE_KEYWORDS = {
    "Data Scientist": [r"data\s*scientist", r"data\s*science"],
    "Machine Learning Engineer": [r"machine\s*learning", r"ml\s*engineer", r"ai\s*engineer", r"deep\s*learning"],
    "Data Engineer": [r"data\s*engineer", r"big\s*data", r"database\s*engineer", r"etl", r"data\s*pipeline"],
    "Data Analyst": [r"data\s*analyst", r"analytics\s*analyst", r"data\s*analytics"],
    "Business Intelligence": [r"bi\s*developer", r"bi\s*analyst", r"business\s*intelligence", r"tableau", r"power\s*bi"],
    "Software Engineer": [r"software\s*engineer", r"software\s*developer", r"full\s*stack", r"backend", r"frontend"],
    "Research / AI": [r"research\s*scientist", r"research\s*engineer", r"nlp", r"computer\s*vision"]
}


def parse_salary_string(val: Union[str, float, int]) -> Optional[float]:
    """
    Parses complex salary strings like '$100k - $140k', '$90,000 / year', '$50 - $70 /hour'
    into an annualized USD estimation.
    """
    if pd.isna(val) or val == "" or str(val).strip().lower() in ["nan", "none", "not disclosed"]:
        return None

    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None

    text = str(val).lower().replace(",", "").strip()

    # Hourly rate detection
    is_hourly = "/hr" in text or "/hour" in text or "per hour" in text or "hr" in text.split()
    is_monthly = "/mo" in text or "/month" in text or "per month" in text

    # Extract all numeric values or 'k' representations
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(k)?", text)
    if not matches:
        return None

    numbers = []
    for num_str, k_suffix in matches:
        try:
            n = float(num_str)
            if k_suffix == "k":
                n *= 1000
            elif n < 500 and not is_hourly:
                # Handle cases like "120 - 150" meaning $120k-$150k
                n *= 1000
            numbers.append(n)
        except ValueError:
            continue

    if not numbers:
        return None

    # Calculate average of range
    salary_est = sum(numbers) / len(numbers)

    # Annualize if hourly (2080 working hours/year) or monthly
    if is_hourly and salary_est < 500:
        salary_est = salary_est * 2080
    elif is_monthly and salary_est < 25000:
        salary_est = salary_est * 12

    # Plausibility check for annual full-time compensation
    if 10000 <= salary_est <= 1500000:
        return round(salary_est, 2)
    return None


def standardize_role(title: str) -> str:
    """Standardizes job titles into primary core categories."""
    if not isinstance(title, str) or not title.strip():
        return "Other"
    
    title_lower = title.lower()
    for role_name, patterns in ROLE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, title_lower):
                return role_name
    return "Other / Specialist"


def standardize_experience(exp: Union[str, float]) -> str:
    """Normalizes experience level into Entry-level, Mid-level, Senior, or Executive."""
    if pd.isna(exp) or not str(exp).strip():
        return "Mid-level"  # default median
    
    exp_clean = str(exp).strip().lower()
    for key, val in EXPERIENCE_MAPPING.items():
        if key == exp_clean or key in exp_clean.split():
            return val
    return "Mid-level"


def clean_job_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw DataFrame, standardizes columns, parses salaries,
    cleans titles, handles missing values, and removes duplicates.
    """
    logger.info(f"Initial raw record count: {len(df)}")
    df = df.copy()

    # Normalize column names to lowercase snake_case
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Map possible column variations
    col_mapping = {
        "job_title": ["title", "position", "role", "jobtitle", "job_name"],
        "company": ["company_name", "employer", "organization"],
        "location": ["candidate_required_location", "job_location", "company_location", "city", "country", "employee_residence"],
        "salary": ["salary_in_usd", "salary_usd", "annual_salary", "pay", "compensation"],
        "experience": ["experience_level", "seniority", "exp_level", "level"],
        "employment_type": ["job_type", "type", "contract_type", "work_type"],
        "description": ["job_description", "job_details", "summary", "responsibilities"],
        "skills": ["tags", "job_tags", "required_skills", "skillset"]
    }

    rename_dict = {}
    for standard_col, alternates in col_mapping.items():
        if standard_col not in df.columns:
            for alt in alternates:
                if alt in df.columns:
                    rename_dict[alt] = standard_col
                    break
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Ensure critical columns exist
    if "job_title" not in df.columns:
        raise ValueError("Dataset does not contain a discernible job title column.")

    for col in ["company", "location", "experience", "employment_type", "description", "skills"]:
        if col not in df.columns:
            df[col] = "Not Specified"

    # Standardize job title and derive role category
    df["job_title"] = df["job_title"].astype(str).str.strip()
    df["role_category"] = df["job_title"].apply(standardize_role)

    # Standardize experience level
    df["experience_level"] = df["experience"].apply(standardize_experience)

    # Process and clean salary
    if "salary" in df.columns:
        df["salary_clean"] = df["salary"].apply(parse_salary_string)
    else:
        df["salary_clean"] = np.nan

    # Clean location
    df["location"] = df["location"].astype(str).str.strip().replace({"": "Remote / Not Specified", "nan": "Remote / Not Specified"})

    # Clean company
    df["company"] = df["company"].astype(str).str.strip().replace({"": "Confidential", "nan": "Confidential"})

    # Standardize employment type
    df["employment_type"] = df["employment_type"].astype(str).str.upper().replace({
        "FT": "Full-Time", "FULL_TIME": "Full-Time", "FULL TIME": "Full-Time",
        "PT": "Part-Time", "PART_TIME": "Part-Time",
        "CT": "Contract", "CONTRACTOR": "Contract",
        "FL": "Freelance", "FREELANCER": "Freelance"
    })

    # Drop exact duplicate rows based on title, company, location
    subset_cols = [c for c in ["job_title", "company", "location"] if c in df.columns]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=subset_cols, keep="first")
    logger.info(f"Removed {before_dedup - len(df)} duplicate records. Current count: {len(df)}")

    # Summary of salary coverage
    valid_salaries = df["salary_clean"].notna().sum()
    logger.info(f"Records with valid salary: {valid_salaries} / {len(df)} ({valid_salaries / len(df) * 100:.1f}%)")

    return df


def load_and_clean_data(file_path: str, output_path: Optional[str] = "data/processed/cleaned_jobs.csv") -> pd.DataFrame:
    """
    Loads raw CSV/JSON, applies clean_job_dataset pipeline, and optionally saves to processed directory.
    """
    logger.info(f"Loading raw data from: {file_path}")
    if file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        df = pd.read_csv(file_path)

    cleaned_df = clean_job_dataset(df)

    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset successfully saved to: {output_path}")

    return cleaned_df
