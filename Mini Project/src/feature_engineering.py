"""
Skill Extraction and Feature Engineering Module.
Extracts top technical skills from job descriptions and tags using NLP/regex matching,
generates multi-hot skill encodings, and prepares feature matrices for regression and classification models.
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Curated comprehensive tech skills dictionary across Data Science, AI, and Engineering
TECH_SKILLS = {
    # Programming Languages
    "python": r"\bpython\b",
    "r": r"\br\b",
    "sql": r"\bsql\b",
    "java": r"\bjava\b",
    "c++": r"\bc\+\+\b",
    "c#": r"\bc#\b",
    "scala": r"\bscala\b",
    "julia": r"\bjulia\b",
    "go": r"\b(golang|go)\b",
    "rust": r"\brust\b",
    "javascript": r"\b(javascript|js)\b",
    "typescript": r"\btypescript\b",
    "bash": r"\b(bash|shell)\b",

    # Machine Learning & AI
    "machine_learning": r"\bmachine\s+learning\b",
    "deep_learning": r"\bdeep\s+learning\b",
    "nlp": r"\b(nlp|natural\s+language\s+processing)\b",
    "computer_vision": r"\b(computer\s+vision|cv)\b",
    "pytorch": r"\bpytorch\b",
    "tensorflow": r"\btensorflow\b",
    "keras": r"\bkeras\b",
    "scikit_learn": r"\b(scikit-learn|sklearn)\b",
    "xgboost": r"\b(xgboost|xgb)\b",
    "llm": r"\b(llm|llms|large\s+language\s+models|generative\s+ai|genai)\b",
    "huggingface": r"\bhuggingface\b",
    "mlops": r"\bmlops\b",

    # Data Engineering & Big Data
    "spark": r"\b(spark|pyspark)\b",
    "hadoop": r"\bhadoop\b",
    "airflow": r"\bairflow\b",
    "kafka": r"\bkafka\b",
    "snowflake": r"\bsnowflake\b",
    "bigquery": r"\bbigquery\b",
    "redshift": r"\bredshift\b",
    "databricks": r"\bdatabricks\b",
    "etl": r"\betl\b",
    "data_warehouse": r"\bdata\s+warehous(e|ing)\b",
    "dbt": r"\bdbt\b",

    # Cloud & DevOps
    "aws": r"\baws\b",
    "azure": r"\bazure\b",
    "gcp": r"\b(gcp|google\s+cloud)\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\b(kubernetes|k8s)\b",
    "git": r"\bgit\b",
    "ci_cd": r"\b(ci/cd|cicd)\b",

    # BI, Analytics & Databases
    "tableau": r"\btableau\b",
    "power_bi": r"\bpower\s*bi\b",
    "excel": r"\bexcel\b",
    "looker": r"\blooker\b",
    "postgresql": r"\b(postgres|postgresql)\b",
    "mongodb": r"\bmongodb\b",
    "nosql": r"\bnosql\b"
}


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extracts matching tech skills from a text string (description or tags) using regex patterns.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    text_lower = text.lower()
    matched = []
    for skill_name, pattern in TECH_SKILLS.items():
        if re.search(pattern, text_lower):
            matched.append(skill_name)
    return matched


def add_skill_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts skills for each job posting by analyzing both the 'description', 'skills', and 'job_title' columns.
    Adds a list column 'extracted_skills' and binary indicator columns for each detected skill.
    """
    df = df.copy()

    # Combine text representations
    combined_text = (
        df.get("job_title", "").astype(str) + " " +
        df.get("skills", "").astype(str) + " " +
        df.get("description", "").astype(str)
    )

    logger.info("Extracting technical skills from text fields...")
    df["extracted_skills"] = combined_text.apply(extract_skills_from_text)

    # Generate binary indicator columns for skills that appear in at least 1 job
    skill_cols = []
    for skill in TECH_SKILLS.keys():
        col_name = f"skill_{skill}"
        df[col_name] = df["extracted_skills"].apply(lambda s: 1 if skill in s else 0)
        if df[col_name].sum() > 0:
            skill_cols.append(col_name)

    logger.info(f"Extracted {len(skill_cols)} active skill indicator features.")
    return df, skill_cols


def prepare_regression_dataset(
    df: pd.DataFrame,
    min_samples_per_category: int = 5
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict]:
    """
    Prepares features (X) and target (y) for salary regression.
    Filters to records with non-null salary.
    Encodes categorical features and includes skill binary indicators.
    """
    # Filter valid positive salaries
    valid_mask = df["salary_clean"].notna() & (df["salary_clean"] > 0)
    data = df[valid_mask].copy()

    if len(data) == 0:
        raise ValueError("No valid salary records found for regression training.")

    y = data["salary_clean"].values

    # Categorical features to encode
    cat_cols = ["role_category", "experience_level", "employment_type"]
    avail_cats = [c for c in cat_cols if c in data.columns]

    # Location grouping (keep top locations, group rest as 'Other')
    if "location" in data.columns:
        top_locs = data["location"].value_counts()[lambda x: x >= min_samples_per_category].index.tolist()
        data["location_grouped"] = data["location"].apply(lambda loc: loc if loc in top_locs else "Other")
        avail_cats.append("location_grouped")

    # One-Hot Encoding for categorical features
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = ohe.fit_transform(data[avail_cats])
    cat_feature_names = ohe.get_feature_names_out(avail_cats).tolist()
    X_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=data.index)

    # Skill indicators
    skill_cols = [c for c in data.columns if c.startswith("skill_")]
    X_skills = data[skill_cols]

    # Combine features
    X = pd.concat([X_cat, X_skills], axis=1)
    feature_names = list(X.columns)

    metadata = {
        "one_hot_encoder": ohe,
        "categorical_columns": avail_cats,
        "feature_names": feature_names,
        "skill_columns": skill_cols
    }

    logger.info(f"Regression dataset prepared: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y, feature_names, metadata


def prepare_classification_dataset(
    df: pd.DataFrame,
    target_col: str = "role_category",
    min_class_samples: int = 5
) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict]:
    """
    Prepares features (X) and target (y) for job role or seniority level classification.
    """
    data = df.copy()

    # Filter classes with at least min_class_samples
    class_counts = data[target_col].value_counts()
    valid_classes = class_counts[class_counts >= min_class_samples].index
    data = data[data[target_col].isin(valid_classes)].copy()

    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(data[target_col])

    # Feature columns
    cat_cols = [c for c in ["experience_level", "employment_type"] if c != target_col and c in data.columns]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = ohe.fit_transform(data[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    X_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=data.index)

    # Skill indicators
    skill_cols = [c for c in data.columns if c.startswith("skill_")]
    X_skills = data[skill_cols]

    X = pd.concat([X_cat, X_skills], axis=1)
    feature_names = list(X.columns)

    metadata = {
        "label_encoder": le,
        "one_hot_encoder": ohe,
        "categorical_columns": cat_cols,
        "feature_names": feature_names,
        "classes": list(le.classes_),
        "skill_columns": skill_cols
    }

    logger.info(f"Classification dataset prepared for '{target_col}': {X.shape[0]} rows, {len(le.classes_)} classes.")
    return X, y, feature_names, metadata
