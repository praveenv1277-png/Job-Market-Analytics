"""
Model Training and Evaluation Module.
Trains, evaluates, and saves:
1. Salary Prediction Regression Models (RandomForest, GradientBoosting, Ridge)
2. Role / Seniority Classification Models (RandomForest, LogisticRegression)
Generates evaluation metrics (RMSE, MAE, R2, Accuracy, F1-score) and serializes artifacts.
"""

import os
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_salary_regressors(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Trains multiple regression models to predict salary, compares them,
    and returns the best performing model based on test R2 score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=random_state),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state),
        "RidgeRegression": Ridge(alpha=1.0)
    }

    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    best_model = None

    for name, model in models.items():
        logger.info(f"Training regression model: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        logger.info(f"{name} -> RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    logger.info(f"Best regression model: {best_model_name} with R2 = {best_r2:.4f}")

    # Extract feature importances if available
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx[:15]:
            feature_importances[X.columns[idx]] = round(float(importances[idx]), 4)

    return best_model, results, feature_importances


def train_role_classifiers(
    X: pd.DataFrame,
    y: np.ndarray,
    target_names: list,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Dict[str, float]], str]:
    """
    Trains classification models to predict job role/category,
    and returns the best performing model based on weighted F1-score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state)
    }

    results = {}
    best_model_name = None
    best_f1 = -float("inf")
    best_model = None
    best_report = ""

    for name, model in models.items():
        logger.info(f"Training classifier: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        report = classification_report(y_test, y_pred, target_names=[str(c) for c in target_names], zero_division=0)

        results[name] = {"Accuracy": acc, "Weighted_F1": f1}
        logger.info(f"{name} -> Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            best_report = report

    return best_model, results, best_report


def save_artifacts(salary_model, salary_meta, role_model, role_meta):
    """Saves models and preprocessing metadata to disk with joblib."""
    salary_payload = {
        "model": salary_model,
        "metadata": salary_meta
    }
    joblib.dump(salary_payload, os.path.join(MODELS_DIR, "salary_model.pkl"))
    logger.info("Saved models/salary_model.pkl")

    if role_model is not None:
        role_payload = {
            "model": role_model,
            "metadata": role_meta
        }
        joblib.dump(role_payload, os.path.join(MODELS_DIR, "role_model.pkl"))
        logger.info("Saved models/role_model.pkl")


def load_salary_model():
    """Loads saved salary prediction model and metadata."""
    path = os.path.join(MODELS_DIR, "salary_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_role_model():
    """Loads saved role classification model and metadata."""
    path = os.path.join(MODELS_DIR, "role_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_single_salary(
    role: str,
    experience: str,
    employment_type: str,
    location: str,
    selected_skills: list
) -> Dict[str, Any]:
    """
    Performs inference to predict salary for a specific combination of inputs.
    """
    bundle = load_salary_model()
    if bundle is None:
        return {"error": "Salary model is not trained yet."}

    model = bundle["model"]
    meta = bundle["metadata"]
    ohe = meta["one_hot_encoder"]
    cat_cols = meta["categorical_columns"]
    feature_names = meta["feature_names"]

    # Construct categorical row
    input_cat = pd.DataFrame([{
        "role_category": role,
        "experience_level": experience,
        "employment_type": employment_type,
        "location_grouped": location if "location_grouped" in cat_cols else "Other"
    }])
    # Align to expected cat cols
    input_cat = input_cat[[c for c in cat_cols if c in input_cat.columns]]
    encoded_cat = ohe.transform(input_cat)
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(cat_cols))

    # Construct skill vector
    skill_data = {}
    for col in meta["skill_columns"]:
        skill_name = col.replace("skill_", "")
        skill_data[col] = 1 if skill_name in selected_skills else 0
    skill_df = pd.DataFrame([skill_data])

    full_x = pd.concat([encoded_cat_df, skill_df], axis=1)
    # Ensure all feature columns match model input
    for col in feature_names:
        if col not in full_x.columns:
            full_x[col] = 0
    full_x = full_x[feature_names]

    predicted_val = float(model.predict(full_x)[0])
    lower_bound = max(15000, round(predicted_val * 0.88, -2))
    upper_bound = round(predicted_val * 1.14, -2)

    return {
        "estimated_salary": round(predicted_val, -2),
        "range_lower": lower_bound,
        "range_upper": upper_bound,
        "currency": "USD"
    }
