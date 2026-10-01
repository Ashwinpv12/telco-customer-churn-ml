"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses category and binary mappings fitted during training
- Applies one-hot encoding against the saved category schema
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import json
import math
import os

import joblib
import mlflow.sklearn
import pandas as pd
from src.features.build_features import transform_features
from src.serving.decision import churn_class_from_probability

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.abspath(
    os.getenv("MODEL_DIR", os.path.join(project_root, "src", "serving", "model", "production"))
)
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Serving model bundle not found at {MODEL_DIR}. "
        "Package a specific MLflow run with scripts/package_model.py first."
    )

feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
preprocessing_file = os.path.join(MODEL_DIR, "preprocessing.pkl")
manifest_file = os.path.join(MODEL_DIR, "bundle.json")
if not os.path.isfile(os.path.join(MODEL_DIR, "MLmodel")):
    raise FileNotFoundError(f"MLflow model metadata is missing from bundle: {MODEL_DIR}")
if not all(os.path.isfile(path) for path in (feature_file, preprocessing_file, manifest_file)):
    raise FileNotFoundError(
        f"Bundle must contain MLmodel, feature_columns.txt, preprocessing.pkl, and bundle.json: {MODEL_DIR}"
    )

with open(feature_file, encoding="utf-8") as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]
PREPROCESSING_SCHEMA = joblib.load(preprocessing_file)
with open(manifest_file, encoding="utf-8") as f:
    BUNDLE_MANIFEST = json.load(f)
try:
    DECISION_THRESHOLD = float(BUNDLE_MANIFEST["decision_threshold"])
except (KeyError, TypeError, ValueError) as error:
    raise ValueError("Bundle manifest must contain a numeric decision_threshold") from error
if not math.isfinite(DECISION_THRESHOLD) or not 0.0 <= DECISION_THRESHOLD <= 1.0:
    raise ValueError("Bundle decision_threshold must be between 0 and 1")
if PREPROCESSING_SCHEMA.get("feature_columns") != FEATURE_COLS:
    raise ValueError("Feature columns do not match the preprocessing schema in the model bundle")
if "categorical_categories" not in PREPROCESSING_SCHEMA:
    raise ValueError(
        "Model bundle has no fitted category mappings. "
        "Package a model trained with the current scripts/run_pipeline.py."
    )

# The exact sklearn model, preprocessing, and decision threshold come from one run.
model = mlflow.sklearn.load_model(MODEL_DIR)
print(f"✅ Model bundle loaded from {MODEL_DIR}")

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
        1. Coerce numeric fields and clean column names
        2. Apply fitted binary and categorical mappings
        3. One-hot encode using training-time categories
        4. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return transform_features(df, PREPROCESSING_SCHEMA)

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Apply the threshold selected and logged with this exact model run.
    try:
        probability = float(model.predict_proba(df_enc)[0][1])
        result = churn_class_from_probability(probability, DECISION_THRESHOLD)
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service
