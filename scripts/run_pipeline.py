#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data                    # Data loading with error handling
from src.data.preprocess import preprocess_data, save_processed_data  # Canonical data cleaning/output
from src.features.build_features import fit_feature_encoder  # Fit preprocessing mappings on training data
from src.utils.validate_data import validate_telco_data    # Data quality validation

RANDOM_STATE = 42

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or Path(project_root, "mlruns").resolve().as_uri()  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")           # Model type for comparison
        mlflow.log_param("threshold", args.threshold)   # Classification threshold (default: 0.35)
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        input_path = Path(args.input).resolve()
        dataset_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
        df = load_data(input_path)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        mlflow.log_params(
            {
                "target": args.target,
                "data_filename": input_path.name,
                "data_sha256": dataset_sha256,
                "data_rows": int(df.shape[0]),
                "data_columns": int(df.shape[1]),
                "split_random_state": RANDOM_STATE,
                "python_version": platform.python_version(),
                "mlflow_version": mlflow.__version__,
                "pandas_version": pd.__version__,
                "scikit_learn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
            }
        )

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df, target_col=args.target)

        # Save processed dataset for reproducibility and debugging
        processed_path = save_processed_data(
            df, Path(project_root, "data", "processed", "telco_churn_processed.csv")
        )
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Apply feature engineering transformations
        df_enc, preprocessing_schema = fit_feature_encoder(df, target_col=target)
        
        # IMPORTANT: Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} features")

        # === Log Feature Metadata for Serving Consistency ===
        # This records the exact feature order alongside the fitted encoder.
        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        mlflow.log_param("feature_count", len(feature_cols))

        # Keep the fitted preprocessing schema inside this exact MLflow run.
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessing_path = Path(temp_dir, "preprocessing.pkl")
            joblib.dump(preprocessing_schema, preprocessing_path)
            mlflow.log_artifact(str(preprocessing_path))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: Train/Test Split ===
        print("📊 Splitting data...")
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        
        # Stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    # Default: 20% for testing
            stratify=y,                  # Maintain class balance
            random_state=RANDOM_STATE     # Reproducible splits
        )
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === CRITICAL: Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 5: Model Training with Optimized Hyperparameters ===
        print("🤖 Training XGBoost model...")
        
        # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
        # In production, consider using hyperparameter optimization tools like Optuna
        model_params = {
            # Tree structure parameters
            "n_estimators": 301,        # Number of trees (OPTIMIZED)
            "learning_rate": 0.034,     # Step size shrinkage (OPTIMIZED)
            "max_depth": 7,             # Maximum tree depth (OPTIMIZED)
            
            # Regularization parameters
            "subsample": 0.95,          # Sample ratio of training instances
            "colsample_bytree": 0.98,   # Sample ratio of features for each tree
            
            # Performance parameters
            "n_jobs": -1,               # Use all CPU cores
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            
            # ESSENTIAL: Handle class imbalance
            "scale_pos_weight": float(scale_pos_weight),
        }
        mlflow.log_params({f"model_{key}": value for key, value in model_params.items()})
        model = XGBClassifier(**model_params)

        # === Train Model and Track Training Time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("📊 Evaluating model performance...")
        
        # Generate predictions and track inference time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)
        
        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

        # === CRITICAL: Log Evaluation Metrics to MLflow ===
        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)    # Of predicted churners, how many actually churned?
        recall = recall_score(y_test, y_pred)          # Of actual churners, how many did we catch?
        f1 = f1_score(y_test, y_pred)                  # Harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)         # Area under ROC curve (threshold-independent)
        
        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"🎯 Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)
