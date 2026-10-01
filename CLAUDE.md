# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training Pipeline
```bash
# Run the complete ML training pipeline
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Prepare processed data only
python scripts/prepare_processed_data.py
```

### Testing
```bash
# Run the automated regression suite
python -m pytest tests -q
```

### Local Development
```bash
# Run the FastAPI + Gradio application locally
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Legacy compatibility alias; main.py is the canonical implementation
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build and run the containerized application
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

## Architecture Overview

### ML Pipeline Flow
This project implements a complete MLOps pipeline with two distinct phases:

**Training Pipeline** (`scripts/run_pipeline.py`):
1. **Data Loading** → **Data Validation** (Great Expectations) → **Preprocessing** → **Feature Engineering** → **XGBoost Training** → **MLflow Logging**
2. All artifacts (model, feature columns, preprocessing logic) are stored in MLflow for reproducibility

**Serving Pipeline** (`src/app/main.py` + `src/serving/inference.py`):
1. **FastAPI REST API** (`/predict` endpoint) + **Gradio Web UI** (`/ui` endpoint) → **MLflow Model Loading** → **Feature Transformation** → **Prediction**
2. Feature processing mirrors training-time transformations for consistency

### MLflow Integration Patterns
- **Experiment Name**: "Telco Churn" (default, can be overridden)
- **Tracking URI**: File-based at `{project_root}/mlruns`
- **Logged Artifacts**: `model/`, `feature_columns.txt`, `preprocessing.pkl`
- **Tracked Metrics**: precision, recall, f1, roc_auc, train_time, pred_time, data_quality_pass
- **Parameters**: model hyperparameters, threshold, split seed, dataset SHA-256, row/column counts, and Python/library versions

### Feature Engineering Consistency
Critical pattern: Training and serving must use identical feature transformations.

**Training** (`src/features/build_features.py`):
- Binary features (Yes/No, Male/Female) → deterministic 0/1 mapping
- Multi-category features → one-hot encoding with `drop_first=True`
- Boolean columns → integers

**Serving** (`src/serving/inference.py`):
- Loads fitted binary/category mappings from the selected model's `preprocessing.pkl`
- Applies the training-time category schema to individual requests
- Aligns features to the paired `feature_columns.txt` in the same model bundle

### Model Loading and Serving
- **Container Path**: Model loaded from `/app/model` (MLflow sklearn format)
- **Feature Order**: Enforced using `feature_columns.txt` from training
- **Prediction Format**: Returns "Likely to churn" or "Not likely to churn" strings

### Data Validation
- **Tool**: Great Expectations with custom validation suite
- **Location**: `src/utils/validate_data.py`
- **Checks**: CustomerID presence, gender values, numeric ranges for tenure/charges
- **Integration**: Results logged to MLflow as `data_quality_pass` metric

### Docker Containerization
- **Base Image**: `python:3.11-slim`
- **Key Setting**: `PYTHONPATH=/app/src` for proper module imports
- **Model Artifacts**: Specific MLflow run copied to `/app/model` during build
- **Serving**: uvicorn with FastAPI app on port 8000

### CI/CD Pipeline
- **Trigger**: Pull requests and pushes to main
- **Actions**: Run pytest; on main, build and smoke-test Docker image before pushing to Docker Hub
- **Requirements**: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets
- **Deployment**: Manual ECS service update (AWS Fargate + ALB)

## Key Implementation Details

### XGBoost Model Configuration
Optimized hyperparameters are hardcoded in `scripts/run_pipeline.py:100-110`:
- `n_estimators=301`, `learning_rate=0.034`, `max_depth=7`
- `scale_pos_weight` calculated dynamically for class imbalance handling

### API Endpoints
- `GET /` - Health check returning `{"status": "ok"}`
- `POST /predict` - Accepts `CustomerData` Pydantic model with 18 customer attributes
- `/ui` - Gradio interface mounted via `gr.mount_gradio_app()`

### File System Layout
- `data/raw/` - Original datasets
- `data/processed/` - Canonical cleaned CSVs (categoricals remain unencoded)
- `mlruns/` - MLflow experiment tracking database
- `src/serving/model/production/` - Promoted model, feature schema, preprocessing mappings, and threshold
- `artifacts/` - Legacy/generated metadata; not used for serving model selection

### Development Notes
- Automated regression tests live in `tests/` and run in CI
- MLflow UI can be accessed with: `mlflow ui --backend-store-uri file:./mlruns`
- The project uses file-based MLflow tracking (not a tracking server)
- Model serving expects exact feature column order from training time