## Telco Churn – End-to-End ML Project
### Purpose

Build a machine-learning solution for telecom customer churn, with repeatable data preparation and training, a FastAPI prediction API, and a Gradio web UI deployable as a public Render web service.

### Problem solved & benefits

- Faster decisions: Predicts which customers are likely to churn so teams can act before they leave.
- Operationalized ML: Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- Repeatable delivery: CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- Traceable experiments: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST) and a root health check /.
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000.
- CI/CD: GitHub Actions runs tests and a container smoke test; Render is configured to deploy after checks pass.
- Hosting: Render Docker web service with managed public HTTPS, configured via `render.yaml`.

### Deployment flow (high-level)

- Push to main → GitHub Actions runs the automated tests and container smoke test.
- After checks pass, Render builds the Dockerfile and deploys the web service.
- Render checks `/` for health; users open `/ui` for Gradio or call `POST /predict` for predictions.

### Selecting and packaging the serving model

The serving app loads exactly one bundle from `MODEL_DIR`; it no longer guesses a model based on filesystem timestamps. First run the training pipeline, then choose the resulting finished MLflow run ID and package that run before building the Docker image. The package command checks that the run passed data validation and that its model, feature list, fitted preprocessing schema, and classification threshold are present and consistent. Serving applies that run's threshold to its predicted churn probability.

The default bundle destination is `src/serving/model/production`, which is included in the repository and copied by the Dockerfile to `/app/model`. Commit the generated bundle when promoting a model so CI builds use the same version. Set `MODEL_DIR` to another complete bundle directory for local or alternate deployments. Do not copy a model from one run and preprocessing files from another.

```powershell
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
python scripts/package_model.py --run-id <finished-mlflow-run-id>
docker build -t telco-churn .
```

### Deploy the live app on Render

This repository includes a Render Blueprint in `render.yaml`. After the latest project commit is pushed to GitHub, connect the repository to Render and create a Blueprint from that file. Render will build the Docker image and deploy the FastAPI service. The interactive Gradio page is at `/ui`; the prediction API is `POST /predict`, and `/` is the health check. The service binds to Render's `PORT` setting (10000 in the Blueprint, 8000 for local Docker runs).

The Blueprint uses Render's free web-service plan where available; free services may sleep while idle and take time to wake. For a continuously available demo or heavier traffic, select a paid plan. The app itself does not require application secrets.

The GitHub repository must contain the deployment commit before Render can build it. If you don't have write access, push the changes to a fork and connect that repository to Render.

### Deployment notes

- The API loads the exact bundle in `src/serving/model/production` by default, or the directory selected by `MODEL_DIR`.
- The Docker container honors Render's `PORT`; local runs default to port 8000.
- The model bundle metadata targets Python 3.12, while the current Docker base is Python 3.11. The container build/smoke workflow must pass before the public service is considered ready.
