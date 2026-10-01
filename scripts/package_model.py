#!/usr/bin/env python3
"""Package a model and its preprocessing artifacts from one explicit MLflow run."""

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import joblib
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_bundle(bundle_dir: Path) -> None:
    """Reject incomplete or mismatched model/preprocessing bundles."""
    model_metadata = bundle_dir / "MLmodel"
    feature_file = bundle_dir / "feature_columns.txt"
    preprocessing_file = bundle_dir / "preprocessing.pkl"
    manifest_file = bundle_dir / "bundle.json"
    required_files = (model_metadata, feature_file, preprocessing_file, manifest_file)
    missing = [str(path.name) for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(f"Model bundle is missing required files: {', '.join(missing)}")

    with feature_file.open(encoding="utf-8") as file:
        feature_columns = [line.strip() for line in file if line.strip()]
    if not feature_columns:
        raise ValueError("feature_columns.txt is empty")

    schema = joblib.load(preprocessing_file)
    if not isinstance(schema, dict):
        raise ValueError("preprocessing.pkl must contain a mapping")
    if schema.get("feature_columns") != feature_columns:
        raise ValueError("Feature columns do not match preprocessing.pkl")
    if "categorical_categories" not in schema:
        raise ValueError(
            "Preprocessing artifact has no fitted category mappings; retrain with the updated pipeline"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    try:
        threshold = float(manifest["decision_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Bundle manifest must contain a numeric decision_threshold") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1")


def package_run(run_id: str, tracking_uri: str, output_dir: Path) -> Path:
    """Download one completed MLflow run and atomically publish its serving bundle."""
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run {run_id} is not finished (status={run.info.status})")
    if run.data.metrics.get("data_quality_pass") != 1:
        raise ValueError(f"MLflow run {run_id} did not record a passing data-quality check")
    try:
        decision_threshold = float(run.data.params["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"MLflow run {run_id} has no valid threshold parameter") from error
    if not math.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise ValueError(f"MLflow run {run_id} threshold must be between 0 and 1")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-bundle-", dir=output_dir.parent) as temp_name:
        staging_dir = Path(temp_name) / "bundle"
        staging_dir.mkdir()
        downloaded_model = Path(client.download_artifacts(run_id, "model", temp_name))
        if not (downloaded_model / "MLmodel").is_file():
            raise ValueError(f"Run {run_id} does not contain an MLflow model at artifacts/model")
        shutil.copytree(downloaded_model, staging_dir, dirs_exist_ok=True)

        for artifact_name in ("feature_columns.txt", "preprocessing.pkl"):
            downloaded_file = Path(client.download_artifacts(run_id, artifact_name, temp_name))
            shutil.copy2(downloaded_file, staging_dir / artifact_name)

        manifest = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "source_artifact_path": "model",
            "decision_threshold": decision_threshold,
        }
        (staging_dir / "bundle.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        validate_bundle(staging_dir)

        backup_dir = output_dir.with_name(f"{output_dir.name}.previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if output_dir.exists():
            output_dir.replace(backup_dir)
        try:
            staging_dir.replace(output_dir)
        except Exception:
            if backup_dir.exists() and not output_dir.exists():
                backup_dir.replace(output_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Exact finished MLflow run ID to package")
    parser.add_argument(
        "--tracking-uri",
        default=Path(PROJECT_ROOT, "mlruns").resolve().as_uri(),
        help="MLflow tracking URI (default: this project's local mlruns directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "src" / "serving" / "model" / "production",
        help="Destination model bundle directory",
    )
    args = parser.parse_args()
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    destination = package_run(args.run_id, args.tracking_uri, args.output)
    print(f"Packaged MLflow run {args.run_id} to {destination}")


if __name__ == "__main__":
    main()
