import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pytest
from sklearn.linear_model import LogisticRegression

from scripts.package_model import package_run, validate_bundle


def test_validate_bundle_rejects_schema_mismatch(tmp_path: Path):
    (tmp_path / "MLmodel").write_text("model metadata", encoding="utf-8")
    (tmp_path / "feature_columns.txt").write_text("feature_a\n", encoding="utf-8")
    joblib.dump({"feature_columns": ["feature_b"], "categorical_categories": {}}, tmp_path / "preprocessing.pkl")
    (tmp_path / "bundle.json").write_text('{"decision_threshold": 0.35}', encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        validate_bundle(tmp_path)


def test_package_run_bundles_model_and_metadata_from_selected_run(tmp_path: Path):
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("bundle-test")

    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_metric("data_quality_pass", 1)
        mlflow.log_param("threshold", 0.35)
        model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_text("feature_a", "feature_columns.txt")
        schema = {
            "feature_columns": ["feature_a"],
            "binary_mappings": {},
            "categorical_categories": {},
            "categorical_columns": [],
            "target": "Churn",
        }
        schema_path = tmp_path / "preprocessing.pkl"
        joblib.dump(schema, schema_path)
        mlflow.log_artifact(str(schema_path))

    output_dir = tmp_path / "published" / "serving_model"
    package_run(run_id, tracking_dir.resolve().as_uri(), output_dir)

    assert (output_dir / "MLmodel").is_file()
    assert joblib.load(output_dir / "preprocessing.pkl") == schema
    manifest = json.loads((output_dir / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["decision_threshold"] == 0.35
    validate_bundle(output_dir)
