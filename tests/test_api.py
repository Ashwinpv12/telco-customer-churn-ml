import pytest
from fastapi.testclient import TestClient

from src.app.app import app as compatibility_app
from src.app.main import app
from src.serving.inference import FEATURE_COLS, MODEL_DIR, PREPROCESSING_SCHEMA, model


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def customer_payload() -> dict:
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.75,
    }


def test_health_endpoint_returns_ok(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_legacy_app_module_reexports_canonical_app():
    assert compatibility_app is app


def test_predict_endpoint_returns_model_prediction(client: TestClient, customer_payload: dict):
    response = client.post("/predict", json=customer_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] in {"Likely to churn", "Not likely to churn"}


def test_predict_endpoint_validates_required_fields(client: TestClient, customer_payload: dict):
    customer_payload.pop("gender")

    response = client.post("/predict", json=customer_payload)

    assert response.status_code == 422


def test_predict_endpoint_returns_server_error_on_inference_failure(
    client: TestClient, customer_payload: dict, monkeypatch: pytest.MonkeyPatch
):
    def fail_prediction(_data: dict) -> str:
        raise RuntimeError("inference failed")

    monkeypatch.setattr("src.app.main.predict", fail_prediction)
    response = client.post("/predict", json=customer_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "inference failed"


def test_production_model_bundle_has_matching_feature_schema():
    assert MODEL_DIR.endswith("production")
    assert FEATURE_COLS == PREPROCESSING_SCHEMA["feature_columns"]
    assert callable(model.predict_proba)
