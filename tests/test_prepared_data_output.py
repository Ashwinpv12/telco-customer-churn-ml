from pathlib import Path

import pandas as pd

from scripts.prepare_processed_data import prepare_processed_data


def test_prepared_csv_is_cleaned_but_not_feature_encoded(tmp_path: Path):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "TotalCharges": ["10.5", "20", "30"],
            "Churn": ["No", "Yes", "No"],
        }
    ).to_csv(raw_path, index=False)

    cleaned = prepare_processed_data(raw_path, output_path)
    saved = pd.read_csv(output_path)

    assert "customerID" not in saved.columns
    assert "Contract" in saved.columns
    assert not any(column.startswith("Contract_") for column in saved.columns)
    assert saved["Churn"].tolist() == [0, 1, 0]
    assert pd.api.types.is_numeric_dtype(saved["TotalCharges"])
    assert saved.columns.tolist() == cleaned.columns.tolist()
    assert saved["SeniorCitizen"].tolist() == cleaned["SeniorCitizen"].tolist()
    assert saved["Contract"].tolist() == cleaned["Contract"].tolist()
