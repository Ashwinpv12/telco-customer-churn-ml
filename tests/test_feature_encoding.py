import pandas as pd

from src.features.build_features import fit_feature_encoder, transform_features


def test_single_row_transform_matches_training_encoding_for_each_category():
    training = pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male"],
            "Partner": ["No", "Yes", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaymentMethod": [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "tenure": [1, 12, 24, 3],
            "Churn": [0, 1, 0, 1],
        }
    )

    encoded_training, schema = fit_feature_encoder(training)
    expected = encoded_training.drop(columns="Churn").reset_index(drop=True)
    transformed_batch = transform_features(training.drop(columns="Churn"), schema)

    pd.testing.assert_frame_equal(transformed_batch, expected)
    for index in range(len(training)):
        transformed_one = transform_features(training.iloc[[index]].drop(columns="Churn"), schema)
        pd.testing.assert_frame_equal(
            transformed_one.reset_index(drop=True), expected.iloc[[index]].reset_index(drop=True)
        )

    assert schema["binary_mappings"]["gender"] == {"Female": 0, "Male": 1}
    assert schema["categorical_categories"]["Contract"] == [
        "Month-to-month",
        "One year",
        "Two year",
    ]


def test_unseen_category_keeps_schema_and_encodes_as_reference():
    training = pd.DataFrame(
        {
            "InternetService": ["DSL", "Fiber optic", "No"],
            "Churn": [0, 1, 0],
        }
    )
    _, schema = fit_feature_encoder(training)

    transformed = transform_features(pd.DataFrame({"InternetService": ["Satellite"]}), schema)

    assert transformed.columns.tolist() == schema["feature_columns"]
    assert transformed.iloc[0].eq(0).all()