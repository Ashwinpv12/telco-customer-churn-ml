import pandas as pd


def _binary_mapping(series: pd.Series) -> dict[str, int]:
    values = sorted(series.dropna().astype(str).unique().tolist())
    if set(values) == {"Yes", "No"}:
        return {"No": 0, "Yes": 1}
    if set(values) == {"Male", "Female"}:
        return {"Female": 0, "Male": 1}
    return {value: index for index, value in enumerate(values)}


def fit_feature_encoder(
    df: pd.DataFrame, target_col: str = "Churn"
) -> tuple[pd.DataFrame, dict]:
    """Fit categorical mappings on training data and return encoded data and schema."""
    df = df.copy()
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    categorical_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    binary_mappings = {c: _binary_mapping(df[c]) for c in binary_cols}
    for column, mapping in binary_mappings.items():
        df[column] = df[column].astype("string").map(mapping).fillna(0).astype(int)

    categories = {
        c: sorted(df[c].dropna().astype(str).unique().tolist())
        for c in categorical_cols
    }
    if categorical_cols:
        for column in categorical_cols:
            df[column] = pd.Categorical(df[column].astype("string"), categories=categories[column])
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    feature_columns = [column for column in df.columns if column != target_col]
    schema = {
        "binary_mappings": binary_mappings,
        "categorical_categories": categories,
        "categorical_columns": categorical_cols,
        "feature_columns": feature_columns,
        "target": target_col,
    }
    return df, schema


def transform_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Transform raw rows using only categorical mappings fitted during training."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = schema.get("target", "Churn")
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    for column, mapping in schema["binary_mappings"].items():
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().map(mapping).fillna(0).astype(int)

    categorical_categories = schema["categorical_categories"]
    categorical_cols = [c for c in schema["categorical_columns"] if c in df.columns]
    for column in categorical_cols:
        df[column] = pd.Categorical(
            df[column].astype("string"), categories=categorical_categories[column]
        )
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    return df.reindex(columns=schema["feature_columns"], fill_value=0)


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """
    encoded, _ = fit_feature_encoder(df, target_col)
    return encoded

