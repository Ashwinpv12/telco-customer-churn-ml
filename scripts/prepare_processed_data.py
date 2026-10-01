import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data, save_processed_data


def prepare_processed_data(
    input_path: str | Path, output_path: str | Path, target_col: str = "Churn"
) -> pd.DataFrame:
    """Load and clean data using the same stages and output format as training."""
    cleaned = preprocess_data(load_data(input_path), target_col=target_col)
    save_processed_data(cleaned, output_path)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the canonical cleaned Telco CSV")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv",
    )
    args = parser.parse_args()

    cleaned = prepare_processed_data(args.input, args.output)
    output_path = args.output.resolve()
    print(f"✅ Cleaned dataset saved to {output_path} | Shape: {cleaned.shape}")


if __name__ == "__main__":
    main()
