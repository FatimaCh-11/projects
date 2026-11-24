"""Run an experimental classification on fingerprint feature rows."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("outputs/fingerprint_research_model.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV containing the trained model's fingerprint feature columns.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python src/train_model.py' first."
        )

    saved = joblib.load(MODEL_PATH)
    data = pd.read_csv(args.input)

    feature_columns = saved["feature_columns"]
    missing = set(feature_columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing)}"
        )

    predictions = saved["model"].predict(data[feature_columns])
    result = data.copy()
    result["experimental_model_output"] = predictions
    print(result.to_string(index=False))
    print(
        "\nEducational output only. Do not use this to determine blood type "
        "or make healthcare decisions."
    )


if __name__ == "__main__":
    main()