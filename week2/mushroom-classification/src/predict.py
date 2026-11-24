"""Classify mushrooms from a CSV of UCI-format feature rows."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("outputs/mushroom_models.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV file containing mushroom feature columns.",
    )
    parser.add_argument(
        "--model",
        choices=["Decision Tree", "Random Forest"],
        default="Random Forest",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. Run 'python src/train_model.py' first."
        )

    saved_models = joblib.load(MODEL_PATH)
    saved = saved_models[args.model]
    model = saved["model"]
    threshold = saved["poisonous_threshold"]

    rows = pd.read_csv(args.input)
    missing_columns = set(saved["features"]) - set(rows.columns)
    if missing_columns:
        raise ValueError(
            "Input CSV is missing feature columns: "
            + ", ".join(sorted(missing_columns))
        )

    rows = rows[saved["features"]].replace("?", pd.NA)

    class_labels = model.named_steps["classifier"].classes_
    poisonous_index = list(class_labels).index("p")
    probabilities = model.predict_proba(rows)[:, poisonous_index]
    predictions = ["p" if value >= threshold else "e" for value in probabilities]

    results = pd.DataFrame(
        {
            "prediction": predictions,
            "poisonous_probability": probabilities.round(4),
        }
    )
    results["prediction"] = results["prediction"].map(
        {"p": "poisonous", "e": "edible"}
    )

    print(results.to_string(index=False))
    print("\nClassification output only; do not use it to decide whether to eat a mushroom.")


if __name__ == "__main__":
    main()