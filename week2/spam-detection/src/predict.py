"""Classify one message with a saved spam model."""

import argparse
from pathlib import Path

import joblib


MODEL_PATH = Path("outputs/spam_models.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        required=True,
        help="Message text to classify.",
    )
    parser.add_argument(
        "--model",
        choices=["Naive Bayes", "Linear SVM"],
        default="Linear SVM",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved models not found. Run 'python src/train_model.py' first."
        )

    models = joblib.load(MODEL_PATH)
    prediction = models[args.model].predict([args.text])[0]

    print(f"Model: {args.model}")
    print(f"Prediction: {prediction.upper()}")
    print("Review the original message before taking action.")


if __name__ == "__main__":
    main()