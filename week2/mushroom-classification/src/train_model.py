"""Train mushroom classifiers and measure poisonous-class sensitivity."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo


OUTPUT_DIR = Path("outputs")
POISONOUS_LABEL = "p"


def make_pipeline(classifier):
    """Encode categorical mushroom features, then fit a classifier."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(range(22)),
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def threshold_for_full_validation_sensitivity(model, x_val, y_val):
    """Choose the highest threshold that catches all validation poison cases."""
    probabilities = model.predict_proba(x_val)
    class_labels = model.named_steps["classifier"].classes_
    poisonous_index = list(class_labels).index(POISONOUS_LABEL)
    poisonous_probabilities = probabilities[:, poisonous_index]

    poison_mask = y_val.to_numpy() == POISONOUS_LABEL
    lowest_poison_probability = poisonous_probabilities[poison_mask].min()

    # A tiny step below the minimum includes examples equal to that minimum.
    return float(np.nextafter(lowest_poison_probability, -np.inf))


def predict_with_threshold(model, threshold, x):
    """Predict poisonous when its probability meets the chosen threshold."""
    class_labels = model.named_steps["classifier"].classes_
    poisonous_index = list(class_labels).index(POISONOUS_LABEL)
    probabilities = model.predict_proba(x)[:, poisonous_index]

    return np.where(probabilities >= threshold, POISONOUS_LABEL, "e")


def report_results(name, y_true, y_pred):
    """Print metrics, treating poisonous as the positive class."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["e", "p"],
    )
    tn, fp, fn, tp = matrix.ravel()

    print(f"\n{name}")
    print(f"Accuracy:                 {accuracy_score(y_true, y_pred):.3f}")
    print(
        "Poisonous sensitivity:    "
        f"{recall_score(y_true, y_pred, pos_label='p'):.3f}"
    )
    print(
        "Poisonous precision:      "
        f"{precision_score(y_true, y_pred, pos_label='p', zero_division=0):.3f}"
    )
    print(f"Edible specificity:        {tn / (tn + fp):.3f}")
    print(f"False negatives (poison):  {fn}")
    print("Confusion matrix (rows=true [edible, poisonous], "
          "columns=predicted [edible, poisonous]):")
    print(matrix)


def main():
    # Fetch the official UCI dataset. The target is edible (e) or poisonous (p).
    mushroom = fetch_ucirepo(id=73)
    features = mushroom.data.features.copy()
    target = mushroom.data.targets.iloc[:, 0].astype(str).str.lower()

    # The dataset encodes missing stalk-root values with '?'.
    features = features.replace("?", np.nan)

    # Reserve a test set that is not used to train or calibrate thresholds.
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    # Split the remaining data into a model-training set and threshold-validation set.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_models = {}

    for name, classifier in models.items():
        model = make_pipeline(classifier)
        model.fit(x_train, y_train)

        # Report normal model predictions on the untouched test set.
        default_predictions = model.predict(x_test)
        report_results(
            f"{name} — default threshold, test set",
            y_test,
            default_predictions,
        )

        # Set a threshold using validation data, then evaluate on test data.
        threshold = threshold_for_full_validation_sensitivity(
            model,
            x_val,
            y_val,
        )
        sensitive_predictions = predict_with_threshold(
            model,
            threshold,
            x_test,
        )
        report_results(
            f"{name} — validation-calibrated threshold, test set",
            y_test,
            sensitive_predictions,
        )
        print(f"Threshold selected on validation data: {threshold:.6f}")

        saved_models[name] = {
            "model": model,
            "poisonous_threshold": threshold,
            "features": list(features.columns),
        }

    model_path = OUTPUT_DIR / "mushroom_models.joblib"
    joblib.dump(saved_models, model_path)
    print(f"\nSaved trained models to {model_path}")


if __name__ == "__main__":
    main()