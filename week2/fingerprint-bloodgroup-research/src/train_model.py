"""Educational experiment: test fingerprint-feature blood-group classification."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


DATA_PATH = Path("data/fingerprint_data.csv")
OUTPUT_DIR = Path("outputs")

GROUP_COLUMN = "participant_id"
TARGET_COLUMN = "blood_group"


def group_split(features, target, groups, test_size, random_state):
    """Split rows while keeping each participant in only one partition."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        splitter.split(features, target, groups)
    )
    return train_indices, test_indices


def make_pipeline(model, numeric_columns, categorical_columns):
    """Impute and prepare features, then fit a classifier."""
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(transformers=transformers)

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", model),
        ]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Prepare your consented, "
            "lab-labeled dataset first."
        )

    data = pd.read_csv(DATA_PATH)

    required = {GROUP_COLUMN, TARGET_COLUMN}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.dropna(subset=[GROUP_COLUMN, TARGET_COLUMN]).copy()
    groups = data[GROUP_COLUMN]
    target = data[TARGET_COLUMN].astype(str)
    features = data.drop(columns=[GROUP_COLUMN, TARGET_COLUMN])

    if features.empty:
        raise ValueError("Add at least one fingerprint measurement column.")

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Hold out participants for final evaluation.
    train_validation_idx, test_idx = group_split(
        features,
        target,
        groups,
        test_size=0.20,
        random_state=42,
    )

    x_train_validation = features.iloc[train_validation_idx]
    y_train_validation = target.iloc[train_validation_idx]
    groups_train_validation = groups.iloc[train_validation_idx]

    x_test = features.iloc[test_idx]
    y_test = target.iloc[test_idx]

    # Separate validation participants to choose between models.
    train_idx, validation_idx = group_split(
        x_train_validation,
        y_train_validation,
        groups_train_validation,
        test_size=0.25,
        random_state=43,
    )

    x_train = x_train_validation.iloc[train_idx]
    y_train = y_train_validation.iloc[train_idx]
    x_validation = x_train_validation.iloc[validation_idx]
    y_validation = y_train_validation.iloc[validation_idx]

    candidate_models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42,
        ),
    }

    fitted_models = {}
    validation_scores = {}

    for name, classifier in candidate_models.items():
        pipeline = make_pipeline(
            classifier,
            numeric_columns,
            categorical_columns,
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_validation)

        score = balanced_accuracy_score(y_validation, predictions)
        fitted_models[name] = pipeline
        validation_scores[name] = score
        print(f"{name} validation balanced accuracy: {score:.3f}")

    best_name = max(validation_scores, key=validation_scores.get)
    best_model = fitted_models[best_name]

    # Evaluate the validation-selected model once on held-out participants.
    test_predictions = best_model.predict(x_test)
    print(f"\nSelected model: {best_name}")
    print(
        "Held-out test balanced accuracy: "
        f"{balanced_accuracy_score(y_test, test_predictions):.3f}"
    )
    print("\nHeld-out test classification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "fingerprint_research_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_columns": features.columns.tolist(),
        },
        model_path,
    )
    print(f"Saved research model to {model_path}")


if __name__ == "__main__":
    main()