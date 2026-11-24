"""Train and compare models that predict students' final course grades."""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
TARGET = "G3"


def load_data(path=None, course="math"):
    """Load a local CSV, or download a course dataset from UCI."""
    if path:
        data = pd.read_csv(path, sep=None, engine="python")
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            archive_data = response.read()

        with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
            filename = (
                "student/student-mat.csv"
                if course == "math"
                else "student/student-por.csv"
            )
            with archive.open(filename) as csv_file:
                data = pd.read_csv(csv_file, sep=";")

    if TARGET not in data.columns:
        raise ValueError(f"The dataset must contain a {TARGET!r} column.")

    return data


def make_pipeline(model, numeric_columns, categorical_columns):
    """Create preprocessing steps followed by a regression model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Optional path to a local CSV file.",
    )
    parser.add_argument(
        "--course",
        choices=["math", "portuguese"],
        default="math",
        help="UCI course dataset to download when --data is not used.",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=10,
        help="Flag predictions below this grade (UCI grades range from 0 to 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Folder where result CSV files will be saved.",
    )
    args = parser.parse_args()

    data = load_data(args.data, args.course)
    data = data.dropna(subset=[TARGET]).copy()

    # G3 is the final grade; all remaining columns are model features.
    features = data.drop(columns=[TARGET])
    target = data[TARGET].astype(float)

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, estimator in models.items():
        pipeline = make_pipeline(
            estimator,
            numeric_columns,
            categorical_columns,
        )
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        results.append(
            {
                "model": name,
                "r2": r2_score(y_test, predictions),
                "rmse": mean_squared_error(y_test, predictions) ** 0.5,
            }
        )
        trained_models[name] = pipeline

    metrics = pd.DataFrame(results).sort_values("rmse")
    print("Model scores on the held-out test set:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # Use the model with the lowest test RMSE to make the support-review list.
    best_model_name = metrics.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    all_predictions = best_model.predict(features)

    flagged_mask = all_predictions < args.support_threshold
    support_list = data.loc[flagged_mask].copy()
    support_list.insert(
        0,
        "predicted_G3",
        all_predictions[flagged_mask].round(2),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    support_list.to_csv(
        args.output_dir / "student/students_for_support.csv",
        index=False,
    )

    print(f"\nBest model by RMSE: {best_model_name}")
    print(f"Students flagged for review: {len(support_list)}")
    print(f"Results saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()