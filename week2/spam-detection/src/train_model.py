"""Train and compare spam classifiers using the UCI SMS Spam Collection."""

import argparse
import io
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms%2Bspam%2Bcollection.zip"
)
OUTPUT_DIR = Path("outputs")
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()


def tokenize_and_stem(text):
    words = re.findall(r"(?u)\b[a-zA-Z][a-zA-Z']+\b", str(text).lower())
    return [STEMMER.stem(word) for word in words if word not in STOP_WORDS]


def load_data(csv_path=None):
    """Read a local CSV, or download and read the UCI ZIP."""
    if csv_path:
        data = pd.read_csv(csv_path)

        if not {"label", "message"}.issubset(data.columns):
            raise ValueError(
                "Your CSV must have columns named 'label' and 'message'."
            )
    else:
        print("Downloading the UCI SMS Spam Collection...")
        with urlopen(DATA_URL, timeout=60) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            data_filename = next(
                name for name in archive.namelist()
                if Path(name).name == "SMSSpamCollection"
            )
            text = archive.read(data_filename).decode(
                "utf-8",
                errors="replace",
            )

        data = pd.read_csv(
            io.StringIO(text),
            sep="\t",
            header=None,
            names=["label", "message"],
        )

    messages = data["message"].fillna("").astype(str)
    labels = data["label"].astype(str).str.lower().str.strip()

    labels = labels.replace(
        {
            "legitimate": "ham",
            "not spam": "ham",
            "0": "ham",
            "junk": "spam",
            "1": "spam",
        }
    )

    if not set(labels.unique()).issubset({"ham", "spam"}):
        raise ValueError(
            "Labels must be 'ham' or 'spam'. "
            f"Found: {sorted(labels.unique())}"
        )

    return messages, labels


def make_pipeline(classifier):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize_and_stem,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    max_features=50000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Optional local CSV with 'label' and 'message' columns.",
    )
    args = parser.parse_args()

    messages, labels = load_data(args.data)
    print(f"Loaded {len(messages)} messages.")
    print(f"Class counts:\n{labels.value_counts().to_string()}")

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_models = {}

    for name, classifier in models.items():
        model = make_pipeline(classifier)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        print(f"\n{name} — held-out test set")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("Confusion matrix (rows/columns: ham, spam):")
        print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
        print(
            classification_report(
                y_test,
                predictions,
                labels=["ham", "spam"],
                zero_division=0,
            )
        )

        saved_models[name] = model

    model_path = OUTPUT_DIR / "spam_models.joblib"
    joblib.dump(saved_models, model_path)
    print(f"Saved trained models to {model_path}")


if __name__ == "__main__":
    main()