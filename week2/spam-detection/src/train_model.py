"""Train and compare spam classifiers on message text."""

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from ucimlrepo import fetch_ucirepo


OUTPUT_DIR = Path("outputs")
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()


def tokenize_and_stem(text):
    """Lowercase, tokenize, remove stop words, and stem a message."""
    words = re.findall(r"(?u)\b[a-zA-Z][a-zA-Z']+\b", str(text).lower())
    return [
        STEMMER.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]


def load_data(csv_path=None):
    """Load UCI SMS data, or a local CSV with label and message columns."""
    if csv_path:
        data = pd.read_csv(csv_path)
        required = {"label", "message"}
        if not required.issubset(data.columns):
            raise ValueError(
                "Local CSV must contain columns named 'label' and 'message'."
            )
        messages = data["message"].fillna("").astype(str)
        labels = data["label"].astype(str).str.lower().str.strip()
    else:
        dataset = fetch_ucirepo(id=228)
        messages = dataset.data.features.iloc[:, 0].fillna("").astype(str)
        labels = (
            dataset.data.targets.iloc[:, 0]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # Accept common equivalents for the two classes.
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
            "Labels must identify messages as 'ham' or 'spam'. "
            f"Found: {sorted(labels.unique())}"
        )

    return messages, labels


def make_pipeline(classifier):
    """Create a TF-IDF text pipeline followed by a classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize_and_stem,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
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
        help="Optional CSV with 'label' and 'message' columns.",
    )
    args = parser.parse_args()

    messages, labels = load_data(args.data)

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_models = {}

    for name, classifier in models.items():
        model = make_pipeline(classifier)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        print(f"\n{name} — held-out test set")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("Confusion matrix (rows=true [ham, spam], columns=predicted [ham, spam]):")
        print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
        print(classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            zero_division=0,
        ))

        saved_models[name] = model

    model_path = OUTPUT_DIR / "spam_models.joblib"
    joblib.dump(saved_models, model_path)
    print(f"Saved models to {model_path}")


if __name__ == "__main__":
    main()