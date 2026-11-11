"""Download, train, and evaluate the SMS spam classification pipeline."""
from pathlib import Path
import json
import zipfile
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
TSV_PATH = DATA_DIR / "SMSSpamCollection"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

def download_dataset():
    if TSV_PATH.exists():
        print("Using existing SMS Spam Collection dataset.")
        return
    print("Downloading SMS Spam Collection from the UCI Machine Learning Repository...")
    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            member = next((n for n in archive.namelist() if n.endswith("SMSSpamCollection")), None)
            if member is None:
                raise RuntimeError("The downloaded archive does not contain SMSSpamCollection.")
            TSV_PATH.write_bytes(archive.read(member))
    except Exception as exc:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection and retry. "
            f"Dataset source: {DATA_URL}\nOriginal error: {exc}"
        ) from exc

def main():
    download_dataset()
    df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"].str.len() > 0]
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20,
        random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    print(f"Training on {len(X_train):,} messages...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spam": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "test_samples": int(len(y_test)),
        "dataset_samples": int(len(df)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4%}")
    print(f"Spam precision: {metrics['precision_spam']:.4%}")
    print(f"Spam recall: {metrics['recall_spam']:.4%}")
    print(f"Spam F1: {metrics['f1_spam']:.4%}")
    print(f"Metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
