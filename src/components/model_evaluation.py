# src/model_evaluation.py
import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")
fh = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))
ch = logging.StreamHandler()
for h in (fh, ch):
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)


def load_pickle(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.debug("Loaded object from %s", path)
    return obj


def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    logger.debug("Test data loaded from %s with shape %s", path, df.shape)
    return df


def evaluate_model(model, vectorizer, test_df: pd.DataFrame):
    X_test_tfidf = vectorizer.transform(test_df["clean_comment"].values)
    y_test = test_df["category"].values
    y_pred = model.predict(X_test_tfidf)

    report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    fig_path = "reports/confusion_matrix.png"
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(fig_path)
    plt.close()

    metrics_path = "reports/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info("Evaluation complete. Saved metrics and confusion matrix.")

    return report, cm, fig_path, metrics_path
