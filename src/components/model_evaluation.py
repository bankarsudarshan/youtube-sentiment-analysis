# src/model_evaluation.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.logger import logging as logger


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
