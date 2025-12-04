import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.utils.main_utils import load_data, load_json, load_object 
from src.logger import logging


def log_confusion_matrix(cm, classes, artifact_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    plt.savefig(artifact_path)
    plt.close()


def evaluate_model():
    try:
        # metadata saved by model_trainer
        trainer_output = load_json("artifacts/model_trainer/output.json")

        run_id = trainer_output["run_id"]
        vectorizer_path = "artifacts/data_transformation/tfidf_vectorizer.pkl"
        model_uri = f"runs:/{run_id}/model"
        test_data_path = "artifacts/data_preprocessing/test_processed.csv"


        # Load test data & vectorizer
        test_data = load_data(test_data_path)
        X_test, y_test = test_data["clean_comment"], test_data["category"]

        vectorizer = load_object(vectorizer_path)
        X_test_transformed = vectorizer.transform(X_test).tocsr()

        # ------------------------continuing MLflow run and log metrics------------------------
        logging.info(f"Continuing MLflow run: {run_id}")
        mlflow.set_tracking_uri("http://ec2-51-20-74-217.eu-north-1.compute.amazonaws.com:5000")
        mlflow.set_experiment("Training Pipeline")

        # load model
        model = mlflow.lightgbm.load_model(model_uri)

        # computing sklearn metrics
        y_pred = model.predict(X_test_transformed)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)

        logging.debug("MLflow tracking URI and experiment is set")
        with mlflow.start_run(run_id=run_id):

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_f1_weighted", f1)
            mlflow.log_metric("test_precision_weighted", precision)
            mlflow.log_metric("test_recall_weighted", recall)

            # Log confusion matrix image
            cm_path = "artifacts/model_evaluation/confusion_matrix.png"
            log_confusion_matrix(cm, classes=sorted(y_test.unique()), artifact_path=cm_path)
            mlflow.log_artifact(cm_path)

            logging.info("Logged evaluation metrics & confusion matrix to MLflow.")

        # ------------------------continuing MLflow run and log metrics------------------------
        logging.info("Model evaluation completed.")

    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    evaluate_model()
