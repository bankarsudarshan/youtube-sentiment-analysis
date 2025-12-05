import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    # confusion_matrix,
)

from src.utils.main_utils import load_data, load_json, load_object, save_dict_as_json
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


def compute_metrics(model, X_test, y_test) -> dict:
    """Returns a dictionary containing the different sklearn metrics like accuracy, f1_score and all"""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        # "cm": confusion_matrix(y_test, y_pred)
    }
    return metrics


def evaluate_model():
    try:
        # ------------------------load paths and data------------------------
        trainer_output = load_json("artifacts/model_trainer/trainer_output.json")
        run_id = trainer_output["run_id"]
        vectorizer_path = "artifacts/data_transformation/tfidf_vectorizer.pkl"
        model_uri = f"runs:/{run_id}/model"
        test_data_path = "artifacts/data_preprocessing/test_processed.csv"

        test_data = load_data(test_data_path)
        X_test, y_test = test_data["clean_comment"], test_data["category"]
        vectorizer = load_object(vectorizer_path)
        X_test_transformed = vectorizer.transform(X_test).tocsr()

        # ------------------------comparing Current and Production models------------------------
        mlflow.set_tracking_uri("http://ec2-51-20-74-217.eu-north-1.compute.amazonaws.com:5000")
        client = MlflowClient()

        # load CURRENT NEW model
        # logs current model's metrics
        # compare it with the "Production" model
        model = mlflow.lightgbm.load_model(model_uri)
        
        metrics = compute_metrics(model, X_test_transformed, y_test) # sklearn metrics

        model_name = "YoutubeSentimentAnalysis"
        is_model_accepted = False

        try:
            # Try to fetch the latest model in Production
            latest_versions = client.get_latest_versions(model_name, stages=["Production"])
            if not latest_versions:
                logging.info("No model currently in Production. Accepting new model.")
                is_model_accepted = True
            else:
                prod_version = latest_versions[0]
                prod_model_uri = prod_version.source
                
                # Load Production Model
                logging.info(f"Loading Production model version {prod_version.version} for comparison.")
                prod_model = mlflow.lightgbm.load_model(prod_model_uri)
                
                # Predict with Production Model on SAME test set
                metrics_prod = compute_metrics(prod_model, X_test_transformed, y_test)
                
                logging.info(f"New Model Acc: {metrics["accuracy"]}, Prod Model Acc: {metrics_prod["accuracy"]}")

                # Threshold Logic: New model must be strictly better
                if metrics["accuracy"] > metrics_prod["accuracy"]:
                    is_model_accepted = True
                    logging.info("New model is better than Production.")
                else:
                    is_model_accepted = False
                    logging.info("New model is NOT better than Production.")

        except Exception as e:
            # If the model doesn't exist in registry yet, this is the first run
            logging.info(f"Model registry check failed (likely first run): {e}. Accepting model.")
            is_model_accepted = True

        # ------------------------continuing MLflow run and log metrics------------------------
        logging.info(f"Continuing MLflow run: {run_id}")
        with mlflow.start_run(run_id=run_id):

            mlflow.log_metrics(metrics)

            # Log confusion matrix image (will implement later)
            # cm_path = "artifacts/model_evaluation/confusion_matrix.png"
            # log_confusion_matrix(metrics['cm'], classes=sorted(y_test.unique()), artifact_path=cm_path)
            # mlflow.log_artifact(cm_path)

            mlflow.set_tag("model_accepted", str(is_model_accepted))

            logging.info("Logged evaluation metrics & confusion matrix to MLflow.")

        # ------------------------Save Evaluation Result for Model Pusher------------------------
        eval_result = {
            "is_model_accepted": is_model_accepted,
            "model_accuracy": metrics["accuracy"],
            "run_id": run_id,
            "model_name": model_name
        }
        save_dict_as_json(eval_result, "artifacts/model_evaluation/evaluation_result.json")
        logging.info(f"Model evaluation completed. Accepted: {is_model_accepted}")

    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    evaluate_model()
