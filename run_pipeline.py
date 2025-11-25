import os
import pickle

import mlflow
from mlflow.tracking import MlflowClient

from src.model_building import (
    load_train_data,
    build_tfidf,
    train_lgbm,
    save_pickle,
)
from src.model_evaluation import load_test_data, evaluate_model


MLFLOW_TRACKING_URI = "http://<your-ec2-dns>:5000"  # set this or use env var
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sentiment_pipeline")


def train_and_eval_lgbm(config, train_df, test_df):
    # TF-IDF
    X_train, y_train, vectorizer = build_tfidf(
        train_df,
        max_features=config["tfidf_max_features"],
        ngram_range=tuple(config["tfidf_ngram_range"]),
    )

    # Train model
    model, train_time = train_lgbm(
        X_train,
        y_train,
        params={
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "n_estimators": config["n_estimators"],
        },
    )

    # Save artifacts locally
    vec_path = "models/tfidf_vectorizer.pkl"
    model_path = "models/lgbm_model.pkl"
    save_pickle(vectorizer, vec_path)
    save_pickle(model, model_path)

    # Evaluate
    report, cm, fig_path, metrics_path = evaluate_model(model, vectorizer, test_df)

    # Return everything for logging / comparison
    return {
        "model": model,
        "vectorizer": vectorizer,
        "train_time": train_time,
        "report": report,
        "cm": cm,
        "vec_path": vec_path,
        "model_path": model_path,
        "fig_path": fig_path,
        "metrics_path": metrics_path,
    }


def main():
    # Example: multiple configs (could be different models later)
    configs = [
        {
            "name": "lgbm_v1",
            "tfidf_max_features": 1000,
            "tfidf_ngram_range": [1, 2],
            "learning_rate": 0.09,
            "max_depth": 20,
            "n_estimators": 367,
        },
        {
            "name": "lgbm_v2",
            "tfidf_max_features": 2000,
            "tfidf_ngram_range": [1, 3],
            "learning_rate": 0.05,
            "max_depth": 25,
            "n_estimators": 400,
        },
    ]

    train_df = load_train_data("./data/processed/train_processed.csv")
    test_df = load_test_data("./data/processed/test_processed.csv")

    best_run = None
    best_score = -1.0
    client = MlflowClient()

    with mlflow.start_run(run_name="full_pipeline_run") as parent_run:
        mlflow.log_param("num_candidates", len(configs))

        for cfg in configs:
            with mlflow.start_run(
                run_name=cfg["name"], nested=True
            ) as child_run:
                # Log config
                for k, v in cfg.items():
                    if k != "name":
                        mlflow.log_param(k, v)

                result = train_and_eval_lgbm(cfg, train_df, test_df)

                # Log training time
                mlflow.log_metric("training_time_sec", result["train_time"])

                # Log metrics from classification_report (e.g. macro avg f1)
                report = result["report"]
                if "macro avg" in report:
                    for metric, value in report["macro avg"].items():
                        mlflow.log_metric(f"macro_{metric}", value)

                    score = report["macro avg"]["f1-score"]
                else:
                    # fallback: accuracy if present
                    score = report.get("accuracy", 0.0)
                    mlflow.log_metric("accuracy", score)

                # Log artifacts
                mlflow.log_artifact(result["vec_path"])
                mlflow.log_artifact(result["model_path"])
                mlflow.log_artifact(result["fig_path"])
                mlflow.log_artifact(result["metrics_path"])

                # Also log model in MLflow format
                mlflow.lightgbm.log_model(result["model"], artifact_path="model")

                # Track best model
                if score > best_score:
                    best_score = score
                    best_run = {
                        "run_id": child_run.info.run_id,
                        "config": cfg,
                        "model_path": result["model_path"],
                        "vec_path": result["vec_path"],
                    }

        # Log which run won
        if best_run:
            mlflow.set_tag("best_run_id", best_run["run_id"])
            mlflow.log_metric("best_score", best_score)
            mlflow.set_tag("best_model_name", best_run["config"]["name"])

            # Copy best model to models/best_model.pkl (DVC will track this)
            os.makedirs("models", exist_ok=True)
            with open(best_run["model_path"], "rb") as src, open(
                "models/best_model.pkl", "wb"
            ) as dst:
                dst.write(src.read())
            with open(best_run["vec_path"], "rb") as src, open(
                "models/best_vectorizer.pkl", "wb"
            ) as dst:
                dst.write(src.read())

            mlflow.log_artifact("models/best_model.pkl")
            mlflow.log_artifact("models/best_vectorizer.pkl")


if __name__ == "__main__":
    main()
