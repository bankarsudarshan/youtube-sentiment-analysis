import lightgbm as lgb
import mlflow

from src.logger import logging
from src.utils.main_utils import read_yaml_file, load_data, load_object, save_dict_as_json


def train_model():
    try:
        params = read_yaml_file("params.yaml")
        max_features = params["data_transformation"]["max_features"]
        ngram_range = params["data_transformation"]["ngram_range"]
        learning_rate = params["model_trainer"]["learning_rate"]
        max_depth = params["model_trainer"]["max_depth"]
        n_estimators = params["model_trainer"]["n_estimators"]

        train_data = load_data("artifacts/data_preprocessing/train_processed.csv")

        # load pre-processed data and vectorizer object
        X_train, y_train = train_data["clean_comment"], train_data["category"]
        vectorizer = load_object("artifacts/data_transformation/tfidf_vectorizer.pkl")

        # text data transform
        X_train_transformed = vectorizer.transform(X_train).tocsr()

        # model
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )

        # ------------------------Training and MLflow auto-logging------------------------
        mlflow.set_tracking_uri("http://ec2-51-20-74-217.eu-north-1.compute.amazonaws.com:5000")
        mlflow.set_experiment("Training Pipeline")
        logging.debug("MLflow tracking URI and experiment is set")

        mlflow.lightgbm.autolog()

        with mlflow.start_run() as run:

            # Log TF-IDF params manually
            mlflow.log_param("vectorizer_max_features", max_features)
            mlflow.log_param("vectorizer_ngram_range", ngram_range)
            mlflow.log_param("vocab_size", len(vectorizer.vocabulary_))

            model.fit(X_train_transformed, y_train)
            logging.info("LightGBM model training completed")

            save_dict_as_json({"run_id": run.info.run_id}, 'artifacts/model_trainer/output.json')

            logging.info(f"MLflow run completed: {run.info.run_id}")
        # ------------------------Training and MLflow auto-logging------------------------

    except Exception as e:
        logging.error(f"Failed to complete model training: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    train_model()
