import logging
import os
import pickle
import time

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

# logging configuration
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    logger.debug(f"Training data loaded with shape {df.shape}")
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.debug(f"Saved featurized data to {path}")

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug('Parameters retrieved from %s', params_path)
    return params

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(
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
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.debug(f"Model saved at {path}")

def main():
    try:
        params = load_params('params.yaml')
        # TF-IDF params
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        # lightgbm parameters
        lr = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load the preprocessed training data
        train_data = load_data('./data/processed/train_processed.csv')

        with mlflow.start_run(run_name='model_training'):
            # Log TF-IDF params
            mlflow.log_param("tfidf_max_features", max_features)
            mlflow.log_param("tfidf_ngram_range", ngram_range)

            # Build and fit TF-IDF
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            X_train_tfidf = vectorizer.fit_transform(train_data["clean_comment"])
            y_train = train_data["category"].values

            # Log vectorizer artifact
            vectorizer_path = "models/tfidf_vectorizer.pkl"
            save_model(vectorizer, vectorizer_path)
            mlflow.log_artifact(vectorizer_path)

            # Log model hyperparameters
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)
            
            # Train model
            start = time.time()
            model = train_lgbm(X_train_tfidf, y_train, lr, max_depth, n_estimators)
            end = time.time()
            mlflow.log_metric("training_time_sec", end - start)

            # Save LightGBM model locally
            model_path = "models/lgbm_model.pkl"
            save_model(model, model_path)
            mlflow.log_artifact(model_path)

            # Log model to MLflow in a deployable format
            mlflow.lightgbm.log_model(model, artifact_path="lgbm_model")

            logger.info("Model training completed and logged to MLflow")

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
