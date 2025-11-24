import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# logging configuration
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug(f'Data loaded from {file_path} and with shape {df.shape}')
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f'Model {model} loaded from {model_path}')
        return model
    except Exception as e:
        logger.error(f'Error loading model from {model_path}: {e}')
        raise

def main():
    try:
        model = load_model('./models/lgbm_model.pkl')
        vectorizer = load_model('./models/tfidf_vectorizer.pkl')
        
        test_data = load_data('data/processed/test_processed.csv')

        # Prepare test data
        X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
        X_test = pd.DataFrame(X_test_tfidf.toarray())
        y_test = test_data['category'].values

        with mlflow.start_run(run_name="model_evaluation"):
            y_pred = model.predict(X_test)

            # metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric}", value)
            
            # confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            cm_path = "reports/confusion_matrix.png"
            os.makedirs("reports", exist_ok=True)
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)

            # Save metrics.json
            metrics_path = "reports/metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(report, f, indent=4)
            mlflow.log_artifact(metrics_path)

            # Log the model (again for evaluation)
            mlflow.sklearn.log_model(model, artifact_path="evaluated_model")

            logger.info("Evaluation logged successfully")
    except Exception as e:
        logger.error(f'Failed to complete the model evaluation process: {e}')
        raise


if __name__ == '__main__':
    main()