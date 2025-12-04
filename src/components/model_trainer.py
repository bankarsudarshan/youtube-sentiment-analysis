import lightgbm as lgb
import numpy as np

from src.logger import logging
from src.utils.main_utils import read_yaml_file, load_data, load_object, save_object


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
        logging.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logging.error('Error during LightGBM model training: %s', e)
        raise

def main():
    try:

        params = read_yaml_file('params.yaml')

        learning_rate = params['model_trainer']['learning_rate']
        max_depth = params['model_trainer']['max_depth']
        n_estimators = params['model_trainer']['n_estimators']

        # Load the processed training data
        train_data = load_data('artifacts/data_preprocessing/train_processed.csv')
        X_train, y_train = train_data['clean_comment'], train_data['category']

        vectorizer = load_object("artifacts/data_transformation/tfidf_vectorizer.pkl")

        # transform (feature engineer) the processed training data
        X_train_transformed = vectorizer.transform(X_train)

        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_lgbm(X_train_transformed, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory
        save_object('artifacts/model_trainer/lgbm_model.pkl', best_model)

    except Exception as e:
        logging.error(f'Failed to complete the feature engineering and model building process: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()