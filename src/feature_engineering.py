import logging
import os
import pickle
import yaml

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# logging configuration
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building_errors.log')
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
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        X_train_tfidf = vectorizer.fit_transform(X_train) # TF-IDF transformation
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['category'] = y_train
        logger.debug(f"TF-IDF applied with bigrams training data transformed. Train shape: {train_df.shape}")
        return train_df, vectorizer
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        max_features = 1000
        ngram_range = (1, 2)

        # Load the preprocessed training data
        train_data = load_data('./data/processed/train_processed.csv')

        # Apply TF-IDF feature engineering on training data
        train_tfidf, vectorizer = apply_tfidf(train_data, max_features, ngram_range)

        save_data(train_tfidf, os.path.join("./data", "feature-engineered", "train_tfidf.csv"))
        
        model_save_path = 'models/tfidf_vectorizer.pkl'
        save_model(vectorizer, model_save_path)
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
