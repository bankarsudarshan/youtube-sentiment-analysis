import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import logging
from src.utils.main_utils import read_yaml_file, load_data, save_data, save_object


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        vectorizer.fit(X_train)
        return vectorizer
    
    except Exception as e:
        logging.error('Error during TF-IDF transformation: %s', e)
        raise e


def main():
    try:

        params = read_yaml_file('params.yaml')
        max_features = params['data_transformation']['max_features']
        ngram_range = tuple(params['data_transformation']['ngram_range'])

        # Load the preprocessed training data
        train_data = load_data('artifacts/data_preprocessing/train_processed.csv')

        # Apply TF-IDF feature engineering on training data
        vectorizer = apply_tfidf(train_data, max_features, ngram_range)

        # Save the transformed data and vectorizer\
        save_object('artifacts/data_transformation/tfidf_vectorizer.pkl', vectorizer)
        logging.debug('Transformation object saved in artifacts/model_trainer')

    except Exception as e:
        logging.error(f'Failed to complete the feature engineering process: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()