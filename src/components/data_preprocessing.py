import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

from src.logger import logging
from src.utils.main_utils import load_data, save_data


# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower() # to lowercase
        comment = comment.strip() # trailing and leading whitespaces
        comment = re.sub(r'\n', ' ', comment) # remove newline characters
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment) # remove non-alphanumeric characters, except punctuation

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words]) # remove unimportant stopwords
        
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()]) # lemmatize
        
        return comment
    except Exception as e:
        logging.error(f"Error in preprocessing comment: {e}")
        return comment
    
def normalize_text(df: pd.DataFrame):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logging.debug('Text normalization completed')
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def main():
    try:
        logging.debug("Starting data preprocessing...")
        
        # Fetch the raw data
        train_data = load_data('artifacts/data_ingestion/ingested/train.csv')
        test_data = load_data('artifacts/data_ingestion/ingested/test.csv')
        logging.debug('Raw train and test data loaded successfully')

        # Preprocessing
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save
        save_data(train_processed_data, 'artifacts/data_preprocessing/train_processed.csv')
        save_data(test_processed_data, 'artifacts/data_preprocessing/test_processed.csv')

    except Exception as e:
        logging.error(f'Failed to complete the data preprocessing process: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
