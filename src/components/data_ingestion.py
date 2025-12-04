import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging as logger
from src.utils.main_utils import load_data, read_yaml_file, save_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        
        df.dropna(inplace=True) # remove missing values
        df.drop_duplicates(inplace=True) # remove duplicates
        df = df[df['clean_comment'].str.strip() != ''] # remove rows with empty strings
        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def main():
    try:
        params = read_yaml_file('params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_url = 'https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        df = load_data(path=data_url)
        save_data(df, 'artifacts/data_ingestion/feature_store/raw_original.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, 'artifacts/data_ingestion/ingested/train.csv')
        save_data(df, 'artifacts/data_ingestion/ingested/test.csv')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
