import pandas as pd
import argparse
from urllib.request import urlretrieve
import os
import logging

def download_data(url: str, output_path: str) -> str:
    """Загружает данные из URL и сохраняет локально"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urlretrieve(url, output_path)
    return output_path

def load_data(file_path: str) -> pd.DataFrame:
    """Загружает данные из CSV файла"""
    return pd.read_csv(file_path)

def validate_data(df: pd.DataFrame) -> bool:
    """Проверяет структуру данных"""
    required_columns = ['id', 'diagnosis'] + [f'{stat}_{feature}' 
                    for feature in ['radius', 'texture', 'perimeter', 'area', 
                                  'smoothness', 'compactness', 'concavity',
                                  'concave_points', 'symmetry', 'fractal_dimension']
                    for stat in ['mean', 'se', 'worst']]
    
    return all(col in df.columns for col in required_columns)

def main(params):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Параметры
    data_url = params.url
    raw_data_path = params.output
    
    try:
        logger.info("Downloading data from %s", data_url)
        file_path = download_data(data_url, raw_data_path)
        
        logger.info("Loading data from %s", file_path)
        df = load_data(file_path)
        
        logger.info("Validating data structure")
        if not validate_data(df):
            raise ValueError("Invalid data structure")
            
        logger.info("Data extracted successfully. Shape: %s", df.shape)
        return df
    
    except Exception as e:
        logger.error("Error in data extraction: %s", str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True,
                      help='URL to download the dataset')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the raw data')
    
    args = parser.parse_args()
    main(args)