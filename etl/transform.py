import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Основная функция предобработки данных"""
    # Кодирование целевой переменной
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Удаление ненужных колонок
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Разделение данных на train/test"""
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test, scaler_path: str = None):
    """Масштабирование признаков"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_test_scaled, scaler

def main(params):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка данных
        logger.info("Loading data from %s", params.input)
        df = pd.read_csv(params.input)
        
        # Предобработка
        logger.info("Preprocessing data")
        df = preprocess_data(df)
        
        # Разделение на train/test
        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = split_data(df, params.test_size, params.random_state)
        
        # Масштабирование
        logger.info("Scaling features")
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_test, params.scaler_output
        )
        
        # Сохранение данных
        logger.info("Saving processed data")
        os.makedirs(os.path.dirname(params.train_output), exist_ok=True)
        
        joblib.dump({
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled,
            'y_test': y_test
        }, params.train_output)
        
        logger.info("Data transformation completed successfully")
        
    except Exception as e:
        logger.error("Error in data transformation: %s", str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                      help='Path to raw data CSV file')
    parser.add_argument('--train_output', type=str, required=True,
                      help='Path to save processed train data')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random state for splitting (default: 42)')
    parser.add_argument('--scaler_output', type=str,
                      help='Path to save scaler object')
    
    args = parser.parse_args()
    main(args)