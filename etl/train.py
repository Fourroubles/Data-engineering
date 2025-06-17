import argparse
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import json
import os

def train_model(X_train, y_train, param_grid=None, cv=5):
    """Обучение модели логистической регрессии"""
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    
    model = LogisticRegression(max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def main(params):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка данных
        logger.info("Loading training data from %s", params.input)
        data = joblib.load(params.input)
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Обучение модели
        logger.info("Training Logistic Regression model")
        model, best_params = train_model(X_train, y_train)
        
        # Сохранение модели
        logger.info("Saving model to %s", params.model_output)
        os.makedirs(os.path.dirname(params.model_output), exist_ok=True)
        joblib.dump(model, params.model_output)
        
        # Сохранение параметров
        if params.params_output:
            logger.info("Saving best parameters to %s", params.params_output)
            with open(params.params_output, 'w') as f:
                json.dump(best_params, f)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error("Error in model training: %s", str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                      help='Path to processed training data')
    parser.add_argument('--model_output', type=str, required=True,
                      help='Path to save trained model')
    parser.add_argument('--params_output', type=str,
                      help='Path to save best parameters')
    
    args = parser.parse_args()
    main(args)