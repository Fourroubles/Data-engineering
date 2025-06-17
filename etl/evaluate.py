import argparse
import joblib
import json
import logging
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score,
                           confusion_matrix)
import pandas as pd
import os

def evaluate_model(model, X_test, y_test):
    """Вычисление метрик качества модели"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=['Actual Negative', 'Actual Positive'],
                        columns=['Predicted Negative', 'Predicted Positive'])
    
    return metrics, cm_df

def main(params):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка данных и модели
        logger.info("Loading test data from %s", params.input)
        data = joblib.load(params.input)
        X_test = data['X_test']
        y_test = data['y_test']
        
        logger.info("Loading model from %s", params.model_input)
        model = joblib.load(params.model_input)
        
        # Оценка модели
        logger.info("Evaluating model")
        metrics, cm = evaluate_model(model, X_test, y_test)
        
        # Сохранение метрик
        logger.info("Saving metrics to %s", params.metrics_output)
        os.makedirs(os.path.dirname(params.metrics_output), exist_ok=True)
        with open(params.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Сохранение матрицы ошибок
        if params.cm_output:
            logger.info("Saving confusion matrix to %s", params.cm_output)
            cm.to_csv(params.cm_output)
        
        logger.info("Model evaluation completed successfully")
        logger.info("Metrics: %s", json.dumps(metrics, indent=2))
        
    except Exception as e:
        logger.error("Error in model evaluation: %s", str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                      help='Path to processed test data')
    parser.add_argument('--model_input', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--metrics_output', type=str, required=True,
                      help='Path to save metrics JSON')
    parser.add_argument('--cm_output', type=str,
                      help='Path to save confusion matrix CSV')
    
    args = parser.parse_args()
    main(args)