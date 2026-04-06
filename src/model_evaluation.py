import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, accuracy_score,precision_score,recall_score
import logging
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Create a file handler to write logs to a file
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Create a formatter and set it for the handlers

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path: str) -> dict:
    """ load parameters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrived successfully from: %s', [params_path])
        return params
    except FileNotFoundError:
        logger.error('file not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the parameters; %s', e)
        raise


def load_model(file_path: str):
    """load the modelfrom a pickle file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from: %s' , file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found:%s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """ Load data from Csv file """ 
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError:
        logger.error('Fail to read file %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occure %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model and return the evaluation metrics.
    """
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test,y_pred)
        recall= recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)

        metrics_dict ={
            'accuracy' :accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }

        logger.debug('Model evaluate metrics calcualted')
        return metrics_dict
    except Exception as e:
        logging.error('Unexpected error occure: %s', e)
        raise


def save_metrics(metrics : dict, file_path: str) -> None:
    """
    Save the evaluation metrics to json file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent = 4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occured: %s', e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')

        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logging.error('Failed to complete model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__=='__main__':
    main()


    

    
