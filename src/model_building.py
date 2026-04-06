import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Create a formatter and set it for the handlers

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters retrived successfully from: %s', params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('Parameters file not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('Error parsing YAML file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error while loading the parameters: %s', e)
#         raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to file csv file: %s', e)
        raise
    except FileNotFoundError:
        logger.error('Data file not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the random fprest model.
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('The number of samples in X_train and y_train must be the same')
            
        logger.debug('Initializing thr Random forest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training complete')

        return clf
    except ValueError as e:
        logger.error('Value error during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """ Save the trainded model to a file using pickle

    :param model: The trained model to be saved
    :param model_path: The file path where the model will be saved
    """
    try:
        #Ensure the directory for the model path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved successfully to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error while saving the model: %s', e)
        raise


def main():
    try:
        params = {'n_estimators' : 25, 'random_state' : 2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1]. values

        clf = train_model(x_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

