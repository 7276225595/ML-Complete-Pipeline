import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')


#Ensure Log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Settigup logger

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transform the input text by converting to lowercase, tokenizing, remove stopwords and punctuation and steming.
    """

    ps = PorterStemmer()
    #Convert text to lowercase
    text = text.lower()
    #Tokeize the words
    text = nltk.word_tokenize(text)
    #Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    #Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #Stem the words
    text = [ps.stem(word) for word in text]
    #Join the words back to a single string
    return ' '.join(text)


def preprocess_data(df, text_column='text', target_column='target'):
    """
    preprocessing the data by encoding the targeted column ,removing duplicates and tranforming the text column.
    """
    try:
        logger.debug('Sart preprocessing for DataFrame')
        #Encoding the targeted column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded successfully')

        #Remove duplicates rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicate Removed')

        #Apply text transformation to specific text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text Column transformed')
        return df
    
    except KeyError as e:
        logger.error('column not found : %s', e)
        raise

    except Exception as e:
        logger.error('Error during text normalization')
        raise

def main(text_column='text', target_column='target'):
    """
    Main fucntion to load the data preprocess it and save the preprocessed data
    """
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        #Transform the data
        train_processed_data = preprocess_data(train_data, text_column, target_column)
        test_processed_data = preprocess_data(train_data, text_column, target_column)

        # Store the data inside data/processsed directory

        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'),index = True)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index= True)

        logger.debug('Processed data saved : %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found:%s', e)
    except pd.errors.ParserError as e:
        logger.error('No data: %s',e)
    except Exception as e:
        logger.error('Failed to complete data transformation: %s', e)
        print(f"Error:{e}")

if __name__ == '__main__':
    main()

