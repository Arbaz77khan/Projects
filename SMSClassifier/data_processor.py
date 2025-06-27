# data_processor.py

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_dataset(path='spam.csv'):
    df = pd.read_csv(path, encoding='latin-1')
    return df

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove mentions, hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove emojis and non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Rejoin
    cleaned = ' '.join(tokens)

    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def prepare_data(path='spam.csv'):
    df = load_dataset(path)
    df['message_cleaned'] = df['message'].apply(clean_text)
    return df

if __name__ == "__main__":
    df_cleaned = prepare_data()
    print(df_cleaned.head())
