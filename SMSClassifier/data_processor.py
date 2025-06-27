# data_processor.py

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def load_dataset(path='spam.csv'):
    df = pd.read_csv(path, encoding='latin-1')
    # df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def prepare_data(path='spam.csv'):
    df = load_dataset(path)
    df['message_cleaned'] = df['message'].apply(clean_text)
    return df

if __name__ == "__main__":
    df_cleaned = prepare_data()
    print(df_cleaned.head())
