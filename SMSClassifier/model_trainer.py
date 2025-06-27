# model_trainer.py

import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_and_save_model(df, output_path='sms_classifier.pkl'):
    X = df['message_cleaned']
    y = df['label']

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline: TF-IDF + Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ('model', LogisticRegression(solver='liblinear'))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"\nModel + vectorizer saved to: {output_path}")

if __name__ == "__main__":
    from data_processor import prepare_data
    df = prepare_data()
    train_and_save_model(df)
