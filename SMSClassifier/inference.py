# inference.py

import pickle
import pandas as pd

def Load_model(path='SMSClassifier/sms_classifier.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_sms(messages, model=None):
    if model is None:
        model = Load_model()
    
    if isinstance(messages, str):
        messages = [messages]
    
    predictions = model.predict(messages)
    probabilities = model.predict_proba(messages)

    results = pd.DataFrame({
        'Message': messages,
        'Prediction': predictions,
        'Spam_Probability (%)': (probabilities[:, 1] * 100).round(2)
    })
    
    return results

if __name__ == "__main__":
    model = Load_model()
    test_messages = [
        "Congratulations! You have won a $1000 Walmart gift card. Go to http://bit.ly/123456",
        "Hi Arbaz, are we still on for the meeting today?",
        "You’ve been selected for a FREE loan worth ₹5 Lakh! Apply now."
    ]
    
    output = predict_sms(test_messages, model)
    print(output)
