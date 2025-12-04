# import & setup
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

# model_path = 'D:/Master_Folder/Data Science Course/Projects/churn_prediction_project/models/logistic_model.joblib'

# --- sklearn compatibility shim for older pickled models ---
try:
    from sklearn.compose import _column_transformer

    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass
# --- end shim ---

# load model
def load_pipeline():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "logistic_model.joblib"
    return joblib.load(model_path)

def feature_engineer(df):
    df = df.drop(columns=['customerID'])
    df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)

    df['services_count'] = (
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int)
    )

    df['recent_drop'] = (df['tenure'] <=3).astype(int)

    return df

# predict 
def predict_proba(input_data, pipeline):
    X = pd.DataFrame([input_data])
    X = feature_engineer(X)

    proba = pipeline.predict_proba(X)[:,1]
    
    return float(proba[0]) if len(proba) == 1 else proba

# churn or not churn
def predict_label(input_data, pipeline, threshold=0.5):
    p = predict_proba(input_data, pipeline)
    if isinstance(p, (list, np.ndarray)):
        return (np.array(p) >= threshold).astype(int).tolist()
    return int(p >= threshold)

if __name__ == "__main__":
    X = {
        "customerID":"0001-ABCD",
        "gender":"Female",
        "SeniorCitizen":0,
        "Partner":"No",
        "Dependents":"No",
        "tenure":2,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"Fiber optic",
        "OnlineSecurity":"No",
        "OnlineBackup":"No",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"No",
        "StreamingMovies":"No",
        "Contract":"Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "MonthlyCharges":89.10,
        "TotalCharges":178.2,
    }

    pipe = load_pipeline()
    prob = predict_proba(X, pipeline=pipe)
    label = predict_label(X, threshold=0.5, pipeline=pipe)
    print(f"Churn probability: {prob:.4f}  Label: {label}")