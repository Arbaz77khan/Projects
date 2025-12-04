# import and setup
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score

train_path = 'D:/Master_Folder/Data Science Course/Projects/churn_prediction_project/data/processed/train.csv'
test_path = 'D:/Master_Folder/Data Science Course/Projects/churn_prediction_project/data/processed/test.csv'
model_dir = 'D:/Master_Folder/Data Science Course/Projects/churn_prediction_project/models'

# load data
def load_processed_data():

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test

# Column transformation and preprocessing
def build_preprocessor(df):

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # remove target if present
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    return preprocessor, numeric_cols, categorical_cols

    
# train model
def train_models(X_train, y_train, preprocessor):

    pipe_lr = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    pipe_lr.fit(X_train, y_train)

    pipe_rf = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)

    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    pipe_xgb = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', XGBClassifier(
            eval_metric='logloss',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            random_state=42
        ))
    ])
    pipe_xgb.fit(X_train, y_train)

    return pipe_lr, pipe_rf, pipe_xgb

# evaluate model
def evaluate(m, X_test, y_test):
    preds = m.predict(X_test)
    proba = m.predict_proba(X_test)[:,1]

    print(classification_report(y_test, preds))
    print('ROC-AUC:', roc_auc_score(y_test, proba))

if __name__ == '__main__':
    train, test = load_processed_data()

    X_train = train.drop(columns=['Churn']) 
    y_train = train['Churn']
    X_test = test.drop(columns=['Churn']) 
    y_test = test['Churn']

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(1, n_pos)
    print(f"n_pos={n_pos}, n_neg={n_neg}, scale_pos_weight={scale_pos_weight:.2f}")


    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train)

    lr, rf, xgb = train_models(X_train, y_train, preprocessor)
    evaluate(lr, X_test, y_test)
    evaluate(rf, X_test, y_test)
    evaluate(xgb, X_test, y_test)

    joblib.dump(lr, f"{model_dir}/logistic_model.joblib")
    joblib.dump(rf, f"{model_dir}/random_forest_model.joblib")
    joblib.dump(xgb, f"{model_dir}/xgb_model.joblib")