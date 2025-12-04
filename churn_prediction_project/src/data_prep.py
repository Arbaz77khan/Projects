# import & setups
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'D:/Master_Folder/Data Science Course/Projects/churn_prediction_project/data/raw/telco_customer_churn_dataset.csv'
processed_dir = 'D:/Master_Folder/Data Science Course/Projects/churn_prediction_project/data/processed'


# load data
def load_data(path=file_path):
    return pd.read_csv(path)

# clean data
def clean_data(df):
    # change TotalCharges type to int and fill NAN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # map churn column to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

# derive features
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

# split data into X,y & train,test
def split_data(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41, stratify=y
    )
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return(train, test)

# save processed data
def save_data(train, test, processed_dir=processed_dir):
    train.to_csv(f'{processed_dir}/train.csv', index=False)
    test.to_csv(f'{processed_dir}/test.csv', index=False)


if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    df = feature_engineer(df)
    train, test = split_data(df)
    save_data(train, test)



