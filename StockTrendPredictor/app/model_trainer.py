# Import libraries
import pandas as pd
from dotenv import load_dotenv
import os
from io import BytesIO
import joblib
import psycopg2
from dotenv import load_dotenv
import gdown
import logging
import tempfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from db_manager import connect_db, update_model_meta, create_table_name, get_entire_processed_data
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from inference import run_inference
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Train model 
def train_data(df):

    # Remove last 7 days data
    df = df.iloc[:-8, :].copy()
    # Features (X)
    features = ['sma_14', 'ema_14', 'rsi_14', 'macd', 'close_lag1', 'close_lag2', 'close_lag3', 'close_lag4', 'close_lag5', 'close_lag6', 'close_lag7']
    X = df[features]

    # Multioutput target columns (Y)
    target_cols = [f'target_day_{i}' for i in range(1,8)]
    Y = df[target_cols]

    # Train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train multioutput regressor
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_model = MultiOutputRegressor(base_regressor)

    multi_model.fit(X_train, Y_train)
    logging.info("Multi Output Regression model trained")

    # Evaluate
    Y_pred = multi_model.predict(X_test)

    # Calculate avg MSE and R2 across all 7 output
    mse_total, r2_total = 0, 0
    for i, col in enumerate(target_cols):
        mse = mean_squared_error(Y_test.iloc[:,i], Y_pred[:,i])
        r2 = r2_score(Y_test.iloc[:,i], Y_pred[:,i])
        mse_total += mse
        r2_total += r2
        logging.info(f"Day {i + 1} - MSE: {mse:.2f}, R2: {r2:.2f}")

    avg_mse = round(mse_total / 7, 2)
    avg_r2 = round(r2_total / 7, 2)

    logging.info(f"Average MSE: {avg_mse}")
    logging.info(f"Average R2: {avg_r2}")

    return multi_model, avg_mse, avg_r2

# Upload model
def upload_model_object_to_drive(symbol, model_object, upload_flag=True):

    if not upload_flag:
        logging.info("Skipping model upload (upload_flag is False)")
        return None

    
    # load_dotenv()
    # drive_folder_id = os.getenv('DRIVE_FOLDER_ID')
    # initial_creds = os.getenv("GDRIVE_CREDENTIALS_INITIAL")
    # final_creds = os.getenv("GDRIVE_CREDENTIALS_FINAL")

    drive_folder_id = st.secrets('DRIVE_FOLDER_ID')
    initial_creds = st.secrets("GDRIVE_CREDENTIALS_INITIAL")
    final_creds = st.secrets("GDRIVE_CREDENTIALS_FINAL")

    if initial_creds:
        with open("gdrive_credentials_initial.json", "w") as f:
            f.write(initial_creds)

    if final_creds:
        with open("gdrive_credentials_final.json", "w") as f:
            f.write(final_creds)

    # Create temp file and dump model directly
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        temp_path = tmp_file.name
        joblib.dump(model_object, temp_path)

    # Authenticate Google Drive
    gauth = GoogleAuth()
    # Try to load saved credentials
    if os.path.exists("gdrive_credentials_final.json"):
        gauth.LoadCredentialsFile("gdrive_credentials_final.json")
        if gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
    else:
        # First time or no saved token → browser login
        gauth.LoadClientConfigFile("gdrive_credentials_initial.json")
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline', 'prompt': 'consent'})
        # Run auth
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile("gdrive_credentials_final.json")
        
    drive = GoogleDrive(gauth)

    # Check if file already exists
    file_name = f"{create_table_name(symbol)}_model.pkl"
    query = f"'{drive_folder_id}' in parents and title = '{file_name}' and trashed = false"
    file_list = drive.ListFile({'q': query}).GetList()

    if file_list:
        # Update existing file
        file_drive = file_list[0]
        logging.info(f"Found existing file, updating: {file_name}")
    else:
        # Create new file
        file_drive = drive.CreateFile({'title': file_name, 'parents': [{'id': drive_folder_id}]})
        logging.info(f"Creating new file: {file_name}")

    file_drive.SetContentFile(temp_path)
    file_drive.Upload()
    logging.info(f"Uploaded {file_name} to Google Drive")

    # Make public
    file_drive.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
    model_url = f"https://drive.google.com/file/d/{file_drive['id']}/view?usp=sharing"
    logging.info(f"Shareable Link: {model_url}")

    try:
        os.remove(temp_path)
        logging.info(f"Deleted temp file: {temp_path}")
    except Exception as e:
        logging.warning(f"Could not delete temp file: {e}")

    return model_url

if __name__ == '__main__':

    symbol = 'TSLA'

    conn = connect_db()
    if conn is not None:
        df = get_entire_processed_data(symbol, conn)
        multi_model, avg_mse, avg_r2 = train_data(df)
        
        model_url = upload_model_object_to_drive(symbol, multi_model)
        update_model_meta(conn, symbol, avg_mse, avg_r2, model_url)
