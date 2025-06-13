# Import libraries
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import logging
from io import BytesIO
import requests
import json
import tempfile
from db_manager import connect_db, get_latest_feature_row, get_model_url, create_table_name, update_inference_result

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# download model
def download_model(conn, symbol):
    try:
        model_url = get_model_url(conn, symbol)

        file_id = model_url.split('/d/')[1].split('/')[0]
        gdown_url = f"https://drive.google.com/uc?id={file_id}"

        # Create a safe temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name

        # Download using gdown (handles confirmation tokens)
        gdown.download(gdown_url, temp_path, quiet=False)

        # Load model from the downloaded file
        model = joblib.load(temp_path)
        logging.info(f"Model for {symbol} downloaded and loaded successfully")

        # Delete the local file after loading
        os.remove(temp_path)

        return model

    except Exception as e:
        logging.error(f"Error downloading/loading model: {e}")
        return None

# generate trend
def generate_trend_action(predicted_prices, current_close, threshold=0.02):
    x = np.arange(len(predicted_prices))  # Days [0, 1, 2, ..., 6]
    y = np.array(predicted_prices)  # Predicted prices

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator

    change = (predicted_prices[-1] - current_close) / current_close

    if slope > 0 and change > threshold:
        return "BUY"
    elif slope < 0 and change < -threshold:
        return "SELL"
    else:
        return "HOLD"

# run inference
def run_inference(conn, symbol, model):
    try:
        if model is not None:
            logging.info("RI: at level 1: get_latest_feature_row")
            X_latest = get_latest_feature_row(conn, symbol)

            predicted = model.predict(X_latest)[0]
            predicted_rounded = [round(p, 2) for p in predicted]
            price_predictions = pd.DataFrame({
                'Day': [f'Day {i+1}' for i in range(7)],
                'Predicted Price': predicted_rounded
            })

            current_close = X_latest['close_lag1'].values[0]

            logging.info("RI: at level 2: generate_trend_action")
            trend_action = generate_trend_action(predicted, current_close)

            logging.info("RI: at level 3: update_inference_result")
            inference_dict = price_predictions.set_index('Day')['Predicted Price'].to_dict()
            update_inference_result(conn, symbol, inference_dict, trend_action)

            logging.info(f"Inference completed for {symbol}")

    except Exception as e:
        logging.error(f"Inference error: {str(e)}")

if __name__ == '__main__':
    conn = connect_db()
    symbol = 'TSLA'
    model = download_model(conn, symbol)

    run_inference(conn, symbol, model)

    print(price_predictions)
    print(trend_action)
