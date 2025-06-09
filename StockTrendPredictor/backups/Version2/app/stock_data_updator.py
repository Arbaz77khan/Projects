# Import libraries
from datetime import datetime
from db_manager import connect_db, update_model_meta
from dotenv import load_dotenv
from io import BytesIO
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from data_processor import collect_data, engineer_features, create_targets
from db_manager import (connect_db, create_core_table, check_stock_in_list, 
    add_stock_to_list, fetch_symbols, create_table_name, create_processed_table, 
    insert_processed_data, get_latest_date, delete_old_data, update_table, 
    update_inference_result, fetch_inference_result, update_model_meta)
from model_trainer import train_data, upload_model_object_to_drive
from inference import generate_trend_action, run_inference
import gdown
import joblib
import logging
import numpy as np
import os
import pandas as pd
import psycopg2
import random
import re
import ta
import time
import tempfile
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_pipeline(conn, symbol, daily_update_flag = False):
    try:
        
        logging.info("DP: at level 1: collect_data")
        df = collect_data(symbol)

        if df is None:
            logging.error("Aborting pipeline.")
            return

        max_table_size = 365
        logging.info("DP: at level 2: create_table_name")
        table_name = create_table_name(symbol)

        logging.info("DP: at level 3: create_core_table")
        create_core_table(conn)

        logging.info("DP: at level 4: check_stock_in_list")
        if check_stock_in_list(conn, symbol):
            logging.info("DU: at level 4: get_latest_date")
            latest_db_date = get_latest_date(conn, table_name)
            latest_db_date = pd.to_datetime(latest_db_date)

            latest_df_date = df['Date'].max()
            latest_df_date = pd.to_datetime(latest_df_date)


            if latest_db_date is not None and latest_df_date <= latest_db_date:
                logging.info(f"No new data for {symbol}; skipping daily updates")
                return

        logging.info("DP: at level 5: engineer_features")
        df = engineer_features(df)

        logging.info("DP: at level 6: create_targets")
        df = create_targets(df) 

        logging.info("DP: at level 7: train_data")
        multi_model, avg_mse, avg_r2 = train_data(df)

        if daily_update_flag:
            logging.info("DU: at level 5: upload_model_object_to_drive")
            model_url = upload_model_object_to_drive(symbol, multi_model)
        else:
            model_url = 'PENDING'

        logging.info("DP: at level 8: check_stock_in_list")
        if not check_stock_in_list(conn, symbol):

            logging.info("DP: at level 9: add_stock_to_list")
            add_stock_to_list(conn, symbol)

            logging.info("DP: at level 10: create_processed_table")
            create_processed_table(conn, table_name)

        logging.info("DP: at level 11: update_table")
        update_table(conn, table_name, df, max_table_size)

        logging.info("DP: at level 12: update_model_meta")
        update_model_meta(conn, symbol, avg_mse, avg_r2, model_url)
        
        logging.info("DP: at level 13: run_inference")
        run_inference(conn, symbol, multi_model)
        
        logging.info(f"Pipeline completed successfully for {symbol}")

    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):  
                logging.info(f"Too Many Requests - raising exception to outer block")
                raise
        logging.error(f"Pipeline error for {symbol}: {str(e)}")

def daily_update(conn):
    daily_update_flag = True

    try:
        logging.info("DU: at level 2: fetch_symbols")
        symbol_list = fetch_symbols(conn)

        daily_update_flag = True

        for symbol in symbol_list:
            try:
                logging.info("DU: at level 3: data_pipeline")
                data_pipeline(conn, symbol, daily_update_flag)
            except Exception as e:
                logging.error(f"Error running pipeline for {symbol}: {str(e)}")
        logging.info("Daily update completed — all stocks checked")
    except Exception as e:
        logging.error(f"Daily update error: {str(e)}")


if __name__ == '__main__':

    streamlit_app = False
    daily_update_trigger = True

    logging.info("Main: at level 1: connect_db")
    conn = connect_db()

    if streamlit_app:
        symbol = 'BEL.NS'

        if conn is not None:

            logging.info("Main: at level 2: data_pipeline")
            data_pipeline(conn, symbol)

            logging.info("Main: at level 3: fetch_inference_result")
            inference, trend = fetch_inference_result(conn, symbol)

            conn.close()
        else:
            logging.error("DB coudn't connect. Refresh it!")

    if daily_update_trigger:
        daily_update(conn)
        

    
