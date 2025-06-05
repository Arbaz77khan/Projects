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
from stock_data_updator import data_pipeline, daily_update
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
import streamlit as st
import tempfile
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(page_title='ProInvest', layout='centered')

# Title Section
st.title("ProInvest — Invest Like a Pro!")
st.caption("Your stock assistant for optimizing investment strategies.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Params
mode = st.experimental_get_query_params().get("mode", ["app"])[0]

# Initialize Session State 
if 'symbol' not in st.session_state:
    st.session_state['symbol'] = ''
if 'retry_count' not in st.session_state:
    st.session_state['retry_count'] = 0

conn = connect_db()
if conn is None:
    st.error("Ah!!! Seems database is playing hide n seek! Try refreshing...")
    st.stop()

# Main Mode
if mode == "app":
    st.markdown("### Search for your stock")
    user_input = st.text_input("Drop a stock symbol (e.g., TSLA, INFY.NS)", value=st.session_state['symbol'])

    if st.button("Smash it!"):
        st.session_state['symbol'] = user_input.upper().strip()
        st.session_state['retry_count'] = 0  # reset retry count
        
    symbol = st.session_state['symbol']

    if symbol:
        st.spinner(f"Looking up `{symbol}`...")

        if not check_stock_in_list(conn, symbol):
            st.session_state['retry_count'] += 1

            if st.session_state['retry_count'] > 10:
                st.error("Seems market is busy digging gold. Please try later!")
                st.stop()

            with st.spinner(f"Checking the vaults for {symbol}..."):
                try:
                    data_pipeline(conn, symbol)
                except Exception as e:
                    error_message = str(e)

                    if "429" in error_message or "Too Many Requests" in error_message:
                        wait_time = random.uniform(2, 4)
                        time.sleep(wait_time)
                        logging.info(f"Waiting {wait_time:.2f} seconds before retrying...")
                        st.rerun()
                    else:
                        logging.info(e)
                        st.error("Server down! Please try later!")
                        st.stop()

        inference, trend = fetch_inference_result(conn, symbol)
        if inference:
            st.success("Here's your fresh stock prediction:")
            st.dataframe(inference, use_container_width=True)
            st.markdown(f"### Move Suggestion: **{trend}**")
        else:
            st.error("🚨 Weird! Data exists, but no predictions found.")

        conn.close()

# Scheduler Trigger (optional)
elif mode == "update":
    st.markdown("## Daily Update Trigger (Admin Mode)")
    st.info("Initiating backend wizardry... might take a moment")
    try:
        daily_update()
        st.success("Stocks updated! We’re fresh as morning chai")
    except Exception as e:
        st.error(f"Daily update failed: {str(e)}")

else:
    st.error("Mode unknown. Are you from the future?")