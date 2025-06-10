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

# ------------------------------------------------

# Project info
if True:
    project_name = 'stock trend predictor'
    latest_version = 2.1
    latest_version_description = 'bug - fixed mode logic;feature - added refresh button, exchange dropdown'
    version_history = {
        1.0: 'initial version - single stock trend predictor static webpage', 
        2.0: 'multi-stock trend predictor with cloud DBMS & daily update function'
    }

# ------------------------------------------------

# App configuration
st.set_page_config(page_title='ProInvest', layout='centered')

# Title Section
st.title("ProInvest — ahead of world!")
st.caption("Your stock assistant for optimizing investment strategies.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Params
mode = int(st.query_params.get("mode", 0))

# Initialize Session State 
if 'symbol' not in st.session_state:
    st.session_state['symbol'] = ''
if 'retry_count' not in st.session_state:
    st.session_state['retry_count'] = 0

conn = connect_db()
if conn is None:
    st.error("Ah!!! Seems database is playing hide n seek! Try refreshing...")
    st.stop()      


def streamlit_inference(inference, trend):
    inference = pd.DataFrame.from_dict(inference, orient="index").reset_index()
    column_names = ["Next", "Close"]
    inference.columns = column_names

    st.success("For the next 7 days, the stock price forcast is:")
    st.dataframe(inference, use_container_width=True)

    day1_price = inference.iloc[0]["Close"]  
    day7_price = inference.iloc[-1]["Close"]
    percentage_change = ((day7_price - day1_price) / day1_price) * 100

    max_price = inference["Close"].max()
    min_price = inference["Close"].min()
    volatility = ((max_price - min_price) / min_price) * 100  # Price fluctuation measure

    # Define analysis message with expanded insights
    analysis_message = f"""
        ### 📈 Stock Trend Analysis
        - **Projected Price Change:** {abs(percentage_change):.2f}%
        - **Price Range:** {min_price:.2f} (Lowest) ➝ {max_price:.2f} (Highest)
        - **Market Volatility:** {volatility:.2f}% (Based on fluctuations in projected values)
    """

    if volatility > 5:
        analysis_message += f"""
            ⚠️ High volatility detected! Expect significant price swings.
        """
    elif volatility > 2:
        analysis_message += f"""
            📊 Moderate volatility detected—some market movements expected.
        """
    else:
        analysis_message += f"""
            ⚪ Low volatility—stock is relatively stable.
        """

    # Adjust recommendation based on percentage change
    if percentage_change > 0:
        analysis_message += f"""
            🚀 **The stock price is projected to rise.**
        """
    elif percentage_change < 0:
        analysis_message += f"""
            🔻 **The stock price is projected to drop.**
        """
    else:
        analysis_message += f"""
            ⏳ **No significant change projected. Wait until next forcast!**
        """

    # Trend-based recommendations
    if trend == 'BUY':
        analysis_message += f"""
        ✅ **Recommendation:** BUY now, as the trend suggests an upward movement.
        """
    elif trend == 'SELL':
        analysis_message += f"""
        🔴 **Recommendation:** SELL now, as prices may decline further.
        """
    else:
        analysis_message += f"""
        ⏳ **Recommendation:** HOLD or WAIT until market fluctuations provide a better opportunity.
        """

    # Display final analysis
    st.markdown(analysis_message)

# Main Mode
if mode == 0:
    st.markdown("### Search for your stock")
    user_input = st.text_input("Drop a stock symbol (e.g., TCS, INFY, TSLA)", value=st.session_state.get('symbol', ''))
    exchange = st.selectbox("Select Exchange", ["NSE", "BSE", "USA"])

    if exchange == "NSE":
        symbol = f"{user_input}.NS"
    elif exchange == "BSE":
        symbol = f"{user_input}.BO"
    else:
        symbol = f"{user_input}"
    
    if st.button("Smash it!") or st.session_state['retry_count'] > 0:
        if not user_input.strip():
            st.warning("Please enter a valid stock symbol before searching.")
            st.stop()

        symbol = symbol.upper()

        st.session_state['symbol'] = symbol

        st.spinner(f"Looking up `{symbol}`...")

        if not check_stock_in_list(conn, symbol):

            st.spinner(f"Looking up `{symbol}`... might take a moment...")

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
                        
                        if st.session_state['retry_count'] <= 10:
                            st.rerun()  
                        else:
                            st.error("Retries exceeded. Please try again later!")
                            st.stop()
                    else:
                        logging.info(e)
                        st.error("Server down! Please try later!")
                        st.stop()

        inference, trend = fetch_inference_result(conn, symbol)
        if inference:
            streamlit_inference(inference, trend)
        else:
            st.error("🚨 Weird! Data exists, but no predictions found.")

    if st.button("Refresh!"):
        st.session_state.pop("symbol", None) 
        st.session_state['retry_count'] = 0  
        st.rerun()

# Daily update (Admin mode)
elif mode == 772001:
    st.markdown("### Daily Update Trigger (Admin Mode)")
    if st.button("Update!"):
        with st.spinner("Initiating backend wizardry... might take a moment"):
            try:
                daily_update(conn)
                st.success("Stocks updated! We’re fresh as morning chai")
            except Exception as e:
                st.error(f"Daily update failed: {str(e)}")

else:
    st.error("Mode unknown. Are you from the future?")


if conn:
    conn.close()