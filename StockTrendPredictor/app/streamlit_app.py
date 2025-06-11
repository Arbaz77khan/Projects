# Import libraries
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
from datetime import datetime
from dotenv import load_dotenv
from io import BytesIO
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# Import project modules
from data_processor import collect_data, engineer_features, create_targets
from db_manager import (connect_db, create_core_table, check_stock_in_list, 
    add_stock_to_list, fetch_symbols, create_table_name, create_processed_table, 
    insert_processed_data, get_latest_date, delete_old_data, update_table, 
    update_inference_result, fetch_inference_result, update_model_meta)
from model_trainer import train_data, upload_model_object_to_drive
from inference import generate_trend_action, run_inference
from stock_data_updator import data_pipeline, daily_update
from project_info import latest_version

# App configuration
warnings.filterwarnings('ignore')
st.set_page_config(page_title='ProInvest', layout='centered')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Session State  
st.session_state.setdefault('symbol', '')
st.session_state.setdefault('retry_count', -1)
MAX_RETRIES = 10 

# URL mode
try:
    mode = int(st.query_params.get("mode", 0))
except:
    mode = 0

# Function: Display Inference Results in Streamlit
def streamlit_inference(symbol, inference, trend):
    inference = pd.DataFrame.from_dict(inference, orient="index").reset_index()
    column_names = ["Next", "Close"]
    inference.columns = column_names

    st.markdown(f"### {symbol} Analysis...")
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

# Title Section
st.title("ProInvest — ahead of world!")
st.caption("Your stock assistant for optimizing investment strategies.")
st.markdown(
    f"""
    <div style="position: fixed; bottom: 5px; right: 10px; font-size: 12px; color: gray;">
        Version {latest_version}
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize database connection
conn = connect_db()
if conn is None:
    st.error("Ah!!! Seems database is playing hide n seek! Try refreshing...")
    st.stop()  

# Main Mode
if mode == 0:
    symbol = st.session_state.get('symbol', '')
    retry_count = st.session_state.get('retry_count', -1)

    if not symbol and retry_count == -1:
        st.markdown("### Search for your stock")
        user_input = st.text_input("Drop a stock symbol (e.g., TCS, INFY, TSLA)").strip().upper()
        exchange = st.selectbox("Select Exchange", ["NSE", "BSE", "USA"])

        if st.button("Smash it!"): 
            if not user_input:
                st.warning("Please enter a valid stock symbol before searching.")
                st.stop()
        
            symbol = f"{user_input}.NS" if exchange == "NSE" else f"{user_input}.BO" if exchange == "BSE" else user_input
            st.session_state.update({'symbol': symbol, 'retry_count': 0})
            st.rerun()

    while 0 <= st.session_state['retry_count'] <= MAX_RETRIES:
        if not check_stock_in_list(conn, symbol):
            with st.spinner(f"Checking the vaults for {symbol}... might take some seconds..."):
                try:
                    data_pipeline(conn, symbol)
                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message or "Too Many Requests" in error_message:
                        st.session_state['retry_count'] += 1
                        wait_time = random.uniform(2, 4)
                        time.sleep(wait_time)
                        logging.info(f"Waiting {wait_time:.2f} seconds before retrying...")
                        st.rerun()  
                    elif "No data found" in  error_message:
                        st.error(f"Oops! The ticker symbol '{symbol}' doesn't seem to exist. Please refresh and enter the exact symbol ")
                        break
                    else:
                        logging.info(e)
                        st.error("Server down! Please try later!")
                        st.stop()
        
        inference, trend = fetch_inference_result(conn, symbol)
        if inference:
            streamlit_inference(symbol, inference, trend)
        else:
            st.error("🚨 Weird! Data exists, but no predictions found.")
        break
    if st.session_state['retry_count'] > MAX_RETRIES:
        st.error("Seems market is busy digging gold. Please try later!")

    if st.button("Refresh!"):
        st.session_state.update({'symbol': '', 'retry_count': -1})
        st.rerun()

# Daily update (Admin mode)
elif mode == 772001:
    st.markdown("### Daily Update Trigger (Admin Mode)")
    tooManyRequestFlag = False

    while st.session_state['retry_count'] < 1:
        with st.spinner("Initiating backend wizardry... might take a moment"):
            try:
                tooManyRequestFlag = False
                st.session_state['retry_count'] += 1
                daily_update(conn)
            except Exception as e:
                tooManyRequestFlag = True
                error_message = str(e)
                if "429" in error_message or "Too Many Requests" in error_message:
                    wait_time = random.uniform(2, 4)
                    time.sleep(wait_time)
                    logging.info(f"Waiting {wait_time:.2f} seconds before retrying...")
                    # st.rerun()  
        if not tooManyRequestFlag:
            break
    else:
        st.error("Daily update failed. Please check log...")

    if not tooManyRequestFlag: 
        st.success("Stocks updated! We’re fresh as morning chai")

    if st.button("Refresh!"):
        st.session_state['retry_count'] = -1
        st.rerun()

else:
    st.error("Mode unknown. Are you from the future?")


if conn:
    conn.close()

st.markdown("---")


