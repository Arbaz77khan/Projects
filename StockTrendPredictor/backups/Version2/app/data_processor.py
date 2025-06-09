# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import logging
import time
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetches data
def collect_data(symbol, max_retries=3, base_wait=1):
    attempts = 0
    while attempts < max_retries:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y', interval='1d')

            if df.empty:
                logging.warning(f"No data found for symbol: {symbol}")
                return None
            
            df.reset_index(inplace=True)
            df = df[['Date', 'Close']]
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

            logging.info(f"Collected 1-year data for symbol {symbol}")
            return df
    
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):  
                logging.info(f"Too Many Requests - raising exception to outer block")
                raise

            attempts += 1
            wait_time = base_wait * (2 ** (attempts))  # Exponential backoff
            wait_time += random.uniform(1, 3)  # Add jitter randomness
            logging.error(f"Attempt {attempts}: Error collecting the data: {e}")

            if attempts < max_retries:
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    logging.error(f"Failed to fetch data for {symbol} after {max_retries} attempts.")
    return None

# Adds technical indicators, close lags
def engineer_features(df):

    # Add technical indicators

    df['sma_14'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=14)
    df['ema_14'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=14)
    df['rsi_14'] = ta.momentum.rsi(df['Close'].squeeze(), window=14)
    df['macd'] = ta.trend.macd_diff(df['Close'].squeeze())

    # Add close lag features

    for i in range(1,8):
        df[f'close_lag{i}'] = df['Close'].shift(i)

    # Drop rows with NaNs due to technical indicators and lag features
    
    df.dropna(inplace=True)
    logging.info("Feature engineering completed.")
    return df

# Adds target columns
def create_targets(df, threshold=0.02):

    # Add future price columns (1 to 7 days ahead)

    for i in range(1,8):
        df[f'target_day_{i}'] = df['Close'].shift(-i)

    # Calculate avarage price of next 7 days

    future_price_cols = [f'target_day_{i}' for i in range(1,8)]
    df['avg_next_7_days'] = df[future_price_cols].mean(axis=1)
        
    # Function to generate BUY/SELL/HOLD label

    def generate_weekly_trend(row):
        try:
            change = (row['avg_next_7_days'] - row['Close'])/ row['Close']
            if change > threshold:
                return "BUY"
            elif change < -threshold:
                return "SELL"
            else:
                return "HOLD"
        except:
            return None
        
    # Add weekly trend label
    df['trend'] = df.apply(generate_weekly_trend, axis=1)
    
    logging.info("Targets added.")
    return df

# Main execution
if __name__ == '__main__':
    
    df = collect_data('TSLA')
    if df is not None:
        df = engineer_features(df)
        df = create_targets(df)   
        print(df['Date']) 
        df.to_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed/data.csv', index=False)   


