# Installed yfinance library.

import yfinance as yf
import os
import pandas as pd
import numpy as np

def download_yf_stock_data(symbol, start_date, end_date, interval='id', save_path='stock_data'):
    """
    Download stock data from Yahoo Finance and save it as a CSV file.
    
    Parameters:
        symbol (str): Stock ticker symbol (e.g., 'RELIANCE.NS' for Reliance).
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        interval (str): Data interval ('1d', '1wk', '1mo').
        save_path (str): Path to save the CSV file.
    """
    #ensure save_path exits
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download data from Yahoo Finance
    print(f"Downloading data for {symbol} from {start_date} to {end_date} with {interval} interval...")
    data = yf.download(symbol, start = start_date, end = end_date, interval = interval)

    # Define filename and path
    file_path = os.path.join(save_path, f"{symbol}_{start_date}_to_{end_date}.csv")
    data.to_csv(file_path)

    print(f"Data downloaded and saved to {file_path}")

    return file_path

def clean_yf_data(file_path):
    data = pd.read_csv(file_path)

    print("Cleaning data...")

    # Renaming columns
    data.rename(columns={'Price': 'Date'}, inplace=True)

    # cleaning rows
    data.drop(index=[0,1], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Correcting datatypes
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    text_to_float_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open']
    data[text_to_float_cols] = data[text_to_float_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').round(3))
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    data.to_csv(file_path, index=False)

    print(f"Data Cleaned and saved to {file_path}")

def fundamental_indicator(file_path, ticker_symbol):
    data = pd.read_csv(file_path)
    ticker = yf.Ticker(ticker_symbol)

    print("Adding fundamental indicator...")    

    # Fetch financial data
    trailing_eps = ticker.info.get('trailingEps')
    book_value = ticker.info.get('bookValue')

    # Define default values if data is missing
    default_pe_ratio = 15 # Conservative market average
    default_pb_ratio = 1.5 # Conservative book value ratio

    
    if trailing_eps:
        data['PE_ratio'] = data['Close'] / trailing_eps
        data['PE_ratio'] = data['PE_ratio'].round(3)
    else:
        print("Trailing EPS not available, using default P/E ratio for approximation.")
        data['PE_ratio'] = default_pe_ratio

    if book_value:
        data['PB_ratio'] = data['Close'] / book_value
        data['PB_ratio'] = data['PB_ratio'].round(3)
    else:
        print("Book Value not available, using default P/B ratio for approximation.")
        data['PB_ratio'] = default_pb_ratio

    data.to_csv(file_path, index=False)
    print(f"Fundamental indicators added to file {file_path}")
    

def technical_indicator(file_path):
    data = pd.read_csv(file_path)

    print("Adding technical indicators...")

    # Calculate Simple & Exponential Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean().round(3)
    data['SMA_50'] = data['Close'].rolling(window=50).mean().round(3)
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean().round(3)
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean().round(3)

    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].round(3)

    # Calculate Moving Average Convergence Divergence (MACD)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean().round(3)
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean().round(3)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD'] = data['MACD'].round(3)
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean().round(3)

    # Calculate Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean().round(3)
    data['Bollinger_Upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Upper'] = data['Bollinger_Upper'].round(3)
    data['Bollinger_Lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Lower'] = data['Bollinger_Lower'].round(3)

    # Removing NAN rows created due to SMA_50 column
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    data.to_csv(file_path, index=False)

    print(f"Technical Indicators added to file {file_path}")

def ml_model_feature(file_path):
    data = pd.read_csv(file_path)

    print("Adding ML model Features...")

    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    # data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday # 0 Monday, 6 Sunday
    data['Month'] = data['Date'].dt.month
    # data['Year'] = data['Date'].dt.year

    # Generate lag feature for close price
    # for lag in range(1, 4):
    #     data[f'Adj_Close_Lag_{lag}'] = data['Adj Close'].shift(lag)

    # Removing NAN rows created due to SMA_50 column
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    data.to_csv(file_path, index=False)

    print(f"ML model features added to file {file_path}")

def rearranging_data(file_path):
    data = pd.read_csv(file_path)
   
    data = data[['Date', 'Weekday', 'Month', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'PE_ratio', 'PB_ratio', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'SMA_20', 'Bollinger_Upper', 'Bollinger_Lower']]

    data.to_csv(file_path, index=False)

if __name__ == '__main__':
    # User inputs
    symbol = 'SUZLON.NS'  # Ticker symbol for Reliance Industries on NSE, NESTLEIND
    start_date = '2023-01-01'
    end_date = '2024-11-21'
    interval = '1d'  # Daily data

    # Download stock data
    file_path = download_yf_stock_data(symbol, start_date, end_date, interval)

    clean_yf_data(file_path)

    fundamental_indicator(file_path, symbol)

    technical_indicator(file_path)

    ml_model_feature(file_path)

    rearranging_data(file_path)
    