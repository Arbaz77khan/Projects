# Import Libraries
import os
import yfinance as yf

#Create directory for raw data if not exists
os.makedirs('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/raw', exist_ok=True)

# Fetch 1 year daily data for TSLA
ticker = yf.Ticker('TSLA')
df = ticker.history(period='1y', interval='1d')

# Save raw data
df.to_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/raw/tesla_raw.csv')
print("Raw data saved to - data/raw/tesla_raw.csv")