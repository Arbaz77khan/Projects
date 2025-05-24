# Import libraries
import os
import pandas as pd
import ta
import ta.momentum
import ta.trend

# Load raw data
df = pd.read_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/raw/tesla_raw.csv')

# Add technical indicators
df['SMA_14'] = ta.trend.sma_indicator(df['Close'], window=14)
df['EMA_14'] = ta.trend.ema_indicator(df['Close'], window=14)
df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd_diff(df['Close'])

# Add close lag features
for i in range(1,8):
    df[f'Close_lag_{i}'] = df['Close'].shift(i)

# Drop rows with NaNs due to technical indicators and lag features
df.dropna(inplace=True)

#Create directory for raw data if not exists
os.makedirs('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed', exist_ok=True)

# Save processed file
df.to_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed/tesla_features.csv', index=False)
print("Processed data saved to /data/processed/tesla_features.csv")