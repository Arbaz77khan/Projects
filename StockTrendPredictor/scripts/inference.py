# Import libraries
import pandas as pd
import numpy as np
import joblib
from ta.trend import sma_indicator, ema_indicator, macd_diff
from ta.momentum import rsi
import warnings
warnings.filterwarnings('ignore')

# Load latest data
recent_close_prices = [176.35, 174.88, 175.92, 177.04, 179.32, 178.20, 180.11]  # Last 7 closing prices (latest is last)

df = pd.DataFrame({"Close": recent_close_prices[::-1]})  # Most recent first
df["SMA_14"] = sma_indicator(df["Close"], window=14)
df["EMA_14"] = ema_indicator(df["Close"], window=14)
df["RSI_14"] = rsi(df["Close"], window=14)
df["MACD"] = macd_diff(df["Close"])
df = df[::-1].reset_index(drop=True)

# Prepare input features
latest_row = df.iloc[-1]
features = {
    "SMA_14": latest_row["SMA_14"],
    "EMA_14": latest_row["EMA_14"],
    "RSI_14": latest_row["RSI_14"],
    "MACD": latest_row["MACD"],
}

# Add lag features
for i in range(1, 8):
    features[f"Close_lag_{i}"] = recent_close_prices[-i]

X_input = pd.DataFrame([features])

# ---------------------------------------------
# STEP 3: Load regression model
# ---------------------------------------------
model = joblib.load("D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/models/price_multioutput_regressor.pkl")
predicted_prices = model.predict(X_input)[0]

# ---------------------------------------------
# STEP 4: Compute Weighted Average for Trend Inference
# ---------------------------------------------
weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05])
weighted_avg = np.dot(predicted_prices, weights)
current_price = recent_close_prices[-1]

# Trend logic
if weighted_avg > current_price * 1.015:
    trend = "BUY"
elif weighted_avg < current_price * 0.985:
    trend = "SELL"
else:
    trend = "HOLD"

# ---------------------------------------------
# STEP 5: Output
# ---------------------------------------------
print("📈 Predicted Prices for Next 7 Days:")
for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i}: {price:.2f}")

print(f"\n🎯 Weighted Avg Predicted Price: {weighted_avg:.2f}")
print(f"📌 Current Price: {current_price:.2f}")
print(f"🚦 Recommended Trend: {trend}")