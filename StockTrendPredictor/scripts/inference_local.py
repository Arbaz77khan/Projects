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
df["sma_14"] = sma_indicator(df["Close"], window=14)
df["ema_14"] = ema_indicator(df["Close"], window=14)
df["rsi_14"] = rsi(df["Close"], window=14)
df["macd"] = macd_diff(df["Close"])
df = df[::-1].reset_index(drop=True)

# Prepare input features
latest_row = df.iloc[-1]
features = {
    "sma_14": latest_row["sma_14"],
    "ema_14": latest_row["ema_14"],
    "rsi_14": latest_row["rsi_14"],
    "macd": latest_row["macd"],
}

# Add lag features
for i in range(1, 8):
    features[f"close_lag{i}"] = recent_close_prices[-i]

X_input = pd.DataFrame([features])

# ---------------------------------------------
# STEP 3: Load regression model
# ---------------------------------------------
model = joblib.load("D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/models/tsla_data_model.pkl")
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