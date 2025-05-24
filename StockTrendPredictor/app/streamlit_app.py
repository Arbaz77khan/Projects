# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ta.trend import sma_indicator, ema_indicator, macd_diff
from ta.momentum import rsi

# App structure
st.set_page_config(page_title='Stock Trend Forecaster', layout='centered')
st.title('Stock Trend Prediction')

# Load model
@st.cache_resource
def load_model():
    return joblib.load('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/models/price_multioutput_regressor.pkl')

model = load_model()

# Input values
st.subheader('Enter last 7 days closing pricces')
default_prices = [176.35, 174.88, 175.92, 177.04, 179.32, 178.20, 180.11]

close_prices = []

for i in range(7):
    price = st.number_input(f'Day {-6+i} Close:', value=default_prices[i], format='%.2f')
    close_prices.append(price)

# Ensure we have 7 clean floats
if len(close_prices) == 7 and all(isinstance(p, float) for p in close_prices):
    df = pd.DataFrame({"Close": close_prices[::-1]})  # Reverse to most recent first
    df["SMA_14"] = sma_indicator(df["Close"], window=14)
    df["EMA_14"] = ema_indicator(df["Close"], window=14)
    df["RSI_14"] = rsi(df["Close"], window=14)
    df["MACD"] = macd_diff(df["Close"])
    df = df[::-1].reset_index(drop=True)

    latest_row = df.iloc[-1]
    features = {
        "SMA_14": latest_row["SMA_14"],
        "EMA_14": latest_row["EMA_14"],
        "RSI_14": latest_row["RSI_14"],
        "MACD": latest_row["MACD"],
    }

    for i in range(1, 8):
        features[f"Close_lag_{i}"] = close_prices[-i]

    X_input = pd.DataFrame([features])

    # -----------------------------------------
    # Prediction
    # -----------------------------------------
    st.subheader("🔮 Prediction")
    if st.button("Predict"):
        predicted_prices = model.predict(X_input)[0]
        weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05])
        weighted_avg = np.dot(predicted_prices, weights)
        current_price = close_prices[-1]

        if weighted_avg > current_price * 1.015:
            trend = "BUY 🟢"
            color = "green"
        elif weighted_avg < current_price * 0.985:
            trend = "SELL 🔴"
            color = "red"
        else:
            trend = "HOLD ⚪"
            color = "gray"

        # Output Table
        forecast_df = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(7)],
            "Predicted Price": [f"${p:.2f}" for p in predicted_prices]
        })

        st.table(forecast_df)
        st.markdown("---")
        st.metric(label="📊 Weighted Avg. Predicted Price", value=f"${weighted_avg:.2f}")
        st.metric(label="📌 Current Price", value=f"${current_price:.2f}")
        st.markdown(f"### Final Recommendation: **:{color}[{trend}]**")

# Footer
st.markdown("---")
st.markdown("Built by Arbaz Khan • GitHub: [@arbaz-ai](https://github.com/yourprofile)")