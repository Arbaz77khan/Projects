# Import libraries
import pandas as pd

# Load processed features
df = pd.read_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed/tesla_features.csv')

# Add future price columns (1 to 7 days ahead)
for i in range(1,8):
    df[f'Target_day_{i}_price'] = df['Close'].shift(-i)

# Calculate avarage price of next 7 days
future_price_cols = [f'Target_day_{i}_price' for i in range(1,8)]
df['Avg_price_next_7_days'] = df[future_price_cols].mean(axis=1)
    
# Function to generate BUY/SELL/HOLD label
def generate_weekly_trend(row, threshold=0.015):
    try:
        change = (row['Avg_price_next_7_days'] - row['Close'])/ row['Close']
        if change > threshold:
            return "BUY"
        elif change < -threshold:
            return "SELL"
        else:
            return "HOLD"
    except:
        return None
    
# Add weekly trend label
df['Target_trend_7day_combined'] = df.apply(generate_weekly_trend, axis=1)

# Drop rows with NaNs (from shift at end)
# df.dropna(inplace=True)

# Save processed file
df.to_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed/tesla_final.csv', index=False)
print("Processed data saved to /data/processed/tesla_final.csv")