# Import libraries
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed/tesla_final.csv')

# Features (X)
features = ['SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4', 'Close_lag_5', 'Close_lag_6', 'Close_lag_7']
X = df[features]

# Multioutput target columns (Y)
target_cols = [f'Target_day_{i}_price' for i in range(1,8)]
Y = df[target_cols]

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train multioutput regressor
base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
multi_model = MultiOutputRegressor(base_regressor)

multi_model.fit(X_train, Y_train)
print("Multi Output Regression model trained.")

# Evaluate
Y_pred = multi_model.predict(X_test)

# Calculate avg MSE and R2 across all 7 output
mse_total, r2_total = 0, 0
for i, col in enumerate(target_cols):
    mse = mean_squared_error(Y_test.iloc[:,i], Y_pred[:,i])
    r2 = r2_score(Y_test.iloc[:,i], Y_pred[:,i])
    mse_total += mse
    r2_total += r2
    print(f"Day {i + 1} - MSE: {mse:.2f}, R2: {r2:.2f}")

print(f"Average MSE: {mse_total/7:.2f}")
print(f"Average R2: {r2_total/7:.2f}")

# Save model
os.makedirs('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/models', exist_ok=True)
joblib.dump(multi_model, 'D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/models/price_multioutput_regressor.pkl')
print("Model saved to D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/models/price_multioutput_regressor.pkl")