import pandas as pd
import numpy as np

def Upward_Downward_Probability(file_path):
    data = pd.read_csv(file_path)

    print("Calculating Probabilities...")

    upward_downward_probabilities = []

    epsilon = 1e-8  # Small value to avoid division by zero

    for index, row in data.iterrows():
        up_score = 0
        down_score = 0
        total_weight = 0

        # SMA (10 vs. 50)
        total_weight += 20
        if row['SMA_10'] > row['SMA_50']:
            up_score += 20
        else:
            down_score += 20

        # EMA (10 vs. 50)
        total_weight += 20
        if row['EMA_10'] > row['EMA_50']:
            up_score += 20
        else:
            down_score += 20

        # MACD Momentum
        total_weight += 15
        if row['MACD'] > row['MACD_Signal']:
            up_score += 15
        else:
            down_score += 15

        # RSI Momentum
        total_weight += 10
        if 30 < row['RSI'] < 70:
            up_score += 5
            down_score += 5
        elif row['RSI'] <= 30:
            up_score += 10
        elif row['RSI'] >= 70:
            down_score += 10

        # Bollinger Bands
        total_weight += 10
        if row['Close'] >= row['Bollinger_Upper']:
            down_score += 10
        elif row['Close'] <= row['Bollinger_Lower']:
            up_score += 10

        # SMA_20 (as Support/Resistance)
        total_weight += 10
        if row['Close'] > row['SMA_20']:
            up_score += 10
        else:
            down_score += 10

        # Volume Confirmation
        total_weight += 5
        average_volume = data['Volume'].mean()
        if row['Volume'] > average_volume:
            up_score += 2.5
            down_score += 2.5

        # Fundamental Analysis (PE and PB Ratios)
        total_weight += 10
        if row['PE_ratio'] > 20 or row['PB_ratio'] > 3:  # Overvalued
            down_score += 5
        elif row['PE_ratio'] < 10 or row['PB_ratio'] < 1:  # Undervalued
            up_score += 5

        # Calculate up_probability and down_probability
        up_probability = (up_score / total_weight)
        down_probability = (down_score / total_weight)

        # Ensure up_probability and down_probability are never zero
        up_probability = max(up_probability, epsilon)
        down_probability = max(down_probability, epsilon)

        # Apply Log Odds Transformation
        log_odds = np.log(up_probability / down_probability)

        upward_downward_probabilities.append(log_odds)

    # Add the Log Odds to the DataFrame
    data['Upward_Downward_Probability'] = upward_downward_probabilities

    print("Analysis done!!!")

    # Save the updated DataFrame to the file
    data.to_csv(file_path, index=False)

    print(f"File updated at: {file_path}")


if __name__ == '__main__':
    Upward_Downward_Probability('D:/Master_Folder/Data Science Course/Projects/StockMarket/stock_data/SUZLON.NS_2023-01-01_to_2024-11-21.csv')
