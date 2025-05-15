import pandas as pd
import numpy as np

def calculate_probability(file_path):
    data = pd.read_csv(file_path)

    print("Calculating Probabilities...")

    up_probabilities = []
    down_probabilities = []
    signals = []

    for index, row in data.iterrows():
        up_score = 0
        down_score = 0
        total_weight = 0

        # SMA (10 vs. 50)
        total_weight +=20
        if row['SMA_10'] > row['SMA_50']:
            up_score +=20
        else:
            down_score +=20

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

        # Calculate probabilities
        up_probability = (up_score / total_weight) * 100
        down_probability = (down_score / total_weight) * 100

        # Determine Signal
        if up_probability > 70:
            signal = 'Buy'
        elif down_probability > 70:
            signal = 'Sell'
        else:
            signal = 'Hold'

        # Append results
        up_probabilities.append(up_probability)
        down_probabilities.append(down_probability)
        signals.append(signal)

    # Add probabilities and signals to the DataFrame
    data['Up_Probability'] = up_probabilities
    data['Down_Probability'] = down_probabilities
    data['Signal'] = signals

    print("Analysis done!!!")

    data.to_csv(file_path, index=False)

    print(f"File location: {file_path}")


if __name__ == '__main__':

    calculate_probability('stock_data/NESTLEIND.NS_2024-01-01_to_2024-11-21.csv')