import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def ml_model(file_path):
    
    data = pd.read_csv(file_path, usecols=['Date', 'Weekday', 'Month', 'Close', 'Adj Close', 'Upward_Downward_Probability'])

    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    
    print("Data refining in progress...")

    # Step 1: Standardize Features
    
    # scaler = StandardScaler()
    
    # # Select the columns to scale
    # columns_to_scale = ['Day', 'Weekday', 'Month', 'Year', 'Adj Close', 
    #                     'Adj_Close_Lag_1', 'Adj_Close_Lag_2', 
    #                     'Adj_Close_Lag_3', 'Upward_Downward_Probability']
    
    # # Fit and transform the data (assumes 'data' is a DataFrame)
    # scaled_data = scaler.fit_transform(data[columns_to_scale])
    
    # # Created a new Dataframe with scaled values and retain column names
    # scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=data.index)
    
    # # Concatenate the scaled data with the 'Close' column
    # refined_data = pd.concat([data[['Close']], scaled_df], axis=1)

    refined_data = data

    # Step: Cyclic encoding for Weekday and Month
    refined_data['Weekday_sin'] = np.sin(2 * np.pi * data['Weekday'] / 5)
    refined_data['Weekday_cos'] = np.cos(2 * np.pi * data['Weekday'] / 5)

    refined_data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    refined_data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # Combine sin and cos into angles
    data['Weekday_angle'] = np.arctan2(data['Weekday_sin'], data['Weekday_cos'])
    data['Month_angle'] = np.arctan2(data['Month_sin'], data['Month_cos'])
    
    # Step 2: Reduce Dimensionality

    group_1 = refined_data[['Weekday_angle', 'Month_angle']]
    # group_2 = refined_data[['Adj Close', 'Adj_Close_Lag_1', 'Adj_Close_Lag_2', 'Adj_Close_Lag_3']]

    pca_group_1 = PCA(n_components=1)
    refined_data['Temporal_Features'] = pca_group_1.fit_transform(group_1)

    # pca_group_2 = PCA(n_components=1)
    # refined_data['Price_Features'] = pca_group_2.fit_transform(group_2)

    refined_data.drop(columns=['Weekday_sin', 'Weekday_cos', 'Month_sin', 'Month_cos','Weekday_angle', 'Month_angle', 'Weekday', 'Month', 'Adj Close'], inplace=True)

    # Step 3: Clustering  

    features = refined_data[['Temporal_Features', 'Upward_Downward_Probability']]

    # Use K-Means with predefined 3 clusters (bullish, bearish, neutral)
    kmeans = KMeans(n_clusters=3, random_state=42)
    refined_data['Cluster'] = kmeans.fit_predict(features)

    # Step 4: Anomaly Detection
    
    # Anomaly Detection with Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # Fit the model to the data and predict anomalies (-1 for anomalies, 1 for normal points)
    refined_data['Anomaly'] = iso_forest.fit_predict(features)

    print("Data refining done!!!")

    # Save the updated DataFrame to the file
    refined_data.to_csv(file_path.replace(".csv", "_ML.csv"), index=True)

    print(f"File updated at: {file_path}")


if __name__ == '__main__':
    ml_model('D:/Master_Folder/Data Science Course/Projects/StockMarket/stock_data/SUZLON.NS_2023-01-01_to_2024-11-21.csv')


    