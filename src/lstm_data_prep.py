import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import load_and_combine_data

def prepare_lstm_data(city_name="Delhi", window_size=30, forecast_horizon=3):
    """
    Ek specific city ka data load karta hai, AQI ko scale karta hai, 
    aur LSTM ke liye sliding windows banata hai.
    
    Parameters:
    - city_name: Kis city ka forecast banana hai (default: Delhi)
    - window_size: Past kitne din ka data dekhna hai (default: 30)
    - forecast_horizon: Future kitne din predict karne hain (default: 3)
    
    Returns:
    - X: Input sequences (shape: [samples, window_size, 1])
    - y: Target sequences (shape: [samples, forecast_horizon])
    - scaler: Fitted MinMaxScaler (taaki baad mein inverse_transform kar sakein)
    - df_city: Cleaned dataframe for the city
    """
    # 1. Pura data load karo
    print(f"Loading data for {city_name}...")
    df = load_and_combine_data(data_dir="data")
    
    # 2. Sirf ek city ka data filter karo
    df_city = df[df["City"] == city_name].copy()
    
    if len(df_city) == 0:
        raise ValueError(f"No data found for city: {city_name}")
        
    # 3. Sort by date (Time-series mein order bohot zaroori hai!)
    df_city = df_city.sort_values(by="Date").reset_index(drop=True)
    
    # 4. Missing AQI values ko fill karo (Linear Interpolation)
    # Interpolation ka matlab: Agar Day1=100 aur Day3=120 hai, toh Day2 automatically 110 ho jayega
    df_city["AQI"] = df_city["AQI"].interpolate(method='linear')
    df_city["AQI"] = df_city["AQI"].fillna(method='bfill').fillna(method='ffill')
    
    # 5. Extract target column and Scale it (0 se 1 ke beech)
    # Neural Networks (LSTM) chote numbers pe fast aur better train hote hain
    aqi_values = df_city["AQI"].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    aqi_scaled = scaler.fit_transform(aqi_values)
    
    # 6. Sliding Window Banao
    X, y = [], []
    total_length = len(aqi_scaled)
    
    # Loop over the data
    for i in range(total_length - window_size - forecast_horizon + 1):
        # Input: Day (i) se leke Day (i + window_size) tak
        window = aqi_scaled[i : i + window_size]
        # Target: Uske baad ke 'forecast_horizon' days
        target = aqi_scaled[i + window_size : i + window_size + forecast_horizon]
        
        X.append(window)
        y.append(target)
    
    # Convert lists to numpy arrays (LSTM numpy arrays hi samajhta hai)
    X = np.array(X)
    # Target (y) ka shape (samples, horizon) hona chahiye, isliye flatten kar rahe hain
    y = np.array(y).reshape(len(y), forecast_horizon)
    
    print(f"Generated {len(X)} sequences of window_size {window_size} and horizon {forecast_horizon}.")
    return X, y, scaler, df_city

if __name__ == "__main__":
    # Test kar lete hain ki code chal raha hai ya nahi
    X, y, scaler, df = prepare_lstm_data("Delhi")
    print("X shape:", X.shape) # Expected: (samples, 30, 1)
    print("y shape:", y.shape) # Expected: (samples, 3)























