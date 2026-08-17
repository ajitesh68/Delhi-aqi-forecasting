# 🌬️ Delhi Multivariate AQI Forecasting

A deep-learning-based air quality forecasting system for Delhi, using Multivariate LSTM networks trained on historical pollution and weather data to predict **72-hour AQI forecasts** across 6 monitoring stations.

## 🚀 Features

- **Multivariate LSTM Forecasting:** Uses 14 days (336 hours) of historical data — including pollutants (PM2.5, PM10, CO, NO2) and weather parameters (Temperature, Humidity, Pressure, Wind Speed) — to predict the next 72 hours of pollution levels.
- **6 Delhi Locations:** Separate trained models for Anand Vihar, Connaught Place, Dwarka, IGI Airport, Okhla Phase III, and Rohini.
- **Advanced Feature Engineering:** Cyclical time encoding (Sin/Cos for hour/month) and a custom `days_to_diwali` proximity feature to capture seasonal pollution spikes.
- **Automated Data Pipeline:** Pulls real-time hourly data from the Open-Meteo API for live predictions.
- **Official AQI Calculation:** Computes AQI using the official CPCB (Central Pollution Control Board) breakpoint formula.
- **Modern UI:** Premium Streamlit application with 3-day forecasts, health advisories, and interactive charts.

## 📊 Model Architecture & Results

| Parameter | Value |
|---|---|
| **Architecture** | Sequential LSTM (64 → 32 units) + Dropout (0.2) + Dense |
| **Input Shape** | `(336, 14)` — 14 days × 14 features |
| **Output Shape** | `(288,)` — 72 hours × 4 pollutants |
| **Optimizer** | Adam (LR: 0.001 with ReduceLROnPlateau) |
| **Loss Function** | Mean Squared Error (MSE) |
| **EarlyStopping** | Patience = 10 epochs |
| **Final Validation Loss** | ~0.009 - 0.010 MSE |
| **Final Validation MAE** | ~0.065 (~6% error) |

## 🛠️ Technology Stack

- **Data Engineering:** Pandas, NumPy
- **Modeling:** TensorFlow / Keras (LSTM)
- **Preprocessing:** Scikit-Learn (MinMaxScaler)
- **Frontend:** Streamlit
- **API:** Open-Meteo (Historical + Forecast weather data)

## 📊 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit Application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to `http://localhost:8501`.

## 📁 Project Structure

```
india-aqi-analysis/
├── app.py                  # Main Streamlit web application
├── src/
│   ├── lstm_model.py       # LSTM training script
│   ├── data_prep.py        # Data preparation & feature engineering
│   └── aqi_calculator.py   # CPCB AQI formula implementation
├── scripts/
│   └── download_data.py    # Open-Meteo API data ingestion
├── models/
│   ├── <location>_lstm.h5              # Trained LSTM model weights
│   ├── <location>_feature_scaler.pkl   # Feature MinMaxScaler
│   └── <location>_target_scaler.pkl    # Target MinMaxScaler
├── data/                   # Raw CSV data
├── requirements.txt
└── README.md
```

## 📈 How Prediction Works

1. **Data Collection:** Last 14 days of hourly pollution + weather data is fetched via Open-Meteo API.
2. **Preprocessing:** Data is scaled using saved MinMaxScaler (`.pkl` files).
3. **Inference:** The scaled 14-day window is fed into the trained LSTM model (`.h5` file).
4. **Post-processing:** Model output is inverse-scaled to get actual pollutant values (PM2.5, PM10, CO, NO2).
5. **AQI Calculation:** Individual sub-indices are computed using CPCB breakpoints, and the maximum is taken as the final AQI.

---
*Developed for advanced predictive analysis of Indian Air Quality.*
