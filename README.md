# 🌬️ Delhi Multivariate AQI Forecasting

A deep-learning-based air quality forecasting system tailored specifically for Delhi's unique environmental conditions (such as the Diwali pollution spike and winter inversions).

## 🚀 Features
- **Multivariate Forecasting:** Predicts base pollutants (PM2.5, PM10, CO, NO2) instead of a direct AQI index.
- **Deep Learning Architecture:** Utilizes Long Short-Term Memory (LSTM) networks across 6 different locations in Delhi.
- **Advanced Feature Engineering:** Incorporates Cyclical Time Encoding (Sin/Cos) and a custom `days_to_diwali` proximity feature to anticipate seasonal pollution spikes.
- **Automated Data Pipeline:** Pulls historical hourly data directly from the Open-Meteo API.
- **Modern UI:** Premium Streamlit application providing 3-day forecasts and health advisories based on official CPCB formulas.

## 🛠️ Technology Stack
- **Data Engineering:** Pandas, Numpy
- **Modeling:** TensorFlow / Keras (LSTMs)
- **Preprocessing:** Scikit-Learn (MinMaxScaler)
- **Frontend:** Streamlit
- **API:** Open-Meteo

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
- `data/`: Contains raw CSV data and `.npy` processed sequences.
- `scripts/`: Data downloading and API ingestion scripts.
- `src/`: Core logic (Data prep, LSTM modeling, CPCB formula logic).
- `models/`: Saved `.h5` LSTM models and `.pkl` scalers.
- `app.py`: The main Streamlit web application.

---
*Developed for advanced predictive analysis of Indian Air Quality.*
