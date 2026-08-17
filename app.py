import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from src.aqi_formula import calculate_aqi, get_category, get_health_advisory

# --- CONFIGURATION ---
st.set_page_config(page_title="Delhi AQI Forecast 2026", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

LOCATIONS = [
    "Anand Vihar", "Connaught Place", "Dwarka", 
    "IGI Airport", "Okhla Phase III", "Rohini"
]
POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]

# --- STYLING (Vibrant & Premium) ---
st.markdown("""
<style>
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .aqi-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .good { color: #10B981; }
    .satisfactory { color: #10B981; }
    .moderate { color: #FBBF24; }
    .poor { color: #F97316; }
    .very-poor { color: #EF4444; }
    .severe { color: #7F1D1D; }
</style>
""", unsafe_allow_html=True)

# --- CACHING DATA & MODELS ---
@st.cache_resource
def load_forecast_model(location):
    loc_tag = location.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path, compile=False)

@st.cache_data
def load_scalers(location):
    loc_tag = location.lower().replace(" ", "_")
    f_scaler_path = os.path.join(MODELS_DIR, f"{loc_tag}_feature_scaler.pkl")
    t_scaler_path = os.path.join(MODELS_DIR, f"{loc_tag}_target_scaler.pkl")
    if not os.path.exists(f_scaler_path) or not os.path.exists(t_scaler_path):
        return None, None
    return joblib.load(f_scaler_path), joblib.load(t_scaler_path)

@st.cache_data
def get_latest_sequence(location):
    loc_tag = location.lower().replace(" ", "_")
    x_test_path = os.path.join(PROCESSED_DIR, f"{loc_tag}_X_test.npy")
    if not os.path.exists(x_test_path):
        return None
    X_test = np.load(x_test_path)
    # Using the last sequence available as our "current" state
    return X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])

# --- UI LOGIC ---
st.title("🌬️ Delhi Multivariate AQI Forecast (LSTM)")
st.markdown("Predicting PM2.5, PM10, CO, and NO2 up to **72 hours** in the future using AI.")

st.sidebar.header("Settings")
selected_loc = st.sidebar.selectbox("Select Location", LOCATIONS)

if st.sidebar.button("Run 72-Hour Forecast"):
    with st.spinner(f"Running LSTM Model for {selected_loc}..."):
        model = load_forecast_model(selected_loc)
        feature_scaler, target_scaler = load_scalers(selected_loc)
        latest_seq = get_latest_sequence(selected_loc)
        
        if model is None or latest_seq is None or target_scaler is None:
            st.error(f"Models or Data not ready for {selected_loc}. Training might be running.")
        else:
            # 1. Predict scaled values
            pred_scaled = model.predict(latest_seq) # shape (1, 288) -> (72 hours * 4 pollutants)
            
            # 2. Reshape to (72, 4) and Inverse Transform
            pred_reshaped = pred_scaled.reshape(72, 4)
            pred_actual = target_scaler.inverse_transform(pred_reshaped)
            
            # Clip negative values just in case
            pred_actual = np.clip(pred_actual, 0, None)
            
            # 3. Process the forecast (Day 1, Day 2, Day 3 max AQI)
            day_aqis = []
            for day in range(3):
                # Get the 24 hours for each day
                day_data = pred_actual[day*24 : (day+1)*24]
                # Calculate daily max average for each pollutant to get AQI
                max_pm25, max_pm10 = day_data[:, 0].max(), day_data[:, 1].max()
                max_co, max_no2 = day_data[:, 2].max(), day_data[:, 3].max()
                
                aqi, dominant, _ = calculate_aqi(max_pm25, max_pm10, max_co, max_no2)
                category = get_category(aqi)
                advisory = get_health_advisory(category)
                
                day_aqis.append({
                    "Day": f"Day {day+1}",
                    "AQI": aqi,
                    "Category": category,
                    "Dominant": dominant.upper(),
                    "Advisory": advisory
                })
            
            st.success("Forecast generated successfully!")
            
            # 4. Display Cards
            cols = st.columns(3)
            for i, data in enumerate(day_aqis):
                cat_class = data['Category'].lower().replace(" ", "-")
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{data['Day']} Forecast</h3>
                        <div class="aqi-value {cat_class}">{data['AQI']}</div>
                        <h4 class="{cat_class}">{data['Category']}</h4>
                        <p><strong>Primary Pollutant:</strong> {data['Dominant']}</p>
                        <p style="font-size:0.9em; opacity:0.8;">{data['Advisory']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # 5. Display Chart
            st.markdown("### 📈 Hourly Pollutant Trend (Next 72 Hours)")
            df_chart = pd.DataFrame(pred_actual, columns=["PM2.5", "PM10", "CO", "NO2"])
            df_chart.index = [f"Hour {i+1}" for i in range(72)]
            st.line_chart(df_chart)

else:
    st.info("👈 Select a location and click 'Run Forecast' to see predictions.")
