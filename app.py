import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from src.aqi_formula import calculate_aqi, get_category, get_health_advisory
from src.prepare_lstm_data import POLLUTANTS, WEATHER, FEATURES, WINDOW_SIZE, FORECAST_HOURS, add_time_features, add_diwali_feature

st.set_page_config(
    page_title="Delhi AQI Forecasting",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

LOCATIONS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.316},
    "Connaught Place": {"lat": 28.6315, "lon": 77.2167},
    "Dwarka": {"lat": 28.5921, "lon": 77.0460},
    "IGI Airport": {"lat": 28.5562, "lon": 77.1000},
    "Okhla Phase III": {"lat": 28.5308, "lon": 77.2713},
    "Rohini": {"lat": 28.7495, "lon": 77.0565},
}

AQI_COLORS = {
    "Good": "#10B981",
    "Satisfactory": "#34D399",
    "Moderate": "#FBBF24",
    "Poor": "#F97316",
    "Very Poor": "#EF4444",
    "Severe": "#7F1D1D",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .forecast-card {
        background: rgba(30, 41, 59, 0.85);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .forecast-card h3 { margin: 0 0 4px 0; font-size: 1rem; opacity: 0.7; }
    .forecast-card .date-label { font-size: 1.1rem; font-weight: 700; margin-bottom: 10px; }
    .forecast-card .aqi-big { font-size: 3rem; font-weight: 800; margin: 8px 0; }
    .forecast-card .cat-label { font-size: 1.1rem; font-weight: 600; }
    .forecast-card .pollutant-info { font-size: 0.85rem; opacity: 0.75; margin-top: 8px; }
    .forecast-card .advisory { font-size: 0.82rem; opacity: 0.65; margin-top: 6px; }
    .section-title { font-size: 1.3rem; font-weight: 700; margin: 28px 0 12px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_forecast_model(location):
    loc_tag = location.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path, compile=False)


@st.cache_data(ttl=3600)
def load_scalers(location):
    loc_tag = location.lower().replace(" ", "_")
    f_path = os.path.join(MODELS_DIR, f"{loc_tag}_feature_scaler.pkl")
    t_path = os.path.join(MODELS_DIR, f"{loc_tag}_target_scaler.pkl")
    if not os.path.exists(f_path) or not os.path.exists(t_path):
        return None, None
    return joblib.load(f_path), joblib.load(t_path)


@st.cache_data(ttl=1800)
def fetch_live_data(location, days_back=21):
    coords = LOCATIONS[location]
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    try:
        # --- 1. Historical weather from archive API (up to yesterday) ---
        w_archive = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "start_date": start_date, "end_date": yesterday,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
                "timezone": "Asia/Kolkata",
            }, timeout=30
        ).json()

        # --- 2. Today's weather from forecast API ---
        w_forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "start_date": today_str, "end_date": today_str,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
                "timezone": "Asia/Kolkata",
            }, timeout=30
        ).json()

        # Merge weather data
        w_times = w_archive["hourly"]["time"] + w_forecast.get("hourly", {}).get("time", [])
        w_temp = w_archive["hourly"]["temperature_2m"] + w_forecast.get("hourly", {}).get("temperature_2m", [])
        w_hum = w_archive["hourly"]["relative_humidity_2m"] + w_forecast.get("hourly", {}).get("relative_humidity_2m", [])
        w_pres = w_archive["hourly"]["surface_pressure"] + w_forecast.get("hourly", {}).get("surface_pressure", [])
        w_wind = w_archive["hourly"]["wind_speed_10m"] + w_forecast.get("hourly", {}).get("wind_speed_10m", [])

        weather_df = pd.DataFrame({
            "datetime": pd.to_datetime(w_times),
            "temp_c": w_temp,
            "humidity": w_hum,
            "pressure_mb": w_pres,
            "windspeed_kph": w_wind,
        })

        # --- 3. Air quality data (supports current dates natively) ---
        a_resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "start_date": start_date, "end_date": today_str,
                "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
                "timezone": "Asia/Kolkata",
            }, timeout=30
        ).json()

        aq_df = pd.DataFrame({
            "datetime": pd.to_datetime(a_resp["hourly"]["time"]),
            "pm2_5": a_resp["hourly"]["pm2_5"],
            "pm10": a_resp["hourly"]["pm10"],
            "co": a_resp["hourly"]["carbon_monoxide"],
            "no2": a_resp["hourly"]["nitrogen_dioxide"],
        })

        # --- 4. Merge weather + air quality on datetime ---
        df = pd.merge(weather_df, aq_df, on="datetime", how="inner")
        df["location"] = location
        df[FEATURES] = df[FEATURES].interpolate(method="linear")
        df.dropna(subset=FEATURES, inplace=True)
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def prepare_live_sequence(df, feature_scaler):
    df_copy = df.copy()
    df_copy = add_time_features(df_copy)
    df_copy = add_diwali_feature(df_copy)

    feature_cols = FEATURES + [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_weekend", "days_to_diwali",
    ]

    df_copy[feature_cols] = feature_scaler.transform(df_copy[feature_cols])
    values = df_copy[feature_cols].values

    if len(values) < WINDOW_SIZE:
        st.error(f"Not enough data. Need {WINDOW_SIZE} hours, got {len(values)}.")
        return None

    seq = values[-WINDOW_SIZE:]
    return seq.reshape(1, WINDOW_SIZE, len(feature_cols))


def run_forecast(model, sequence, target_scaler):
    pred_scaled = model.predict(sequence, verbose=0)
    pred_reshaped = pred_scaled.reshape(FORECAST_HOURS, len(POLLUTANTS))
    pred_actual = target_scaler.inverse_transform(pred_reshaped)
    pred_actual = np.clip(pred_actual, 0, None)
    return pred_actual


def run_backtesting(df, model, feature_scaler, target_scaler, days_back=7):
    results = []
    total_hours = len(df)

    for day_offset in range(days_back, 0, -1):
        end_idx = total_hours - (day_offset * 24)
        if end_idx < WINDOW_SIZE:
            continue

        slice_df = df.iloc[:end_idx].copy()
        seq = prepare_live_sequence(slice_df, feature_scaler)
        if seq is None:
            continue

        pred = run_forecast(model, seq, target_scaler)
        pred_day1_avg = pred[:24].mean(axis=0)
        pred_aqi, _, _ = calculate_aqi(pred_day1_avg[0], pred_day1_avg[1], pred_day1_avg[2], pred_day1_avg[3])

        actual_start = end_idx
        actual_end = min(end_idx + 24, total_hours)
        if actual_end <= actual_start:
            continue

        actual_slice = df.iloc[actual_start:actual_end]
        actual_avg = actual_slice[POLLUTANTS].mean()
        actual_aqi, _, _ = calculate_aqi(actual_avg["pm2_5"], actual_avg["pm10"], actual_avg["co"], actual_avg["no2"])

        target_date = df.iloc[actual_start]["datetime"]
        results.append({
            "Date": target_date.strftime("%b %d"),
            "Day": target_date.strftime("%A"),
            "Predicted AQI": pred_aqi,
            "Actual AQI": actual_aqi,
            "Difference": abs(pred_aqi - actual_aqi),
        })

    return pd.DataFrame(results)


# ==================== MAIN UI ====================

st.title("🌬️ Delhi AQI Forecasting")
st.caption("LSTM-based 72-hour pollutant forecasting with live API data")

with st.sidebar:
    st.header("⚙️ Settings")
    selected_loc = st.selectbox("📍 Location", list(LOCATIONS.keys()))
    run_btn = st.button("🚀 Run Forecast", use_container_width=True, type="primary")

if run_btn:
    model = load_forecast_model(selected_loc)
    feature_scaler, target_scaler = load_scalers(selected_loc)

    if model is None or feature_scaler is None:
        st.error(f"Model or scalers not found for {selected_loc}. Training may still be running.")
        st.stop()

    with st.spinner("Fetching live data from Open-Meteo API..."):
        live_df = fetch_live_data(selected_loc, days_back=21)

    if live_df is None or len(live_df) < WINDOW_SIZE:
        st.error("Could not fetch enough live data.")
        st.stop()

    with st.spinner("Running LSTM forecast..."):
        seq = prepare_live_sequence(live_df, feature_scaler)
        if seq is None:
            st.stop()
        predictions = run_forecast(model, seq, target_scaler)

    now = datetime.now()

    # ========== 3-DAY FORECAST CARDS (with real dates) ==========
    st.markdown('<div class="section-title">📅 3-Day AQI Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for day_idx in range(3):
        day_data = predictions[day_idx * 24 : (day_idx + 1) * 24]
        day_avg = day_data.mean(axis=0)
        aqi, dominant, _ = calculate_aqi(day_avg[0], day_avg[1], day_avg[2], day_avg[3])
        category = get_category(aqi)
        advisory = get_health_advisory(category)
        color = AQI_COLORS.get(category, "#FBBF24")
        target_date = now + timedelta(days=day_idx + 1)
        date_str = target_date.strftime("%b %d, %A")

        with cols[day_idx]:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="date-label">{date_str}</div>
                <div class="aqi-big" style="color:{color};">{aqi}</div>
                <div class="cat-label" style="color:{color};">{category}</div>
                <div class="pollutant-info">Dominant: {dominant.upper()}</div>
                <div class="advisory">{advisory}</div>
            </div>
            """, unsafe_allow_html=True)

    # ========== HOURLY BREAKDOWN PER DAY ==========
    st.markdown('<div class="section-title">⏰ Hourly Forecast Breakdown</div>', unsafe_allow_html=True)
    for day_idx in range(3):
        target_date = now + timedelta(days=day_idx + 1)
        day_data = predictions[day_idx * 24 : (day_idx + 1) * 24]

        hourly_records = []
        for h in range(len(day_data)):
            hour_time = (target_date.replace(hour=0, minute=0, second=0) + timedelta(hours=h))
            aqi_h, dom_h, _ = calculate_aqi(day_data[h][0], day_data[h][1], day_data[h][2], day_data[h][3])
            hourly_records.append({
                "Time": hour_time.strftime("%I %p"),
                "PM2.5": f"{day_data[h][0]:.1f}",
                "PM10": f"{day_data[h][1]:.1f}",
                "CO": f"{day_data[h][2]:.0f}",
                "NO2": f"{day_data[h][3]:.1f}",
                "AQI": aqi_h,
                "Category": get_category(aqi_h),
            })

        with st.expander(f"📆 {target_date.strftime('%b %d, %A')} — Hourly Details", expanded=(day_idx == 0)):
            st.dataframe(pd.DataFrame(hourly_records), use_container_width=True, hide_index=True)

    # ========== SEPARATE POLLUTANT GRAPHS ==========
    st.markdown('<div class="section-title">📈 Pollutant Trends (Next 72 Hours)</div>', unsafe_allow_html=True)
    hours_index = [(now + timedelta(hours=h+1)).strftime("%b %d %I%p") for h in range(FORECAST_HOURS)]

    pollutant_labels = {"pm2_5": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)", "co": "CO (µg/m³)", "no2": "NO₂ (µg/m³)"}
    chart_cols = st.columns(2)
    for idx, (key, label) in enumerate(pollutant_labels.items()):
        col_idx = POLLUTANTS.index(key)
        chart_df = pd.DataFrame({label: predictions[:, col_idx]}, index=hours_index)
        with chart_cols[idx % 2]:
            st.markdown(f"**{label}**")
            st.line_chart(chart_df, height=250)

    # ========== BACKTESTING: ACTUAL vs PREDICTED ==========
    st.markdown('<div class="section-title">🔍 Model Accuracy — Actual vs Predicted (Past 7 Days)</div>', unsafe_allow_html=True)
    with st.spinner("Running backtesting on past 7 days..."):
        backtest_df = run_backtesting(live_df, model, feature_scaler, target_scaler, days_back=7)

    if len(backtest_df) > 0:
        col_bt1, col_bt2 = st.columns([2, 1])
        with col_bt1:
            chart_bt = backtest_df.set_index("Date")[["Predicted AQI", "Actual AQI"]]
            st.line_chart(chart_bt, height=300)
        with col_bt2:
            st.dataframe(backtest_df, use_container_width=True, hide_index=True)
            avg_diff = backtest_df["Difference"].mean()
            st.metric("Avg Error", f"±{avg_diff:.0f} AQI")
    else:
        st.info("Not enough historical data for backtesting comparison.")

else:
    st.info("👈 Select a location and click **Run Forecast** to see predictions.")
