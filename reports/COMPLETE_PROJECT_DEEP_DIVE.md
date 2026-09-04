# Delhi Multivariate AQI Forecasting — Complete Deep Dive Report

**Author:** Ajitesh  
**Last Updated:** September 2026  
**Model:** Multivariate LSTM (per-location)  
**Forecast:** Next 72 hours (3 days) ke 4 pollutants  
**Input:** Previous 336 hours (14 days) ke 14 features  
**Locations:** Anand Vihar, Connaught Place, Dwarka, IGI Airport, Okhla Phase III, Rohini  
**Live App:** [http://localhost:8501](http://localhost:8501) (Streamlit)  
**GitHub:** [https://github.com/ajitesh68/India-AQI-analysis-prediction](https://github.com/ajitesh68/India-AQI-analysis-prediction)

---

## Table of Contents

1. [Project Ka Big Picture](#1-project-ka-big-picture)
2. [Repository Structure — Har File Ka Role](#2-repository-structure--har-file-ka-role)
3. [Data Pipeline — Raw Data Se Training Data Tak](#3-data-pipeline--raw-data-se-training-data-tak)
4. [FILE: `scripts/download_2026_data.py` — Line by Line](#4-file-scriptsdownload_2026_datapy--line-by-line)
5. [FILE: `src/prepare_lstm_data.py` — Line by Line](#5-file-srcprepare_lstm_datapy--line-by-line)
6. [Feature Engineering — Har Feature Ki Intuition](#6-feature-engineering--har-feature-ki-intuition)
7. [Scaling — Kyon, Kaise, Kya Scaler](#7-scaling--kyon-kaise-kya-scaler)
8. [Sequence Creation — Sliding Window Ka Logic](#8-sequence-creation--sliding-window-ka-logic)
9. [FILE: `src/lstm_model.py` — Line by Line](#9-file-srclstm_modelpy--line-by-line)
10. [LSTM Internals — Kaise Kaam Karta Hai](#10-lstm-internals--kaise-kaam-karta-hai)
11. [FILE: `src/aqi_formula.py` — Line by Line](#11-file-srcaqi_formulapy--line-by-lineine)
12. [FILE: `app.py` — Streamlit App Line by Line](#12-file-apppy--streamlit-app-line-by-line)
13. [Forecast Button Workflow — Step by Step](#13-forecast-button-workflow--step-by-step)
14. [Backtesting — Kya Hota Hai Jab Past 7 Days Compare Karte Hai](#14-backtesting--kya-hota-hai-jab-past-7-days-compare-karte-hai)
15. [Calibration — Distribution Shift Fix](#15-calibration--distribution-shift-fix)
16. [Metrics — MAE/MSE Itna Kam Kyon](#16-metrics--maemse-itna-kam-kyon)
17. [Data Source — Open-Meteo vs CPCB](#17-data-source--open-meteo-vs-cpcb)
18. [Limitations Aur Honest Assessment](#18-limitations-aur-honest-assessment)
19. [Future Improvements — Kya Add/Improve Kar Sakte Ho](#19-future-improvements--kya-addimprove-kar-sakte-ho)
20. [Interview Questions Deep Dive](#20-interview-questions-deep-dive)

---

## 1. Project Ka Big Picture

### Ek Line Mein
"Maine ek end-to-end deep learning system banaya hai jo Delhi ke 6 locations ke liye pichhle 14 din ke hourly pollution + weather + seasonal data se agle 3 din ka hour-by-hour PM2.5, PM10, CO, NO₂ predict karta hai, fir CPCB formula se AQI calculate karke Streamlit dashboard mein dikhata hai."

### Business Problem
- Delhi ki air quality unpredictable hai — kab achhi hogi, kab kharab, log plan nahi kar paate
- Agar 3 din pehle pata ho ki AQI 300+ jaane wala hai, toh mask laga sakte hain, outdoor avoid kar sakte hain
- Government agencies crop burning / Diwali ke around pre-emptive measures le sakti hain

### ML Terms Mein Problem
```
INPUT:  336 hourly rows × 14 features  →  shape (336, 14)
OUTPUT: 72 hourly rows × 4 pollutants  →  shape (72, 4) → flattened to (288,)

Yeh ek MULTI-STEP, MULTI-OUTPUT TIME-SERIES REGRESSION problem hai.
```

### Kyon Pollutants Predict Kiye, Direct AQI Nahi?
1. **Interpretability**: User dekh sakta hai ki AQI high hone ka reason PM2.5 hai ya PM10
2. **Granularity**: Har hour ka har pollutant separately available hai
3. **AQI Nonlinear Hai**: AQI = max(sub-indices). Direct predict karne se model ko nonlinear breakpoints learn karne padenge, jo mushkil hai
4. **Dominant Pollutant**: Pollutant-level prediction se dominant pollutant identify ho jaata hai

---

## 2. Repository Structure — Har File Ka Role

```
india-aqi-analysis/
│
├── app.py                          ← Streamlit web app (349 lines)
│                                     UI + Live API fetch + Model inference + Display
│
├── src/
│   ├── __init__.py                 ← Package marker (version info)
│   ├── prepare_lstm_data.py        ← Data loading, cleaning, feature engineering,
│   │                                 scaling, sequence creation, train/test split
│   ├── lstm_model.py               ← LSTM architecture, training loop, metrics saving
│   └── aqi_formula.py              ← CPCB AQI breakpoints, sub-index calculation,
│                                     category mapping, health advisory
│
├── scripts/
│   └── download_2026_data.py       ← Open-Meteo API se 2026 ka raw data download
│
├── models/
│   ├── <location>_lstm.h5          ← Trained Keras LSTM model weights (6 files)
│   ├── <location>_feature_scaler.pkl ← MinMaxScaler for 14 input features (6 files)
│   ├── <location>_target_scaler.pkl  ← MinMaxScaler for 4 output pollutants (6 files)
│   ├── lstm_metrics.json           ← Per-location scaled MSE/MAE after training
│   └── metrics.json                ← Old XGBoost/RF metrics (historical, not active)
│
├── data/
│   ├── raw/
│   │   ├── delhi-weather-aqi-2025.csv  ← 52,560 rows, 16 columns, 6 locations
│   │   └── delhi-weather-aqi-2026.csv  ← 32,976 rows, 13 columns, Jan-Aug 2026
│   └── processed/
│       └── <location>_{X_train,X_test,y_train,y_test}.npy  ← 24 NumPy arrays
│
├── reports/                        ← Documentation & analysis
├── notebooks/                      ← Jupyter notebooks (exploration)
├── requirements.txt                ← 7 Python packages
├── .env                            ← API key (OpenWeather, not actively used)
├── .gitignore                      ← Excludes .venv, __pycache__, etc.
├── README.md                       ← Project overview
└── LICENSE                         ← MIT License
```

### Data ka Size
| File | Rows | Columns | Locations | Period |
|---|---|---|---|---|
| `delhi-weather-aqi-2025.csv` | 52,560 | 16 | 6 | Full 2025 |
| `delhi-weather-aqi-2026.csv` | 32,976 | 13 | 6 | Jan–Aug 2026 |
| **Total** | **85,536** | — | 6 | ~20 months |

Per location: ~14,256 hourly rows (594 days × 24 hours)

---

## 3. Data Pipeline — Raw Data Se Training Data Tak

```
┌─────────────────────────────┐
│  Open-Meteo API             │
│  (Weather + Air Quality)    │
└──────────┬──────────────────┘
           │ download_2026_data.py
           ▼
┌─────────────────────────────┐
│  data/raw/                  │
│  delhi-weather-aqi-2025.csv │
│  delhi-weather-aqi-2026.csv │
└──────────┬──────────────────┘
           │ prepare_lstm_data.py
           │
           ├─→ load_and_combine()     → Sab CSVs merge
           ├─→ add_time_features()    → hour_sin/cos, month_sin/cos, is_weekend
           ├─→ add_diwali_feature()   → days_to_diwali
           ├─→ interpolate + dropna   → Missing value handling
           ├─→ MinMaxScaler.fit()     → Feature scaling (0-1)
           ├─→ create_sequences()     → Sliding window → X(336,14), y(72,4)
           ├─→ 85/15 split            → Train/Test chronological split
           │
           ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  data/processed/            │     │  models/                    │
│  <loc>_X_train.npy          │     │  <loc>_feature_scaler.pkl   │
│  <loc>_X_test.npy           │     │  <loc>_target_scaler.pkl    │
│  <loc>_y_train.npy          │     └─────────────────────────────┘
│  <loc>_y_test.npy           │
└──────────┬──────────────────┘
           │ lstm_model.py
           ▼
┌─────────────────────────────┐
│  models/                    │
│  <loc>_lstm.h5              │
│  lstm_metrics.json          │
└─────────────────────────────┘
```

---

## 4. FILE: `scripts/download_2026_data.py` — Line by Line

Ye file Open-Meteo API se 2026 ka data download karke CSV banati hai.

### Lines 1-3: Imports
```python
import requests          # HTTP requests ke liye (API call)
import pandas as pd      # Data manipulation
from datetime import datetime  # Current date ke liye
```

### Lines 5-12: Location Dictionary
```python
LOCATIONS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.316},
    "Connaught Place": {"lat": 28.6315, "lon": 77.2167},
    "Dwarka": {"lat": 28.5921, "lon": 77.0460},
    "IGI Airport": {"lat": 28.5562, "lon": 77.1000},
    "Okhla Phase III": {"lat": 28.5308, "lon": 77.2713},
    "Rohini": {"lat": 28.7495, "lon": 77.0565},
}
```
**Intuition**: Ye 6 Delhi ke CPCB monitoring stations hain. Lat/lon approximate coordinates hain — Open-Meteo nearest grid point ka data deta hai, exact station sensor ka nahi. Yeh ek limitation hai.

### Lines 14-18: Config
```python
START_DATE = "2026-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")   # Aaj ki date, e.g., "2026-09-04"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
```
**Kyon 2 different APIs?**
- Weather API → temperature, humidity, pressure, wind speed
- Air Quality API → PM2.5, PM10, NO₂, CO
- Dono alag APIs hain Open-Meteo pe, isliye 2 calls zaroori hain

### Lines 21-41: `fetch_weather(lat, lon)`
```python
def fetch_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(WEATHER_URL, params=params)
    resp.raise_for_status()            # Error throw karo agar API fail ho
    data = resp.json()["hourly"]       # JSON response ka "hourly" section
    df = pd.DataFrame(data)            # Dict → DataFrame
    df.rename(columns={                # API column names → hamara naming convention
        "time": "datetime",
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure_mb",
        "wind_speed_10m": "windspeed_kph",
    }, inplace=True)
    return df
```
**API Response Structure:**
```json
{
  "hourly": {
    "time": ["2026-01-01T00:00", "2026-01-01T01:00", ...],
    "temperature_2m": [12.3, 11.8, ...],
    "relative_humidity_2m": [78, 82, ...],
    ...
  }
}
```

### Lines 44-62: `fetch_air_quality(lat, lon)`
Same pattern, but air quality API se:
```python
def fetch_air_quality(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,carbon_monoxide",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df.rename(columns={
        "time": "datetime",
        "nitrogen_dioxide": "no2",       # API name → hamara naam
        "carbon_monoxide": "co",
    }, inplace=True)
    return df
```

### Lines 65-101: `download_all()`
```python
def download_all():
    all_frames = []
    for name, coords in LOCATIONS.items():
        print(f"Downloading {name}...")
        weather = fetch_weather(coords["lat"], coords["lon"])
        air = fetch_air_quality(coords["lat"], coords["lon"])

        # Weather + Air Quality merge on datetime
        merged = weather.merge(air, on="datetime", how="inner")
        merged["location"] = name
        merged["lat"] = coords["lat"]
        merged["lon"] = coords["lon"]

        # Datetime ko date_ist aur time_ist mein split (2025 CSV ke format match ke liye)
        dt = pd.to_datetime(merged["datetime"])
        merged["date_ist"] = dt.dt.strftime("%d/%m/%Y")    # "04/09/2026"
        merged["time_ist"] = dt.dt.hour.astype(str) + ":00" # "15:00"
        # NOTE: Windows pe %-H nahi chalta, isliye .hour.astype(str) use kiya

        # Column ordering — consistent format
        merged = merged[[
            "date_ist", "time_ist", "location", "lat", "lon",
            "temp_c", "humidity", "pressure_mb", "windspeed_kph",
            "pm2_5", "pm10", "co", "no2",
        ]]

        all_frames.append(merged)
        print(f"  {name}: {len(merged)} rows")

    result = pd.concat(all_frames, ignore_index=True)
    output_path = "data/raw/delhi-weather-aqi-2026.csv"
    result.to_csv(output_path, index=False)
```
**Key Points:**
- `inner` join ka matlab sirf wahi rows rahein jahan weather AUR air quality dono available hain
- 2025 CSV mein extra columns the (`condition_text`, `description`, `aqi_index`) — 2026 mein nahi
- Windows compatibility: `dt.dt.hour.astype(str)` use kiya, kyunki `strftime("%-H")` Windows pe fail hota hai

---

## 5. FILE: `src/prepare_lstm_data.py` — Line by Line

Ye file **training pipeline ka core** hai. Raw CSV → processed NumPy arrays + scalers.

### Lines 1-10: Imports & Paths
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib                  # Scaler objects save/load ke liye
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__ = src/prepare_lstm_data.py
# dirname = src/
# dirname(dirname) = india-aqi-analysis/   ← project root
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
```

### Lines 12-16: Constants (BAHUT IMPORTANT)
```python
POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]       # 4 target columns
WEATHER = ["temp_c", "humidity", "pressure_mb", "windspeed_kph"]  # 4 weather columns
FEATURES = POLLUTANTS + WEATHER                     # 8 base features
WINDOW_SIZE = 336   # 14 days × 24 hours = 336 hourly rows → model ka input window
FORECAST_HOURS = 72 # 3 days × 24 hours = 72 hourly rows → model ka output horizon
```
**Intuition:**
- **WINDOW_SIZE = 336**: Model ko 14 din ka context diya jaata hai kyunki pollution patterns weekly cycles follow karte hain (weekday vs weekend traffic), aur weather patterns bhi 7-14 din ke cycles mein change hote hain
- **FORECAST_HOURS = 72**: 3 din ka forecast — practical use ke liye. 1 din too short, 7 din too uncertain

### Lines 18-24: Diwali Dates Dictionary
```python
DIWALI_DATES = {
    2015: "2015-11-11", 2016: "2016-10-30", 2017: "2017-10-19",
    2018: "2018-11-07", 2019: "2019-10-27", 2020: "2020-11-14",
    2021: "2021-11-04", 2022: "2022-10-24", 2023: "2023-11-12",
    2024: "2024-11-01", 2025: "2025-10-20", 2026: "2026-11-08",
    2027: "2027-10-29",
}
```
**Kyon?** Delhi mein Diwali ke around firecracker pollution se AQI 400-500+ jaata hai. Model ko yeh seasonal spike samjhana zaroori hai.

### Lines 27-53: `load_and_combine()` — CSV Files Merge Karna
```python
def load_and_combine():
    frames = []
    for f in sorted(os.listdir(RAW_DIR)):
        # Sirf hamare naming pattern wali files pick karo
        if f.startswith("delhi-weather-aqi") and f.endswith(".csv"):
            path = os.path.join(RAW_DIR, f)
            df = pd.read_csv(path)

            # Extra columns drop karo jo 2025 CSV mein hain par model ko nahi chahiye
            drop_cols = ["aqi_index", "condition_text", "description"]
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

            # Datetime parsing — 2025 aur 2026 CSV ka format alag hai
            sample_time = str(df["time_ist"].iloc[0])
            if "-" in sample_time:
                # 2025 format: time_ist = "2025-01-01T00:00" (ISO format with dash)
                df["datetime"] = pd.to_datetime(df["time_ist"])
            else:
                # 2026 format: date_ist = "01/01/2026", time_ist = "0:00"
                df["datetime"] = pd.to_datetime(
                    df["date_ist"] + " " + df["time_ist"].astype(str),
                    dayfirst=True,    # DD/MM/YYYY format
                    format="mixed",   # Mixed parsing mode
                )

            frames.append(df)
            print(f"Loaded {f}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["location", "datetime"], inplace=True)  # Location-wise sort
    combined.reset_index(drop=True, inplace=True)
    return combined
```
**Key Points:**
- `sorted(os.listdir())` → files alphabetical order mein read hoti hain (2025 pehle, 2026 baad mein)
- Datetime parsing mein defensive coding hai — 2025 CSV ka format alag tha (ISO with `-` in time), 2026 ka alag (`DD/MM/YYYY` + `H:00`)
- Sort `["location", "datetime"]` pe hota hai — yeh crucial hai kyunki sliding window location-wise banana hai
- Result: ~85,536 rows ka combined DataFrame

### Lines 56-66: `add_time_features(df)` — Cyclical Time Encoding
```python
def add_time_features(df):
    hour = df["datetime"].dt.hour       # 0, 1, 2, ..., 23
    month = df["datetime"].dt.month     # 1, 2, ..., 12
    dow = df["datetime"].dt.dayofweek   # 0=Monday, ..., 6=Sunday

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(int)   # Saturday=5, Sunday=6 → 1; rest → 0
    return df
```

**Deep Intuition — Kyon Sin/Cos?**

Agar hum hour ko seedha integer (0-23) dein:
- Model ko lagega ki Hour 23 aur Hour 0 mein `|23 - 0| = 23` ka gap hai
- Reality mein 23:00 aur 00:00 sirf 1 hour apart hain

Sin/Cos se solve hota hai:
```
Hour 0:  sin = 0.000,  cos = 1.000
Hour 6:  sin = 1.000,  cos = 0.000
Hour 12: sin = 0.000,  cos = -1.000
Hour 18: sin = -1.000, cos = 0.000
Hour 23: sin = -0.259, cos = 0.966
```
Ab Hour 23 (sin=-0.259, cos=0.966) aur Hour 0 (sin=0.000, cos=1.000) — bahut paas hain! ✅

**Kyon 2 values chahiye (sin AUR cos)?**
- Sirf sin se: Hour 3 aur Hour 21 ka sin same hoga (dono ~0.707)
- Sin + Cos milake unique point banata hai unit circle pe → har hour uniquely identify hota hai

**Same logic month ke liye**: December (12) aur January (1) paas hain → sin/cos se capture hota hai

**is_weekend**: Weekend pe traffic kam hota hai → vehicular pollution pattern different. Simple 0/1 feature enough hai.

**Kya yeh LSTM ke saath standard practice hai?**
Haan! Time-series deep learning mein cyclical encoding ek **standard feature engineering technique** hai. Koi bhi LSTM/GRU/Transformer based time-series model mein ye karte hain jab periodic patterns (daily, weekly, yearly) capture karne hon.

### Lines 69-84: `add_diwali_feature(df)` — Seasonal Event Feature
```python
def add_diwali_feature(df):
    years = df["datetime"].dt.year.unique()   # [2025, 2026]
    diwali_timestamps = []
    for y in years:
        if y in DIWALI_DATES:
            diwali_timestamps.append(pd.Timestamp(DIWALI_DATES[y]))
    # diwali_timestamps = [Timestamp("2025-10-20"), Timestamp("2026-11-08")]

    def calc_days_to_diwali(dt):
        if not diwali_timestamps:
            return 999                    # Agar Diwali date nahi pata, default value
        diffs = [abs((dt - d).days) for d in diwali_timestamps]
        nearest_idx = np.argmin(diffs)    # Sabse paas wali Diwali dhundho
        return (diwali_timestamps[nearest_idx] - dt).days
        # Positive = Diwali aane waali hai
        # Negative = Diwali guzar chuki hai
        # Zero = Aaj Diwali hai

    df["days_to_diwali"] = df["datetime"].apply(calc_days_to_diwali)
    return df
```
**Intuition:**
- Diwali se 7 din pehle se 3-4 din baad tak pollution extremely high rehta hai
- Model ko `days_to_diwali = 5` dekh ke samajh aayega ki "ab pollution spike aa sakta hai"
- Signed value hai: +5 = 5 din mein Diwali aayegi, -3 = 3 din pehle thi

### Lines 87-92: `create_sequences()` — Sliding Window
```python
def create_sequences(data, target_data, window_size, forecast_hours):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_hours + 1):
        X.append(data[i : i + window_size])               # 336 rows slice
        y.append(target_data[i + window_size : i + window_size + forecast_hours])  # Next 72 rows
    return np.array(X), np.array(y)
```

**Visual Example (simplified, window=3, forecast=2):**
```
Data: [A, B, C, D, E, F, G, H]

i=0: X=[A,B,C]  y=[D,E]
i=1: X=[B,C,D]  y=[E,F]
i=2: X=[C,D,E]  y=[F,G]
i=3: X=[D,E,F]  y=[G,H]
```

**Actual dimensions:**
```
Data length: ~14,256 rows per location (after cleaning)
Possible sequences: 14,256 - 336 - 72 + 1 = 13,849 sequences

Each X[i] shape: (336, 14)  — 336 hours, 14 scaled features
Each y[i] shape: (72, 4)    — 72 hours, 4 scaled pollutants

X shape: (13849, 336, 14)
y shape: (13849, 72, 4)
```

**Important: `data` = scaled features (14 columns), `target_data` = scaled pollutants (4 columns)**
- Input mein weather + pollutants + engineered features sab hain (14)
- Output mein sirf pollutants hain (4) — kyunki hum weather predict nahi kar rahe

### Lines 95-120: `prepare_location_data()` — Per-Location Pipeline
```python
def prepare_location_data(df, location):
    loc_df = df[df["location"] == location].copy()   # Sirf is location ka data
    loc_df.sort_values("datetime", inplace=True)
    loc_df.reset_index(drop=True, inplace=True)

    feature_cols = FEATURES + [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_weekend", "days_to_diwali",
    ]
    # feature_cols = 8 base + 6 engineered = 14 columns

    # Missing value handling
    loc_df[FEATURES] = loc_df[FEATURES].interpolate(method="linear")
    # Linear interpolation: agar row 5 = 100, row 7 = 200, toh row 6 = 150
    loc_df.dropna(subset=FEATURES, inplace=True)
    # Agar start/end pe NaN hai (interpolation nahi kar paata), toh drop

    # TARGET SCALER — sirf 4 pollutants ke liye (y values)
    target_scaler = MinMaxScaler()
    target_values = target_scaler.fit_transform(loc_df[POLLUTANTS])
    # pm2_5: 0-500 → 0-1
    # pm10:  0-600 → 0-1
    # co:    0-50000 → 0-1
    # no2:   0-800 → 0-1

    # FEATURE SCALER — 14 input features ke liye (X values)
    scaler = MinMaxScaler()
    feature_values = scaler.fit_transform(loc_df[feature_cols])
    # Sab 14 features 0-1 range mein

    # Sliding window sequences
    X, y = create_sequences(feature_values, target_values, WINDOW_SIZE, FORECAST_HOURS)

    # Chronological split — 85% train, 15% test
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # IMPORTANT: Random shuffle NAHI kiya — time-series mein future data train mein nahi jaana chahiye

    return X_train, X_test, y_train, y_test, scaler, target_scaler
```

**Kyon 2 Alag Scalers?**
- **Feature scaler** (14 features): Input X ko scale karta hai. Inference mein bhi same scaler use hoga.
- **Target scaler** (4 pollutants): Output y ko scale karta hai. Model ka output inverse_transform se actual values mein aayega.
- Agar ek hi scaler hota toh inverse_transform galat hota — 14 columns ka scaler 4 columns pe lagana impossible hai

### Lines 123-159: `run()` — Main Pipeline Executor
```python
def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)   # data/processed/ banao agar nahi hai
    os.makedirs(MODELS_DIR, exist_ok=True)       # models/ banao agar nahi hai

    print("Loading data...")
    df = load_and_combine()          # Sab CSVs merge → ~85,536 rows

    print("Adding time features...")
    df = add_time_features(df)       # 5 new columns add

    print("Adding Diwali feature...")
    df = add_diwali_feature(df)      # 1 new column add

    locations = sorted(df["location"].unique())
    # ['Anand Vihar', 'Connaught Place', 'Dwarka', 'IGI Airport', 'Okhla Phase III', 'Rohini']

    for loc in locations:
        print(f"\nPreparing {loc}...")
        X_train, X_test, y_train, y_test, scaler, target_scaler = prepare_location_data(df, loc)

        loc_tag = loc.lower().replace(" ", "_")    # "Anand Vihar" → "anand_vihar"

        # NumPy arrays save karo (training data)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_test.npy"), X_test)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_test.npy"), y_test)

        # Scalers save karo (inference mein chahiye)
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{loc_tag}_feature_scaler.pkl"))
        joblib.dump(target_scaler, os.path.join(MODELS_DIR, f"{loc_tag}_target_scaler.pkl"))

        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    print("\nData preparation complete.")
```

**Output Example:**
```
Preparing Anand Vihar...
  Train: X=(11771, 336, 14), y=(11771, 72, 4)
  Test:  X=(2078, 336, 14), y=(2078, 72, 4)
```
Matlab 11,771 training sequences hain, har ek mein 336 rows × 14 features ka input hai, aur 72 rows × 4 pollutants ka target hai.

---

## 6. Feature Engineering — Har Feature Ki Intuition

### 14 Features Ka Complete Breakdown

| # | Feature | Type | Range | Intuition |
|---|---|---|---|---|
| 1 | `pm2_5` | Pollutant | 0-500+ µg/m³ | Fine particles, lungs mein jaate hain. Most dangerous. |
| 2 | `pm10` | Pollutant | 0-600+ µg/m³ | Coarse particles, dust/construction se aate hain. |
| 3 | `co` | Pollutant | 0-50000 µg/m³ | Carbon Monoxide, vehicles se aata hai. |
| 4 | `no2` | Pollutant | 0-800 µg/m³ | Nitrogen Dioxide, diesel vehicles aur power plants se. |
| 5 | `temp_c` | Weather | -5 to 50°C | Temperature affects atmospheric mixing. Cold → inversion → trapped pollution. |
| 6 | `humidity` | Weather | 0-100% | High humidity + pollution → smog/haze. |
| 7 | `pressure_mb` | Weather | 950-1050 mb | Low pressure → wind/rain → pollution disperse. High pressure → stable → trapped. |
| 8 | `windspeed_kph` | Weather | 0-60+ kph | Wind blows pollution away. Low wind = bad AQI. |
| 9 | `hour_sin` | Engineered | -1 to 1 | Cyclical hour encoding (sin component) |
| 10 | `hour_cos` | Engineered | -1 to 1 | Cyclical hour encoding (cos component) |
| 11 | `month_sin` | Engineered | -1 to 1 | Cyclical month encoding (sin component) |
| 12 | `month_cos` | Engineered | -1 to 1 | Cyclical month encoding (cos component) |
| 13 | `is_weekend` | Engineered | 0 or 1 | Weekend traffic pattern different → less vehicular pollution |
| 14 | `days_to_diwali` | Engineered | -365 to +365 | Diwali proximity → firecracker pollution spike |

### Kyon Ye 14 Features Specifically?

**Pollutants as input (self-regression):**
- AQI time-series mein **autocorrelation** hota hai — agar aaj pollution high hai toh kal bhi high rehne ke chances zyada hain
- Past pollution values future pollution ka best predictor hain

**Weather as input (external driver):**
- Pollution sirf emissions se nahi banta — **atmospheric conditions** decide karti hain ki pollution disperse hoga ya trap hoga
- Temperature inversion (cold night + warm air above) = pollution trap
- Rain = pollution wash → AQI drop
- Wind = pollution disperse

**Engineered features (temporal context):**
- Rush hour (8am, 6pm) pe pollution zyada — hour encoding se model samjhega
- Winter (Nov-Feb) mein inversion zyada — month encoding se seasonal pattern
- Weekend pe traffic kam — is_weekend se capture
- Diwali = extreme spike — days_to_diwali se special event handling

---

## 7. Scaling — Kyon, Kaise, Kya Scaler

### Kyon Scaling Zaroori Hai?

Neural networks **gradient descent** se seekhte hain. Agar features ki range bahut alag ho:
```
pm2_5:    0-500
co:       0-50000      ← 100x bada
humidity: 0-100
hour_sin: -1 to 1
```
Toh:
1. **CO ke gradient dominate karenge** kyunki uski value badi hai
2. Model slowly train hoga ya **converge hi nahi karega**
3. Loss landscape mein elongated valleys banenge → optimizer ko navigate karna mushkil hoga

### MinMaxScaler Kya Karta Hai?
```
scaled_value = (value - min) / (max - min)
```
- Har feature ko 0-1 range mein laata hai
- Example: CO range [0, 50000] → CO=25000 → scaled = 25000/50000 = 0.5

### 2 Alag Scalers Kyon?

```python
# Feature scaler: 14 features ke liye
scaler = MinMaxScaler()
feature_values = scaler.fit_transform(loc_df[feature_cols])  # shape: (N, 14)

# Target scaler: 4 pollutants ke liye  
target_scaler = MinMaxScaler()
target_values = target_scaler.fit_transform(loc_df[POLLUTANTS])  # shape: (N, 4)
```

**Problem agar ek hi scaler hota:**
- Model output shape: (72, 4) — sirf 4 pollutants
- Feature scaler: 14 features ke liye fit hua
- `inverse_transform(output)` → error! 4 columns ko 14-column scaler se inverse kaise karoge?

**Isliye:**
- Feature scaler → sirf input transform ke liye
- Target scaler → output predict hone ke baad inverse transform ke liye
- Dono alag-alag `.pkl` files mein save hote hain

### Scaler Leakage Issue (Honest Limitation)
```python
# Current code: scaler poore location data pe fit hota hai
scaler.fit_transform(loc_df[feature_cols])  # Includes test data!
```
Ideally sirf training data pe fit hona chahiye:
```python
# Better approach:
scaler.fit(train_data)           # Only train data pe fit
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)   # Test data sirf transform
```
Current pipeline mein test data ki min/max bhi scaler mein hai — yeh mild information leakage hai. Real world mein mostly minor impact hota hai, par production-level code mein fix karna chahiye.

---

## 8. Sequence Creation — Sliding Window Ka Logic

### Input: (336, 14) → Output: (72, 4) — Yeh Actually Kya Hai?

Socho ek spreadsheet hai:

```
Row 1:    [pm2_5=120, pm10=200, co=5000, no2=80, temp=12, humid=78, press=1013, wind=5, h_sin=0.0, h_cos=1.0, m_sin=0.5, m_cos=0.87, wknd=0, diwali=45]
Row 2:    [pm2_5=115, pm10=190, co=4800, no2=75, ...]
...
Row 336:  [pm2_5=130, pm10=210, co=5200, no2=85, ...]
```

Yeh 336 rows ka "window" → model ka **ek input sample** hai.

Model dekhta hai: "Pichhle 14 din mein har ghante ka pollution, weather, time — sab kuch dekh ke bata ki agle 72 ghante mein kya hoga?"

Output:
```
Hour 1:  [pm2_5=125, pm10=205, co=5100, no2=82]
Hour 2:  [pm2_5=128, pm10=208, co=5150, no2=84]
...
Hour 72: [pm2_5=140, pm10=220, co=5300, no2=90]
```

### Sliding Window Visualization

```
Timeline: ─────────────────────────────────────────────────────→
          |←── 336 hours ──→|←── 72h ──→|
Sample 1: [████████████████████][▓▓▓▓▓▓]
Sample 2:  [████████████████████][▓▓▓▓▓▓]
Sample 3:   [████████████████████][▓▓▓▓▓▓]
...
          ████ = input (X)    ▓▓▓ = target (y)
```

Har sample pichle wale se 1 hour shift hota hai. Isliye ~14,000 samples bante hain ek location se.

---

## 9. FILE: `src/lstm_model.py` — Line by Line

### Lines 1-15: Imports & Constants
```python
import numpy as np
import os
import argparse           # Command-line arguments ke liye
import json               # Metrics save ke liye
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]
FORECAST_HOURS = 72
```

### Lines 18-28: `build_model()` — LSTM Architecture
```python
def build_model(input_shape, output_size):
    model = Sequential([
        # Layer 1: LSTM with 64 units
        LSTM(64, return_sequences=True, input_shape=input_shape),
        # input_shape = (336, 14)
        # return_sequences=True → har timestep ka output next layer ko jaayega
        # Output shape: (336, 64)

        Dropout(0.2),
        # Training mein randomly 20% neurons off karo → overfitting reduce

        # Layer 2: LSTM with 32 units
        LSTM(32, return_sequences=False),
        # return_sequences=False → sirf last timestep ka output (summary vector)
        # Output shape: (32,) ← poore 336 timesteps ka compressed representation

        Dropout(0.2),

        # Dense layer — nonlinear combination
        Dense(64, activation="relu"),
        # 32 → 64 neurons, ReLU activation
        # Yeh intermediate learned representation hai

        # Output layer
        Dense(output_size, activation="linear"),
        # output_size = 72 × 4 = 288
        # Linear activation kyunki regression hai (koi bhi value aa sakti hai)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",         # Mean Squared Error — regression ke liye standard
        metrics=["mae"]     # Mean Absolute Error — human-readable error metric
    )
    return model
```

**Architecture Flow:**
```
Input: (batch, 336, 14)
   ↓
LSTM(64, return_seq=True) → (batch, 336, 64)    # Har hour ka 64-dim representation
   ↓
Dropout(0.2)
   ↓
LSTM(32, return_seq=False) → (batch, 32)         # 336 hours ka summary → 32-dim vector
   ↓
Dropout(0.2)
   ↓
Dense(64, relu) → (batch, 64)                    # Nonlinear transformation
   ↓
Dense(288, linear) → (batch, 288)                # 72h × 4 pollutants = 288 outputs
```

### Lines 31-72: `train_location()` — Training Loop
```python
def train_location(loc_tag, epochs=100, batch_size=64):
    # Load preprocessed NumPy arrays
    X_train = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_test.npy"))

    # y shape: (samples, 72, 4) → flatten to (samples, 288)
    # Kyunki Dense output layer 288 values deta hai, 2D matrix nahi
    y_train_flat = y_train.reshape(y_train.shape[0], -1)    # (11771, 288)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)       # (2078, 288)

    input_shape = (X_train.shape[1], X_train.shape[2])      # (336, 14)
    output_size = y_train_flat.shape[1]                      # 288

    model = build_model(input_shape, output_size)

    callbacks = [
        # EarlyStopping: Agar 10 epochs tak val_loss improve na ho, training rok do
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True    # Best epoch ke weights wapas restore karo
        ),
        # ReduceLROnPlateau: Agar 5 epochs se val_loss improve nahi, learning rate half karo
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,          # LR = LR × 0.5
            patience=5,
            min_lr=1e-6          # 0.000001 se neeche nahi jaayega
        ),
    ]

    history = model.fit(
        X_train, y_train_flat,
        validation_data=(X_test, y_test_flat),
        epochs=epochs,          # Max 100 epochs
        batch_size=batch_size,  # 64 samples per gradient update
        callbacks=callbacks,
        verbose=1,              # Training progress dikhao
    )

    # Save trained model
    model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
    model.save(model_path)

    # Evaluate on test set
    loss, mae = model.evaluate(X_test, y_test_flat, verbose=0)
    return model, history, loss, mae
```

**Callbacks Kya Karte Hain:**

**EarlyStopping:**
- Problem: 100 epochs diye hain, par 40th epoch pe model best hoga, uske baad overfit karega
- Solution: Agar 10 consecutive epochs mein val_loss improve na ho → stop
- `restore_best_weights=True` → best epoch ke weights wapas laao (na ki last epoch ke)

**ReduceLROnPlateau:**
- Problem: High learning rate se model loss ke around oscillate karta hai
- Solution: Agar 5 epochs mein improvement nahi → learning rate half karo
- Isse fine-grained learning hota hai as we approach minimum
- `0.001 → 0.0005 → 0.00025 → ... → min 0.000001`

### Lines 75-107: `train_all()` — All Locations Train
```python
def train_all(force=False, epochs=100, batch_size=64):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Auto-detect locations from processed files
    locations = set()
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith("_X_train.npy"):
            loc_tag = f.replace("_X_train.npy", "")    # "anand_vihar_X_train.npy" → "anand_vihar"
            locations.add(loc_tag)

    metrics = {}
    for loc_tag in sorted(locations):
        model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
        if os.path.exists(model_path) and not force:
            print(f"Skipping {loc_tag}, model already exists.")
            continue    # Skip if model exists AND force=False

        print(f"\n{'='*60}")
        print(f"Training: {loc_tag}")
        print(f"{'='*60}")
        _, _, loss, mae = train_location(loc_tag, epochs=epochs, batch_size=batch_size)
        metrics[loc_tag] = {
            "test_mse_scaled": round(float(loss), 6),
            "test_mae_scaled": round(float(mae), 6),
            "epochs_requested": epochs,
            "batch_size": batch_size,
        }

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "lstm_metrics.json")
    if metrics:
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
```

**`--force` Flag Ka Purpose:**
- Default: Agar `anand_vihar_lstm.h5` already hai → skip
- `--force`: Purana model ignore karo, naya train karo
- Jab data update ho ya architecture change ho, tab `--force` use karo

---

## 10. LSTM Internals — Kaise Kaam Karta Hai

### Simple Explanation

LSTM ek **memory-based neural network** hai. Normal neural network ko sirf current input pata hota hai. LSTM ko **previous inputs yaad rehte hain**.

### LSTM Cell Ka Kaam (Simplified)

Har timestep pe LSTM cell mein 3 gates hote hain:

```
┌─────────────────────────────────────────┐
│              LSTM Cell                   │
│                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────┐ │
│  │ Forget   │   │ Input    │   │Output│ │
│  │ Gate     │   │ Gate     │   │ Gate │ │
│  │ "Kya     │   │ "Kya naya│   │"Kya  │ │
│  │  bhulna  │   │  yaad    │   │batana│ │
│  │  hai?"   │   │  rakhna  │   │hai?" │ │
│  │          │   │  hai?"   │   │      │ │
│  └──────────┘   └──────────┘   └──────┘ │
│                                          │
│  Cell State (Long-term Memory) ────────→ │
│  Hidden State (Short-term Output) ─────→ │
└─────────────────────────────────────────┘
```

**Forget Gate**: "Kya purani memory se delete karna hai?"
- Example: 10 din pehle ka temperature pattern shayad relevant nahi → forget

**Input Gate**: "Kya naya information memory mein add karna hai?"
- Example: Aaj PM2.5 suddenly spike hua → remember this!

**Output Gate**: "Memory se kya information bahar bhejna hai?"
- Example: Recent pollution trend + current weather → output hidden state

### Hamara Model Mein

**Layer 1: LSTM(64, return_sequences=True)**
- 336 timesteps process karta hai, har timestep pe 64-dimensional hidden state output karta hai
- `return_sequences=True` → sabke 336 hidden states output karo (next LSTM ke liye)
- Output: (336, 64)

**Layer 2: LSTM(32, return_sequences=False)**
- Pehle LSTM ke 336 outputs mein se patterns dhundhta hai
- `return_sequences=False` → sirf last timestep ka 32-dim hidden state output karo
- Yeh 32-dim vector poore 14 din ka **compressed summary** hai
- Output: (32,)

**Intuition**: Pehla LSTM "local patterns" samjhta hai (daily cycles, weather changes), doosra LSTM "global patterns" samjhta hai (overall trend, multi-day patterns)

### Kyon LSTM, Kyon Na Simple RNN?

Simple RNN mein **vanishing gradient** problem hoti hai:
- Agar sequence 336 steps lambi hai, toh pehle ke steps ka gradient training mein bahut chhota ho jaata hai
- Model bhool jaata hai ki pehle din kya hua

LSTM ka **cell state** ek highway hai jisse gradient freely flow kar sakta hai → long-term dependencies learn ho jaati hain

---

## 11. FILE: `src/aqi_formula.py` — Line by Line

### Lines 3-36: Breakpoint Tables (CPCB Standard)
```python
BREAKPOINTS = {
    "pm2_5": [
        #  C_low, C_high, I_low, I_high
        (0, 30, 0, 50),        # 0-30 µg/m³ → AQI 0-50 (Good)
        (31, 60, 51, 100),     # 31-60 → AQI 51-100 (Satisfactory)
        (61, 90, 101, 200),    # 61-90 → AQI 101-200 (Moderate)
        (91, 120, 201, 300),   # 91-120 → AQI 201-300 (Poor)
        (121, 250, 301, 400),  # 121-250 → AQI 301-400 (Very Poor)
        (250, 500, 401, 500),  # 250-500 → AQI 401-500 (Severe)
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (400, 800, 401, 500),
    ],
    "co": [
        (0, 1000, 0, 50),           # Note: units µg/m³
        (1001, 2000, 51, 100),
        (2001, 10000, 101, 200),
        (10001, 17000, 201, 300),
        (17001, 34000, 301, 400),
        (34000, 50000, 401, 500),
    ],
}
```
**Intuition**: Yeh CPCB (Central Pollution Control Board) ka official AQI calculation standard hai. Har pollutant ke liye concentration ranges define hain aur unke corresponding AQI index ranges.

### Lines 38-45: AQI Categories
```python
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]
```

### Lines 48-59: `calc_sub_index()` — Linear Interpolation
```python
def calc_sub_index(pollutant, concentration):
    if pollutant not in BREAKPOINTS:
        return 0

    for bp_lo, bp_hi, idx_lo, idx_hi in BREAKPOINTS[pollutant]:
        if bp_lo <= concentration <= bp_hi:
            # Linear interpolation formula
            sub_index = ((idx_hi - idx_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + idx_lo
            return round(sub_index)

    # Agar concentration table se bhi zyada hai
    if concentration > BREAKPOINTS[pollutant][-1][1]:
        return 500    # Cap at 500
    return 0
```

**Example: PM2.5 = 75 µg/m³**
```
Breakpoint: (61, 90, 101, 200)
sub_index = ((200 - 101) / (90 - 61)) × (75 - 61) + 101
          = (99 / 29) × 14 + 101
          = 3.414 × 14 + 101
          = 47.8 + 101
          = 149 (rounded)
```
So PM2.5 = 75 µg/m³ → Sub-Index = 149 (Moderate)

### Lines 62-71: `calculate_aqi()` — Final AQI
```python
def calculate_aqi(pm2_5, pm10, co, no2):
    sub_indices = {
        "pm2_5": calc_sub_index("pm2_5", pm2_5),
        "pm10": calc_sub_index("pm10", pm10),
        "co": calc_sub_index("co", co),
        "no2": calc_sub_index("no2", no2),
    }
    aqi = max(sub_indices.values())                  # Maximum sub-index = Final AQI
    dominant = max(sub_indices, key=sub_indices.get)  # Which pollutant has highest
    return aqi, dominant, sub_indices
```

**Example:**
```
PM2.5 = 75 → sub = 149
PM10 = 120 → sub = 108
CO = 2500 → sub = 125
NO2 = 50 → sub = 62

AQI = max(149, 108, 125, 62) = 149
Dominant = PM2.5
Category = Moderate
```

### Lines 74-90: Category & Health Advisory
```python
def get_category(aqi):
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Severe" if aqi > 500 else "Good"

def get_health_advisory(category):
    advisories = {
        "Good": "Minimal impact. Enjoy outdoor activities.",
        "Satisfactory": "Minor breathing discomfort for sensitive people.",
        "Moderate": "Breathing discomfort for people with lung/heart disease.",
        "Poor": "Breathing discomfort on prolonged exposure. Avoid outdoor exertion.",
        "Very Poor": "Respiratory illness on prolonged exposure. Limit outdoor activity.",
        "Severe": "Serious health impacts. Avoid all outdoor activity.",
    }
    return advisories.get(category, "")
```

---

## 12. FILE: `app.py` — Streamlit App Line by Line

### Lines 1-17: Imports & Page Config
```python
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
    layout="wide",                    # Full-width layout
    initial_sidebar_state="expanded",  # Sidebar open by default
)
```
**Important**: `from src.prepare_lstm_data import ...` — same constants (POLLUTANTS, WINDOW_SIZE etc.) import hote hain taki training aur inference mein **exact same values** use hon. Agar training mein WINDOW_SIZE=336 tha aur inference mein galti se 300 use ho jaaye → predictions galat aayengi.

### Lines 19-38: Paths, Locations, Colors
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

LOCATIONS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.316},
    # ... same 6 locations as everywhere
}

AQI_COLORS = {
    "Good": "#10B981",        # Green
    "Satisfactory": "#34D399", # Light green
    "Moderate": "#FBBF24",     # Yellow
    "Poor": "#F97316",         # Orange
    "Very Poor": "#EF4444",    # Red
    "Severe": "#7F1D1D",       # Dark red
}
```

### Lines 40-62: Custom CSS
```python
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .forecast-card {
        background: rgba(30, 41, 59, 0.85);   /* Dark semi-transparent */
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);            /* Glassmorphism effect */
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    /* ... more styles for aqi-big, cat-label, etc. */
</style>
""", unsafe_allow_html=True)
```
Custom HTML/CSS se forecast cards premium look dete hain. `unsafe_allow_html=True` Streamlit ko bolta hai ki raw HTML render karo.

### Lines 65-81: Model & Scaler Loading (Cached)
```python
@st.cache_resource        # Model object RAM mein cache karo — reload mat karo har baar
def load_forecast_model(location):
    loc_tag = location.lower().replace(" ", "_")     # "Anand Vihar" → "anand_vihar"
    model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path, compile=False)
    # compile=False → model ko training mode mein nahi laana, sirf prediction ke liye load karo

@st.cache_data(ttl=3600)  # Cache for 1 hour, fir re-load
def load_scalers(location):
    loc_tag = location.lower().replace(" ", "_")
    f_path = os.path.join(MODELS_DIR, f"{loc_tag}_feature_scaler.pkl")
    t_path = os.path.join(MODELS_DIR, f"{loc_tag}_target_scaler.pkl")
    if not os.path.exists(f_path) or not os.path.exists(t_path):
        return None, None
    return joblib.load(f_path), joblib.load(t_path)
```
**`@st.cache_resource`**: Heavy objects (model) ko ek baar load karo, fir cache mein rakho. Agar user dobara forecast run kare same location ke liye, model wapas load nahi hoga.

**`@st.cache_data(ttl=3600)`**: Data objects 1 hour ke liye cache. TTL = Time To Live.

### Lines 84-157: `fetch_live_data()` — Live API Data Fetch

```python
@st.cache_data(ttl=1800)  # 30 min cache
def fetch_live_data(location, days_back=21):
    coords = LOCATIONS[location]
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
```

**Kyon 21 din ka data fetch karta hai jabki model ko sirf 14 din (336 hours) chahiye?**
- 21 din = safety margin
- Kuch hours mein data missing ho sakta hai → interpolation ke baad bhi 336+ hours mil jaayein
- Backtesting ke liye bhi extra data chahiye (past 7 days compare ke liye)

**3 API Calls:**
```python
    # API Call 1: Historical Weather (start_date → yesterday)
    w_archive = requests.get("https://archive-api.open-meteo.com/v1/archive", params={...})

    # API Call 2: Today's Weather (today only)
    w_forecast = requests.get("https://api.open-meteo.com/v1/forecast", params={...})

    # API Call 3: Air Quality (start_date → today)
    a_resp = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={...})
```

**Kyon Weather ke liye 2 calls?**
- Archive API mein aaj ka data nahi hota (yesterday tak available)
- Forecast API mein aaj ka real-time data milta hai
- Dono combine karke complete weather timeline banti hai

```python
    # Merge weather timeseries
    w_times = w_archive["hourly"]["time"] + w_forecast["hourly"]["time"]
    w_temp = w_archive["hourly"]["temperature_2m"] + w_forecast["hourly"]["temperature_2m"]
    # ... similarly for humidity, pressure, wind

    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(w_times),
        "temp_c": w_temp, "humidity": w_hum,
        "pressure_mb": w_pres, "windspeed_kph": w_wind,
    })

    # Air quality DataFrame
    aq_df = pd.DataFrame({
        "datetime": pd.to_datetime(a_resp["hourly"]["time"]),
        "pm2_5": ..., "pm10": ..., "co": ..., "no2": ...,
    })

    # Inner join on datetime — sirf wahi rows jahan dono available hain
    df = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    df["location"] = location
    df[FEATURES].interpolate(method="linear")   # Fill missing values
    df.dropna(subset=FEATURES, inplace=True)    # Drop unfillable rows
    return df
```

### Lines 160-178: `prepare_live_sequence()` — Live Data → Model Input

```python
def prepare_live_sequence(df, feature_scaler):
    df_copy = df.copy()
    df_copy = add_time_features(df_copy)       # hour_sin/cos, month_sin/cos, is_weekend
    df_copy = add_diwali_feature(df_copy)       # days_to_diwali

    feature_cols = FEATURES + [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_weekend", "days_to_diwali",
    ]
    # SAME 14 features as training — ORDER MATTERS!

    df_copy[feature_cols] = feature_scaler.transform(df_copy[feature_cols])
    # IMPORTANT: .transform(), NOT .fit_transform()
    # Training ke scaler se transform karo — naya fit mat karo!

    values = df_copy[feature_cols].values

    if len(values) < WINDOW_SIZE:
        st.error(f"Not enough data. Need {WINDOW_SIZE} hours, got {len(values)}.")
        return None

    seq = values[-WINDOW_SIZE:]                 # Last 336 hours
    return seq.reshape(1, WINDOW_SIZE, len(feature_cols))  # (1, 336, 14)
```

**Critical Point: `transform()` vs `fit_transform()`**
- `fit_transform()` → naya min/max calculate karega live data pe → GALAT! Training ke ranges se match nahi karega
- `transform()` → training ke saved min/max use karega → SAHI! Model ko same scale ka data milega

### Lines 181-186: `run_forecast()` — Model Prediction
```python
def run_forecast(model, sequence, target_scaler):
    pred_scaled = model.predict(sequence, verbose=0)
    # Input: (1, 336, 14) → Output: (1, 288)

    pred_reshaped = pred_scaled.reshape(FORECAST_HOURS, len(POLLUTANTS))
    # (1, 288) → (72, 4) — 72 hours × 4 pollutants

    pred_actual = target_scaler.inverse_transform(pred_reshaped)
    # Scaled values (0-1) → actual µg/m³ values
    # Example: 0.15 → 75 µg/m³ for PM2.5

    pred_actual = np.clip(pred_actual, 0, None)
    # Negative predictions ko 0 karo — pollutant negative nahi ho sakta
    return pred_actual
```

### Lines 189-194: `calibrate_forecast()` — Bias Correction
```python
def calibrate_forecast(predictions, live_df):
    """Reduce short-term distribution shift using latest observed pollutants."""
    recent = live_df[POLLUTANTS].tail(24).median().to_numpy(dtype=float)
    # Last 24 hours ka median pollutant level
    # Example: [pm2_5=85, pm10=150, co=3000, no2=55]

    if not np.isfinite(recent).all():
        return predictions    # Agar NaN/Inf hai toh calibrate mat karo

    return np.clip(0.75 * predictions + 0.25 * recent, 0, None)
    # 75% model prediction + 25% recent actual
    # Isse predictions current conditions ke thoda paas aati hain
```

**Kyon Calibration?**
- Model 2025-2026 data pe train hua, par aaj ka pollution pattern training se different ho sakta hai
- Example: Model predict kare PM2.5 = 60, par last 24 hours ka median 100 hai
- Calibrated: 0.75×60 + 0.25×100 = 45 + 25 = 70 → more realistic
- Yeh ek simple **online bias correction** technique hai

### Lines 197-233: `run_backtesting()` — Past 7 Days Accuracy Check

**Yeh section BAHUT IMPORTANT hai — tera specific question tha.**

```python
def run_backtesting(df, model, feature_scaler, target_scaler, days_back=7):
    results = []
    total_hours = len(df)    # e.g., 504 hours (21 days)

    for day_offset in range(days_back, 0, -1):    # 7, 6, 5, 4, 3, 2, 1
        end_idx = total_hours - (day_offset * 24)
        # day_offset=7: end_idx = 504 - 168 = 336 → 14 days of data
        # day_offset=1: end_idx = 504 - 24 = 480 → 20 days of data

        if end_idx < WINDOW_SIZE:
            continue    # Skip if not enough data for input window

        # STEP 1: Pretend ki hum "us din" pe khade hain
        slice_df = df.iloc[:end_idx].copy()
        # Sirf us din tak ka data use karo — future data nahi!

        # STEP 2: Us din tak ke data se model predict karo
        seq = prepare_live_sequence(slice_df, feature_scaler)
        if seq is None:
            continue
        pred = run_forecast(model, seq, target_scaler)

        # STEP 3: Day 1 ka average prediction nikalo
        pred_day1_avg = pred[:24].mean(axis=0)   # First 24 hours ka mean
        pred_aqi, _, _ = calculate_aqi(
            pred_day1_avg[0], pred_day1_avg[1], pred_day1_avg[2], pred_day1_avg[3]
        )

        # STEP 4: Actual data nikalo (next 24 hours)
        actual_start = end_idx
        actual_end = min(end_idx + 24, total_hours)
        actual_slice = df.iloc[actual_start:actual_end]
        actual_avg = actual_slice[POLLUTANTS].mean()
        actual_aqi, _, _ = calculate_aqi(
            actual_avg["pm2_5"], actual_avg["pm10"], actual_avg["co"], actual_avg["no2"]
        )

        # STEP 5: Compare
        target_date = df.iloc[actual_start]["datetime"]
        results.append({
            "Date": target_date.strftime("%b %d"),
            "Day": target_date.strftime("%A"),
            "Predicted AQI": pred_aqi,
            "Actual AQI": actual_aqi,
            "Difference": abs(pred_aqi - actual_aqi),
        })

    return pd.DataFrame(results)
```

---

## 13. Forecast Button Workflow — Step by Step

Jab user "🚀 Run Forecast" dabata hai, exact yeh hota hai:

```
Step 1: load_forecast_model("Anand Vihar")
        → Load models/anand_vihar_lstm.h5 (cached)
        
Step 2: load_scalers("Anand Vihar")
        → Load anand_vihar_feature_scaler.pkl (14 features)
        → Load anand_vihar_target_scaler.pkl (4 pollutants)

Step 3: fetch_live_data("Anand Vihar", days_back=21)
        → 3 API calls to Open-Meteo
        → Weather archive (21 days → yesterday)
        → Weather forecast (today)
        → Air quality (21 days → today)
        → Merge → ~504 rows DataFrame
        → Interpolate missing → dropna

Step 4: prepare_live_sequence(live_df, feature_scaler)
        → add_time_features() → 5 new columns
        → add_diwali_feature() → 1 new column
        → feature_scaler.transform() → 14 features scaled to 0-1
        → Extract last 336 rows → reshape to (1, 336, 14)

Step 5: model.predict(sequence)
        → Forward pass through LSTM
        → Output: (1, 288) scaled values

Step 6: Reshape (288,) → (72, 4)
        → target_scaler.inverse_transform()
        → Actual pollutant values µg/m³
        → clip negatives to 0

Step 7: calibrate_forecast()
        → 0.75 × prediction + 0.25 × recent_24h_median

Step 8: Display
        → 3-day cards with AQI, category, color, advisory
        → Hourly breakdown tables
        → Pollutant trend line charts
        → 7-day backtesting comparison
```

---

## 14. Backtesting — Kya Hota Hai Jab Past 7 Days Compare Karte Hai

### Tera Question: "Kya hum sach mein predict karte hai ussi time pe?"

**Haan! Exactly yahi hota hai.**

Backtesting ka concept:
1. Pretend karo ki hum 7 din pehle khade hain
2. Us waqt tak ka data use karke predict karo
3. Ab actual data available hai (kyunki woh din guzar chuka) — compare karo

**Visual Example:**
```
Today = Sep 4

Backtest Day 1 (Aug 28):
├── Use data: Aug 7 → Aug 27 (14 days window)
├── Model predicts: Aug 28 ka AQI
├── Actual Aug 28 ka data available hai API mein → compare!
└── Difference = |Predicted - Actual|

Backtest Day 2 (Aug 29):
├── Use data: Aug 8 → Aug 28
├── Model predicts: Aug 29 ka AQI
├── Compare with actual Aug 29
└── ...

... same for 7 days
```

### "Actual AQI API se aaya hai na?"

**Haan!** `actual_slice = df.iloc[actual_start:actual_end]` — yeh wahi live DataFrame ka portion hai jo Open-Meteo API se fetch hua tha. Yani:
- **Predicted AQI** = model ne us time pe predict kiya (simulate karke)
- **Actual AQI** = Open-Meteo API ka data (not CPCB, not any other source)

**Important Caveat**: "Actual" yahan Open-Meteo ka estimated data hai, CPCB station ka actual observed data nahi. Isliye backtesting se pata chalta hai ki model **Open-Meteo ke data ke against** kitna accurate hai, par **real-world CPCB AQI ke against** accuracy alag ho sakti hai.

---

## 15. Calibration — Distribution Shift Fix

### Problem
Model 2025-2026 training data pe train hua. Par aaj ka pollution distribution training se different ho sakta hai (seasonal change, new construction, policy changes).

### Solution (Simple)
```python
final = 0.75 × model_prediction + 0.25 × recent_24h_median
```

### Kyon 0.75 / 0.25?
- 0.75 model ko zyada weightage deta hai (model ne pattern learn kiya hai)
- 0.25 recent data ko thoda influence deta hai (current conditions ka correction)
- Yeh ratio heuristic hai — experimentally tune kiya ja sakta hai

### Limitation
- Yeh CPCB mismatch ka permanent fix nahi hai
- Sirf short-term bias reduce karta hai
- Ideal approach: proper post-hoc calibration ya ensemble methods

---

## 16. Metrics — MAE/MSE Itna Kam Kyon

### Current Metrics
| Location | Test MSE | Test MAE |
|---|---|---|
| Anand Vihar | 0.009005 | 0.066251 |
| Connaught Place | 0.008894 | 0.062659 |
| Dwarka | 0.009011 | 0.059861 |
| IGI Airport | 0.009034 | 0.059678 |
| Okhla Phase III | 0.009485 | 0.062593 |
| Rohini | 0.009245 | 0.065023 |

### Kyon Itna Kam?

**Tera intuition SAHI hai — yeh scaled metrics hain!**

- Model MinMaxScaler se scaled data (0-1 range) pe train hua hai
- MSE = 0.009 ka matlab: average squared error 0.009 in 0-1 range
- MAE = 0.066 ka matlab: average 6.6% error in scaled space

### Real Values Mein Kitna Error Hoga?

Rough calculation:
```
PM2.5 range: 0-500 µg/m³
MAE in real values ≈ 0.066 × 500 = 33 µg/m³

PM10 range: 0-600 µg/m³
MAE in real values ≈ 0.066 × 600 = 40 µg/m³

CO range: 0-50000 µg/m³
MAE in real values ≈ 0.066 × 50000 = 3300 µg/m³

NO2 range: 0-800 µg/m³
MAE in real values ≈ 0.066 × 800 = 53 µg/m³
```

**Lekin yeh worst-case estimate hai.** Actual data mein values full range nahi cover karte (PM2.5 rarely 500 jaata hai), toh real MAE chhota hoga. Backtesting section mein actual AQI difference dikhta hai — woh zyada meaningful metric hai.

### Better Evaluation Kaise Karein?
```python
# Ye karo training ke baad:
pred_actual = target_scaler.inverse_transform(pred_scaled)
actual = target_scaler.inverse_transform(y_test)
mae_pm25 = np.mean(np.abs(pred_actual[:,:,0] - actual[:,:,0]))
print(f"PM2.5 MAE: {mae_pm25:.1f} µg/m³")
```

---

## 17. Data Source — Open-Meteo vs CPCB

### Open-Meteo (Current Source)
- **Free API**, no key needed
- **Gridded estimates** — satellite + weather model based
- Not from actual station sensors
- Hourly data, globally available
- Good for development/prototyping

### CPCB (Ideal Source)
- **Government ground-truth** — actual sensor readings from monitoring stations
- More accurate for specific locations
- API available: `api.cpcb.gov.in` (registration needed)
- Real-time + historical data
- What news/apps (IQAir, AQI.in) actually use

### Kya Fark Padta Hai?
```
Open-Meteo PM2.5 at Anand Vihar coordinate:  85 µg/m³
CPCB Station Anand Vihar actual reading:      120 µg/m³
Difference: 35 µg/m³ → AQI difference: ~40-50 points
```

**Yahi wajah hai ki app ka AQI aur kisi website ka AQI different aata hai!**

### Future Improvement
- Training data: CPCB station observations
- Weather inputs: Open-Meteo (fine, weather data is good)
- Comparison: CPCB station readings
- This will make the model genuinely useful

---

## 18. Limitations Aur Honest Assessment

| # | Limitation | Impact | Fix |
|---|---|---|---|
| 1 | Open-Meteo ≠ CPCB ground truth | AQI mismatch | CPCB API data use karo |
| 2 | Scaler fit on full data (leakage) | Slight optimism in metrics | Train-only scaler fit |
| 3 | No confidence intervals | False precision | Quantile loss / MC Dropout |
| 4 | 72h direct output | Day 3 less accurate than Day 1 | Recursive/autoregressive approach |
| 5 | No SO₂, O₃ pollutants | Incomplete AQI | Add more pollutant channels |
| 6 | No wind direction | Missing dispersion info | Add from API |
| 7 | Single architecture | No baseline comparison | Compare XGBoost, ARIMA, etc. |
| 8 | No model versioning | Can't track improvements | MLflow / DVC |
| 9 | API dependency | Offline failure | Local data cache/fallback |
| 10 | No automated retraining | Model gets stale | Scheduled retrain pipeline |

---

## 19. Future Improvements — Kya Add/Improve Kar Sakte Ho

### Priority 1: Data Quality
- [ ] CPCB API integration for ground-truth pollution data
- [ ] Add SO₂ and O₃ to make AQI calculation complete (6 pollutants)
- [ ] Wind direction, rainfall, boundary layer height as features
- [ ] Satellite fire/crop burning data (FIRMS/NASA)

### Priority 2: Model Improvements
- [ ] Train-only scaler fitting (no test leakage)
- [ ] Baseline comparison (XGBoost, ARIMA, Persistence)
- [ ] Day-wise metrics (Day 1 MAE vs Day 2 MAE vs Day 3 MAE)
- [ ] Attention mechanism (Transformer or LSTM + Attention)
- [ ] Probabilistic output (quantile regression → confidence range)

### Priority 3: Engineering
- [ ] ModelCheckpoint callback (save best during training)
- [ ] MLflow for experiment tracking
- [ ] Automated weekly retraining
- [ ] Docker deployment
- [ ] Unit tests for AQI formula

### Priority 4: User Experience
- [ ] Show confidence range: "AQI: 96 (range: 80-118)"
- [ ] Add more locations (Mumbai, Bangalore)
- [ ] Push notifications for severe AQI
- [ ] Comparison with CPCB official AQI

---

## 20. Interview Questions Deep Dive

### Q: "LSTM kyon use kiya, XGBoost ya ARIMA kyon nahi?"
**A:** AQI hourly time-series hai jismein sequential dependencies hain — kal ka pollution aaj ko affect karta hai. LSTM specifically sequential data ke liye designed hai:
- ARIMA univariate hai, humara problem multivariate hai (14 features)
- XGBoost tabular data ke liye achha hai par temporal memory nahi hai — har row independent treat hota hai
- LSTM ki memory cells long-term patterns capture karti hain (14-day patterns)
- **Honest addition**: XGBoost ko bhi compare karna chahiye as baseline — kabhi kabhi simpler models surprisingly achhe hote hain

### Q: "336 aur 72 kahan se aaye?"
**A:** 336 = 14 days × 24 hours. 14 din choose kiye kyunki:
- Weekly cycles capture karne ke liye minimum 2 weeks chahiye (weekday vs weekend)
- Weather patterns 7-14 din ke cycles mein change hote hain
- Too short (3 days) → insufficient context
- Too long (30 days) → model slow + old data irrelevant
72 = 3 days × 24 hours — practical forecast horizon jisme reasonable accuracy milti hai

### Q: "Model perfect hoga?"
**A:** Nahi. AQI forecasting inherently uncertain hai kyunki:
- Extreme events (sudden fire, industrial accident) unpredictable hain
- Weather itself uncertain hai — weather forecast bhi 100% accurate nahi hoti
- Atmospheric chemistry complex hai
- Goal: reliable range prediction, not exact number
- Expected output: "AQI: 96 (likely range: 80-118, Confidence: Medium)"

### Q: "Agar Maine ye project present kiya interview mein, kya challenges face kiye?"
**A:** 
1. **Data format mismatch**: 2025 aur 2026 CSV ka time format alag tha — defensive datetime parsing
2. **Windows compatibility**: `strftime("%-H")` Windows pe fail hota hai — `.hour.astype(str)` fix
3. **Model retraining**: Purane models automatically skip ho rahe the — `--force` flag add kiya
4. **Scaler confusion**: Initially ek scaler tha dono ke liye — separate feature/target scalers banaye
5. **API limitation**: Open-Meteo archive API mein aaj ka data nahi hota — forecast API se today ka data alag fetch karna pada

### Q: "Production mein deploy karoge toh kya challenges aayenge?"
**A:**
1. **API rate limits**: Open-Meteo free hai par production traffic pe throttle ho sakta hai
2. **Model staleness**: Har 2-3 months mein retrain karna padega new data pe
3. **Monitoring**: Model accuracy track karna padega — agar predictions consistently off hain toh alert
4. **Scaling**: 6 models × har request pe prediction → compute heavy → need caching/batching
5. **Ground truth**: CPCB data integration essential hai production-level accuracy ke liye
