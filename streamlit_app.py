"""
India AQI Dashboard — Dual Pipeline (Regressor + Classifier)
=============================================================
Uses pre-trained models from models/ folder:
  - xgb_regressor.pkl  → Predicts AQI value
  - rf_classifier.pkl  → Predicts AQI bucket (Good/Severe/etc.)

Run: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="India AQI Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD MODELS & ARTIFACTS (cached — loads once)
# ============================================================
@st.cache_resource
def load_models():
    """Load all pre-trained models and preprocessing artifacts."""
    xgb_reg = joblib.load("models/xgb_regressor.pkl")
    rf_clf = joblib.load("models/rf_classifier.pkl")
    scaler = joblib.load("models/minmax_scaler.pkl")
    season_encoder = joblib.load("models/season_encoder.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    city_list = joblib.load("models/city_list.pkl")
    bucket_reverse = joblib.load("models/bucket_reverse.pkl")

    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)

    return xgb_reg, rf_clf, scaler, season_encoder, feature_names, city_list, bucket_reverse, metrics


@st.cache_data
def load_data():
    """Load and prepare dataset for EDA charts."""
    df = pd.read_csv("data/city_day.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    def get_season(m):
        if m <= 2 or m == 12: return "Winter"
        elif 3 <= m <= 5: return "Spring"
        elif 6 <= m <= 9: return "Monsoon"
        else: return "Autumn"

    df["Season"] = df["Month"].apply(get_season)
    df = df.drop(columns=["Xylene", "Toluene", "NH3"], errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["Year", "Month"]]
    for col in num_cols:
        df[col] = df.groupby(["Season", "City"])[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    df["AQI_Bucket"] = df.groupby(["Season", "City"])["AQI_Bucket"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Moderate")
    )
    df["AQI_Bucket"] = df["AQI_Bucket"].fillna("Moderate")
    return df


# ============================================================
# CUSTOM STYLING (works in both light & dark mode)
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #a0c4ff; margin: 0.3rem 0 0 0; font-size: 1rem; }

    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .result-card h2 { margin: 0; font-size: 1.5rem; }
    .result-card h3 { margin: 0.3rem 0 0 0; font-size: 1rem; opacity: 0.8; }

    /* Make metric cards theme-aware */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 12px 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🏭 India AQI — Dual Prediction Pipeline</h1>
    <p>XGBoost Regressor (AQI Value) + Random Forest Classifier (AQI Category) | 26 Cities | 2015–2020</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD EVERYTHING
# ============================================================
try:
    xgb_reg, rf_clf, scaler, season_encoder, feature_names, city_list, bucket_reverse, metrics = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.info("Run `python train_and_save_models.py` first to generate model files.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 AQI Predictor", "📊 Dashboard", "🏙️ City Analysis", "🦠 COVID Impact"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Regressor R²", f"{metrics['xgb_regressor']['r2']:.4f}")
st.sidebar.metric("Classifier Accuracy", f"{metrics['rf_classifier']['accuracy']:.2%}")
st.sidebar.metric("Features Used", metrics['feature_count'])
st.sidebar.markdown(f"Training: {metrics['training_samples']:,} samples")


# ============================================================
# HELPER: Build feature vector for prediction
# ============================================================
def build_feature_vector(pollutants, city, season, year, feature_names, city_list, season_encoder, scaler):
    """
    Build the 38-feature input vector matching model training.
    Returns a DataFrame with correct column names.
    """
    row = {}

    # 1) Raw pollutant values
    for col in ["PM2.5", "PM10", "NO", "NO2", "NOx", "CO", "SO2", "O3", "Benzene"]:
        row[col] = pollutants.get(col, 0.0)

    # 2) Year
    row["Year"] = year

    # 3) Composite Score (MinMax scaled then weighted)
    weight_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    weights = {"PM2.5": 0.35, "PM10": 0.20, "NO2": 0.15, "CO": 0.10, "SO2": 0.10, "O3": 0.10}
    raw_vals = np.array([[row[c] for c in weight_cols]])
    scaled_vals = scaler.transform(raw_vals)
    composite = sum(scaled_vals[0][i] * weights[c] for i, c in enumerate(weight_cols))
    row["Composite_Score"] = composite

    # 4) Season encoded
    row["Season_Encoded"] = season_encoder.transform([season])[0]

    # 5) City one-hot
    for c in city_list:
        row[f"City_{c}"] = 1.0 if c == city else 0.0

    # Build DataFrame with exact feature order
    input_df = pd.DataFrame([row])[feature_names]
    return input_df


# ============================================================
# BUCKET STYLING
# ============================================================
BUCKET_COLORS = {
    "Good": "#2ecc71",
    "Satisfactory": "#27ae60",
    "Moderate": "#f39c12",
    "Poor": "#e74c3c",
    "Very Poor": "#8e44ad",
    "Severe": "#c0392b",
}

HEALTH_ADVISORY = {
    "Good": "Air quality is excellent. Enjoy outdoor activities!",
    "Satisfactory": "Air quality is acceptable. Sensitive people should limit prolonged outdoor exertion.",
    "Moderate": "May cause breathing discomfort for people with lung disease. Limit prolonged outdoor exertion.",
    "Poor": "May cause breathing discomfort for many. Avoid prolonged outdoor activities.",
    "Very Poor": "Health alert! Everyone may experience health effects. Avoid all outdoor activities.",
    "Severe": "EMERGENCY! Serious health effects for everyone. Stay indoors with air purifier.",
}


# ============================================================
# PAGE 1: AQI PREDICTOR (MAIN PAGE)
# ============================================================
if page == "🔮 AQI Predictor":

    st.subheader("🔮 Dual Pipeline AQI Prediction")
    st.markdown("Enter pollutant levels → **XGBoost** predicts AQI value → **Random Forest** predicts AQI category")

    st.divider()

    # --- Input Form ---
    with st.form("prediction_form"):
        st.markdown("#### 🧪 Pollutant Concentrations")

        col1, col2, col3 = st.columns(3)

        with col1:
            pm25 = st.number_input("PM2.5 (ug/m3)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
            pm10 = st.number_input("PM10 (ug/m3)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
            no = st.number_input("NO (ug/m3)", min_value=0.0, max_value=500.0, value=15.0, step=0.5)

        with col2:
            no2 = st.number_input("NO2 (ug/m3)", min_value=0.0, max_value=500.0, value=30.0, step=0.5)
            nox = st.number_input("NOx (ppb)", min_value=0.0, max_value=500.0, value=25.0, step=0.5)
            co = st.number_input("CO (mg/m3)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)

        with col3:
            so2 = st.number_input("SO2 (ug/m3)", min_value=0.0, max_value=200.0, value=12.0, step=0.5)
            o3 = st.number_input("O3 (ug/m3)", min_value=0.0, max_value=500.0, value=35.0, step=0.5)
            benzene = st.number_input("Benzene (ug/m3)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)

        st.markdown("#### 🌍 Location & Time")
        loc1, loc2, loc3 = st.columns(3)
        with loc1:
            city = st.selectbox("City", city_list, index=city_list.index("Delhi"))
        with loc2:
            season = st.selectbox("Season", ["Winter", "Spring", "Monsoon", "Autumn"])
        with loc3:
            year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)

        submitted = st.form_submit_button("🔮 Predict AQI", use_container_width=True)

    # --- Prediction ---
    if submitted:
        pollutants = {
            "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2,
            "NOx": nox, "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene,
        }

        input_df = build_feature_vector(
            pollutants, city, season, year,
            feature_names, city_list, season_encoder, scaler,
        )

        # DUAL PREDICTION
        aqi_value = xgb_reg.predict(input_df)[0]
        bucket_code = rf_clf.predict(input_df)[0]
        bucket_name = bucket_reverse[bucket_code]
        bucket_proba = rf_clf.predict_proba(input_df)[0]

        color = BUCKET_COLORS.get(bucket_name, "#333")
        advisory = HEALTH_ADVISORY.get(bucket_name, "")

        st.divider()

        # --- Results ---
        st.markdown("### 📋 Prediction Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("🎯 Predicted AQI", f"{aqi_value:.0f}")
        r2.metric("📊 Category (Classifier)", bucket_name)
        r3.metric("🏙️ City", city)

        # Color banner
        st.markdown(f"""
        <div class="result-card" style="background:{color};">
            <h2 style="color:white;">AQI: {aqi_value:.0f} — {bucket_name}</h2>
            <h3 style="color:rgba(255,255,255,0.85);">{advisory}</h3>
        </div>
        """, unsafe_allow_html=True)

        # --- Classifier confidence ---
        st.markdown("#### 🎯 Classifier Confidence (RF Probabilities)")
        proba_df = pd.DataFrame({
            "Category": [bucket_reverse[i] for i in range(len(bucket_proba))],
            "Probability": bucket_proba,
        })
        proba_df["Probability"] = (proba_df["Probability"] * 100).round(1)

        fig = px.bar(
            proba_df, x="Category", y="Probability",
            color="Category",
            color_discrete_map=BUCKET_COLORS,
            title="Prediction Probability per Category (%)",
            text="Probability",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            height=400, showlegend=False,
            xaxis_title="AQI Category", yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Pipeline comparison ---
        st.markdown("#### 🔄 Dual Pipeline Summary")
        pipe1, pipe2 = st.columns(2)
        with pipe1:
            st.info(f"**Pipeline 1 — XGBoost Regressor**\n\nPredicted AQI Value: **{aqi_value:.1f}**\n\nR² Score: {metrics['xgb_regressor']['r2']:.4f} | MAE: {metrics['xgb_regressor']['mae']:.2f}")
        with pipe2:
            st.info(f"**Pipeline 2 — RF Classifier**\n\nPredicted Category: **{bucket_name}**\n\nAccuracy: {metrics['rf_classifier']['accuracy']:.2%} | Confidence: {max(bucket_proba)*100:.1f}%")


# ============================================================
# PAGE 2: DASHBOARD
# ============================================================
elif page == "📊 Dashboard":

    st.subheader("📊 Overview Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg AQI (All India)", f"{df['AQI'].mean():.0f}")
    c2.metric("Cities", df["City"].nunique())
    c3.metric("Data Points", f"{len(df):,}")
    c4.metric("Years", f"{df['Year'].nunique()}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔴 Most Polluted")
        top5 = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(5).reset_index()
        top5.columns = ["City", "Avg AQI"]
        top5["Avg AQI"] = top5["Avg AQI"].round(1)
        fig = px.bar(top5, x="Avg AQI", y="City", orientation="h", color="Avg AQI",
                     color_continuous_scale="Reds", text="Avg AQI")
        fig.update_layout(height=300, showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🟢 Cleanest")
        bot5 = df.groupby("City")["AQI"].mean().sort_values().head(5).reset_index()
        bot5.columns = ["City", "Avg AQI"]
        bot5["Avg AQI"] = bot5["Avg AQI"].round(1)
        fig = px.bar(bot5, x="Avg AQI", y="City", orientation="h", color="Avg AQI",
                     color_continuous_scale="Greens_r", text="Avg AQI")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    bucket_counts = df["AQI_Bucket"].value_counts().reset_index()
    bucket_counts.columns = ["Category", "Count"]
    fig = px.pie(bucket_counts, values="Count", names="Category", hole=0.4,
                 color="Category", color_discrete_map=BUCKET_COLORS, title="AQI Category Distribution")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: CITY ANALYSIS
# ============================================================
elif page == "🏙️ City Analysis":

    st.subheader("🏙️ City-wise Deep Dive")
    cities = sorted(df["City"].unique())
    selected = st.selectbox("Select City", cities, index=cities.index("Delhi"))
    cdf = df[df["City"] == selected]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg AQI", f"{cdf['AQI'].mean():.0f}")
    k2.metric("Max AQI", f"{cdf['AQI'].max():.0f}")
    k3.metric("Avg PM2.5", f"{cdf['PM2.5'].mean():.1f}")
    k4.metric("Records", f"{len(cdf):,}")

    tab1, tab2 = st.tabs(["📈 AQI Trend", "🧪 Pollutants"])

    with tab1:
        yearly = cdf.groupby("Year")["AQI"].mean().reset_index()
        fig = px.line(yearly, x="Year", y="AQI", markers=True,
                      title=f"Yearly AQI — {selected}")
        fig.update_traces(line=dict(width=3, color="#667eea"), marker=dict(size=10))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pollutants = [p for p in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"] if p in cdf.columns]
        avg = cdf[pollutants].mean()
        fig = px.bar(x=pollutants, y=avg.values, color=avg.values,
                     color_continuous_scale="YlOrRd",
                     labels={"x": "Pollutant", "y": "Average"},
                     title=f"Avg Pollutant Levels — {selected}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: COVID IMPACT
# ============================================================
elif page == "🦠 COVID Impact":

    st.subheader("🦠 COVID-19 Lockdown vs Pre-COVID")
    cities = sorted(df["City"].unique())
    covid_city = st.selectbox("City", cities, index=cities.index("Delhi"))

    cdf = df[(df["City"] == covid_city) & (df["Year"].isin([2019, 2020]))]
    monthly = cdf.groupby(["Month", "Year"])["AQI"].mean().reset_index()
    monthly["Year"] = monthly["Year"].astype(str)

    fig = px.bar(monthly, x="Month", y="AQI", color="Year", barmode="group",
                 title=f"{covid_city}: 2019 vs 2020",
                 color_discrete_map={"2019": "#e74c3c", "2020": "#2ecc71"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    avg19 = cdf[cdf["Year"] == 2019]["AQI"].mean()
    avg20 = cdf[cdf["Year"] == 2020]["AQI"].mean()
    if avg19 > 0:
        pct = ((avg20 - avg19) / avg19) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("2019 Avg AQI", f"{avg19:.0f}")
        m2.metric("2020 Avg AQI", f"{avg20:.0f}")
        m3.metric("Change", f"{pct:+.1f}%", delta=f"{pct:+.1f}%", delta_color="inverse")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; padding:1rem;">
    Built by <a href="https://github.com/ajitesh68" target="_blank">Ajitesh</a> |
    Models: XGBoost + Random Forest |
    <a href="https://www.kaggle.com/rohanrao/air-quality-data-in-india" target="_blank">Kaggle Dataset</a>
</div>
""", unsafe_allow_html=True)
