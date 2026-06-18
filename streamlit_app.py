"""
🏭 India AQI Analysis & Prediction — Streamlit Dashboard
=========================================================
Yeh file Streamlit Cloud pe deploy hogi.
Run locally: streamlit run streamlit_app.py
"""

# ============================================================
# 📦 IMPORTS
# ============================================================
# streamlit — web app framework (yeh poora UI banata hai)
import streamlit as st

# pandas — data manipulation (CSV load, filter, groupby)
import pandas as pd

# numpy — mathematical operations
import numpy as np

# plotly — interactive charts (hover, zoom, pan support)
# plotly.express = simple API, plotly.graph_objects = advanced control
import plotly.express as px
import plotly.graph_objects as go

# sklearn — Machine Learning models & utilities
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================
# ⚙️ PAGE CONFIGURATION
# ============================================================
# set_page_config() MUST be the FIRST Streamlit command
# Iske pehle koi st.xyz() nahi hona chahiye warna error aayega
st.set_page_config(
    page_title="India AQI Dashboard",    # Browser tab mein dikhega
    page_icon="🏭",                      # Tab icon (emoji ya image path)
    layout="wide",                        # "wide" = full width, "centered" = narrow
    initial_sidebar_state="expanded",     # Sidebar khula rahe by default
)


# ============================================================
# 🗂️ DATA LOADING (with caching)
# ============================================================
# @st.cache_data decorator ka matlab:
# - Pehli baar function run hoga, result CACHE mein store hoga
# - Doosri baar se CACHE se directly result milega (instant!)
# - Bina iske, har click pe CSV dubara load hogi — slow!
# - show_spinner=True se loading ke time "Loading..." dikhega
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """Load and prepare the city_day.csv dataset."""

    # CSV read karo — yeh file repo mein honi chahiye
    df = pd.read_csv("data/city_day.csv")

    # Date column ko string se datetime object mein convert karo
    # Isse .dt.year, .dt.month jaisi properties milti hain
    df["Date"] = pd.to_datetime(df["Date"])

    # Date se Year aur Month extract karo — analysis ke liye useful
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Season mapping function — Indian seasons ke hisaab se
    def get_season(month):
        if month <= 2 or month == 12:
            return "Winter"       # Dec, Jan, Feb
        elif 3 <= month <= 5:
            return "Spring"       # Mar, Apr, May
        elif 6 <= month <= 9:
            return "Monsoon"      # Jun–Sep (India mein rainy season)
        else:
            return "Autumn"       # Oct, Nov

    # apply() — har row ke Month pe get_season function chalata hai
    df["Season"] = df["Month"].apply(get_season)

    # High-nullity columns drop (60%+ missing = unreliable data)
    df = df.drop(columns=["Xylene", "Toluene", "NH3"], errors="ignore")

    # Missing values ko Season*City group ke median se fill karo
    # Yeh smarter hai simple mean/fillna(0) se — context-aware imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["Year", "Month"]]

    for col in num_cols:
        # Group median — same Season + same City ka median
        df[col] = df.groupby(["Season", "City"])[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Agar group median bhi NaN hai, toh global median use karo
        df[col] = df[col].fillna(df[col].median())

    # AQI_Bucket bhi fill karo — mode (sabse common value) se
    df["AQI_Bucket"] = df.groupby(["Season", "City"])["AQI_Bucket"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Moderate")
    )
    df["AQI_Bucket"] = df["AQI_Bucket"].fillna("Moderate")

    return df


# @st.cache_resource — heavy objects (ML models) ke liye
# cache_data vs cache_resource:
#   cache_data = serializable data (df, list, dict)
#   cache_resource = non-serializable objects (model, DB connection)
@st.cache_resource
def train_model(_df):
    """Train a Random Forest model for AQI prediction."""

    # Feature columns — yeh pollutant values hain jo model ko input milenge
    feature_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "CO", "SO2", "O3", "Benzene"]
    # Jo columns actually exist karti hain unhi ko use karo
    feature_cols = [c for c in feature_cols if c in _df.columns]

    # Sirf wahi rows lo jaha AQI aur saare features available hain
    model_df = _df[feature_cols + ["AQI"]].dropna()

    # X = features (input), y = target (output — AQI value)
    X = model_df[feature_cols]
    y = model_df["AQI"]

    # 80% data training ke liye, 20% testing ke liye
    # random_state=42 se har baar same split milega (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # RandomForestRegressor — ensemble model (100 decision trees ka group)
    # n_jobs=-1 = saare CPU cores use karo (fast training)
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Model accuracy check
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, feature_cols, r2, mae


# ============================================================
# 🎨 CUSTOM CSS (Styling — optional but looks premium)
# ============================================================
# st.markdown() mein HTML/CSS inject kar sakte ho
# unsafe_allow_html=True zaroori hai HTML ke liye
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #e0e0ff; margin: 0.3rem 0 0 0; font-size: 1rem; }

    [data-testid="stMetric"] {
        background: #f8f9ff;
        border: 1px solid #e0e5ff;
        padding: 12px 16px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] { background: #fafbff; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 🏠 APP HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🏭 India Air Quality Dashboard</h1>
    <p>Analyzing air quality across 26 Indian cities (2015-2020) with ML-powered AQI prediction</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 📊 DATA LOAD
# ============================================================
# try-except se agar CSV na mile toh app crash nahi hoga
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    # st.error() — Red error box dikhata hai
    st.error("Dataset not found! Place `city_day.csv` in the `data/` folder.")
    st.info("Download from: [Kaggle - Air Quality Data in India](https://www.kaggle.com/rohanrao/air-quality-data-in-india)")
    # st.stop() — App execution yahi rok deta hai
    st.stop()


# ============================================================
# 📱 SIDEBAR — Navigation & Filters
# ============================================================
# st.sidebar — left panel mein content daalta hai
# Yeh global filters ke liye best hai — har page pe dikhega

st.sidebar.title("Controls")

# st.sidebar.radio() — Page navigation ke liye radio buttons
# Ek value return karta hai — jo selected hai
page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "City Analysis",
        "Seasonal Trends",
        "COVID Impact",
        "AQI Predictor",
    ],
    index=0,   # Default selection (0 = pehla option)
)

st.sidebar.divider()  # Horizontal line separator

# Sidebar mein dataset info dikhao
st.sidebar.markdown("### Dataset Info")
st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
st.sidebar.write(f"**Cities:** {df['City'].nunique()}")
st.sidebar.write(f"**Period:** {df['Date'].min().year} - {df['Date'].max().year}")


# ============================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================
if page == "Overview":

    # --- KPI METRICS ROW ---
    # st.columns(4) — 4 equal-width columns banata hai
    # Har column ek independent container hai
    c1, c2, c3, c4 = st.columns(4)

    # st.metric() — Dashboard-style KPI card
    # Parameters: label, value, delta (optional — change indicator)
    avg_aqi = df["AQI"].mean()
    c1.metric("Avg AQI (All India)", f"{avg_aqi:.0f}")
    c2.metric("Cities Tracked", df["City"].nunique())
    c3.metric("Data Points", f"{len(df):,}")
    c4.metric("Years Covered", f"{df['Year'].nunique()}")

    st.divider()

    # --- TOP & BOTTOM CITIES ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Most Polluted Cities")
        # groupby + mean + sort = city-wise average AQI ranking
        top5 = (
            df.groupby("City")["AQI"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top5.columns = ["City", "Average AQI"]
        top5["Average AQI"] = top5["Average AQI"].round(1)

        # Plotly horizontal bar chart — interactive!
        fig = px.bar(
            top5,
            x="Average AQI",
            y="City",
            orientation="h",          # h = horizontal bars
            color="Average AQI",
            color_continuous_scale="Reds",  # Red gradient
            text="Average AQI",
        )
        fig.update_layout(
            height=350,
            showlegend=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Cleanest Cities")
        bottom5 = (
            df.groupby("City")["AQI"]
            .mean()
            .sort_values(ascending=True)
            .head(5)
            .reset_index()
        )
        bottom5.columns = ["City", "Average AQI"]
        bottom5["Average AQI"] = bottom5["Average AQI"].round(1)

        fig = px.bar(
            bottom5,
            x="Average AQI",
            y="City",
            orientation="h",
            color="Average AQI",
            color_continuous_scale="Greens_r",
            text="Average AQI",
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- AQI BUCKET DISTRIBUTION ---
    st.subheader("AQI Category Distribution")
    bucket_counts = df["AQI_Bucket"].value_counts().reset_index()
    bucket_counts.columns = ["AQI Bucket", "Count"]

    # Donut chart (pie chart with hole)
    fig = px.pie(
        bucket_counts,
        values="Count",
        names="AQI Bucket",
        hole=0.4,   # 0.4 = 40% hole = donut shape
        color="AQI Bucket",
        color_discrete_map={
            "Good": "#2ecc71",
            "Satisfactory": "#27ae60",
            "Moderate": "#f39c12",
            "Poor": "#e74c3c",
            "Very Poor": "#c0392b",
            "Severe": "#8e44ad",
        },
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: CITY ANALYSIS
# ============================================================
elif page == "City Analysis":

    st.subheader("City-wise Deep Dive")

    selected_city = st.selectbox(
        "Select a City",
        sorted(df["City"].unique()),
        index=list(sorted(df["City"].unique())).index("Delhi"),
    )

    city_df = df[df["City"] == selected_city]

    # --- City KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg AQI", f"{city_df['AQI'].mean():.0f}")
    k2.metric("Max AQI", f"{city_df['AQI'].max():.0f}")
    k3.metric("Avg PM2.5", f"{city_df['PM2.5'].mean():.1f}")
    k4.metric("Data Points", f"{len(city_df):,}")

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["AQI Trend", "Pollutants", "Raw Data"])

    with tab1:
        yearly = city_df.groupby("Year")["AQI"].mean().reset_index()
        fig = px.line(
            yearly, x="Year", y="AQI", markers=True,
            title=f"Yearly Average AQI - {selected_city}",
        )
        fig.update_traces(line=dict(width=3, color="#667eea"), marker=dict(size=10))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
        pollutants = [p for p in pollutants if p in city_df.columns]
        avg_vals = city_df[pollutants].mean()

        fig = px.bar(
            x=pollutants, y=avg_vals.values,
            labels={"x": "Pollutant", "y": "Average Value"},
            title=f"Average Pollutant Levels - {selected_city}",
            color=avg_vals.values,
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(
            city_df[["Date", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI", "AQI_Bucket"]]
            .sort_values("Date", ascending=False)
            .head(100),
            use_container_width=True,
        )


# ============================================================
# PAGE 3: SEASONAL TRENDS
# ============================================================
elif page == "Seasonal Trends":

    st.subheader("Season-wise Air Quality Analysis")

    season_aqi = df.groupby("Season")["AQI"].mean().reset_index()
    season_order = ["Winter", "Spring", "Monsoon", "Autumn"]
    season_aqi["Season"] = pd.Categorical(
        season_aqi["Season"], categories=season_order, ordered=True
    )
    season_aqi = season_aqi.sort_values("Season")

    fig = px.bar(
        season_aqi, x="Season", y="AQI", color="Season",
        color_discrete_map={
            "Winter": "#3498db", "Spring": "#2ecc71",
            "Monsoon": "#1abc9c", "Autumn": "#e67e22",
        },
        title="Average AQI by Season (All Cities)",
        text="AQI",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- City x Season Heatmap ---
    st.subheader("City x Season AQI Heatmap")

    pivot = df.pivot_table(values="AQI", index="City", columns="Season", aggfunc="mean")
    pivot = pivot[[s for s in season_order if s in pivot.columns]]

    fig = px.imshow(
        pivot.round(0),
        text_auto=True,
        color_continuous_scale="YlOrRd",
        labels=dict(x="Season", y="City", color="AQI"),
        title="Average AQI: City x Season",
        aspect="auto",
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: COVID IMPACT
# ============================================================
elif page == "COVID Impact":

    st.subheader("COVID-19 Lockdown Impact on Air Quality")

    st.markdown("""
    > India imposed a **nationwide lockdown** from **March 25, 2020**.
    > Compare air quality before vs during lockdown to see the environmental impact.
    """)

    covid_city = st.selectbox(
        "Select City for COVID Analysis",
        sorted(df["City"].unique()),
        index=list(sorted(df["City"].unique())).index("Delhi"),
    )

    covid_df = df[
        (df["City"] == covid_city) & (df["Year"].isin([2019, 2020]))
    ]

    monthly = (
        covid_df.groupby(["Month", "Year"])["AQI"].mean().reset_index()
    )
    monthly["Year"] = monthly["Year"].astype(str)

    fig = px.bar(
        monthly, x="Month", y="AQI", color="Year",
        barmode="group",
        title=f"{covid_city}: Monthly AQI - 2019 vs 2020",
        color_discrete_map={"2019": "#e74c3c", "2020": "#2ecc71"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # --- Percentage Change ---
    st.subheader("Percentage Change in AQI")

    avg_2019 = covid_df[covid_df["Year"] == 2019]["AQI"].mean()
    avg_2020 = covid_df[covid_df["Year"] == 2020]["AQI"].mean()

    if avg_2019 > 0:
        pct_change = ((avg_2020 - avg_2019) / avg_2019) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("2019 Avg AQI", f"{avg_2019:.0f}")
        c2.metric("2020 Avg AQI", f"{avg_2020:.0f}")
        c3.metric(
            "Change", f"{pct_change:+.1f}%",
            delta=f"{pct_change:+.1f}%",
            delta_color="inverse",
        )


# ============================================================
# PAGE 5: AQI PREDICTOR (ML Model)
# ============================================================
elif page == "AQI Predictor":

    st.subheader("Predict AQI from Pollutant Values")
    st.markdown("Enter pollutant concentrations below and the ML model will predict the AQI.")

    # Train model (cached — sirf pehli baar train hoga)
    with st.spinner("Training model (one-time, cached for future)..."):
        model, feature_cols, r2, mae = train_model(df)

    st.success(f"Model ready! R2 Score: **{r2:.4f}** | MAE: **{mae:.2f}**")

    st.divider()

    # --- Prediction Form ---
    with st.form("aqi_prediction_form"):
        st.markdown("### Enter Pollutant Values")

        col1, col2 = st.columns(2)
        input_values = {}

        for i, feat in enumerate(feature_cols):
            col = col1 if i % 2 == 0 else col2
            feat_min = float(df[feat].min())
            feat_max = float(df[feat].max())
            feat_mean = float(df[feat].mean())

            with col:
                input_values[feat] = st.slider(
                    f"{feat}",
                    min_value=feat_min,
                    max_value=feat_max,
                    value=feat_mean,
                    step=0.1,
                    help=f"Range: {feat_min:.1f} - {feat_max:.1f}",
                )

        submitted = st.form_submit_button(
            "Predict AQI", use_container_width=True,
        )

    # --- Prediction Result ---
    if submitted:
        input_array = np.array([[input_values[f] for f in feature_cols]])
        prediction = model.predict(input_array)[0]

        def get_bucket(aqi):
            if aqi <= 50:
                return "Good", "#2ecc71"
            elif aqi <= 100:
                return "Satisfactory", "#f1c40f"
            elif aqi <= 200:
                return "Moderate", "#e67e22"
            elif aqi <= 300:
                return "Poor", "#e74c3c"
            elif aqi <= 400:
                return "Very Poor", "#8e44ad"
            else:
                return "Severe", "#2c3e50"

        bucket, color = get_bucket(prediction)

        st.divider()
        r1, r2_col, r3 = st.columns(3)
        r1.metric("Predicted AQI", f"{prediction:.0f}")
        r2_col.metric("Category", f"{bucket}")
        r3.metric("Health Advisory",
                   "Safe" if prediction <= 100 else
                   "Caution" if prediction <= 200 else
                   "Unhealthy" if prediction <= 300 else "Dangerous")

        st.markdown(f"""
        <div style="background:{color}; padding:1rem; border-radius:10px; text-align:center; margin-top:1rem;">
            <h2 style="color:white; margin:0;">AQI: {prediction:.0f} - {bucket}</h2>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#888; padding:1rem;">
        Built by <a href="https://github.com/ajitesh68" target="_blank">Ajitesh</a> |
        Data: <a href="https://www.kaggle.com/rohanrao/air-quality-data-in-india" target="_blank">Kaggle</a> |
        Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
