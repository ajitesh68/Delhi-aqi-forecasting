"""
Train both models (XGBoost Regressor + Random Forest Classifier)
from scratch using the cleaned dataset, and save fresh .pkl files.

Run: python train_and_save_models.py
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
from xgboost import XGBRegressor

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
df = pd.read_csv("data/city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ============================================================
# 2. SEASON MAPPING
# ============================================================
print("\nSTEP 2: Adding season...")
def get_season(month):
    if month <= 2 or month == 12:
        return "Winter"
    elif 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 9:
        return "Monsoon"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

# ============================================================
# 3. DROP HIGH-NULLITY COLUMNS
# ============================================================
print("STEP 3: Dropping high-nullity columns...")
df = df.drop(columns=["Xylene", "Toluene", "NH3"], errors="ignore")

# ============================================================
# 4. IMPUTE MISSING VALUES (Season x City median)
# ============================================================
print("STEP 4: Imputing missing values...")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["Year", "Month"]]

for col in num_cols:
    df[col] = df.groupby(["Season", "City"])[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())

# AQI_Bucket imputation
df["AQI_Bucket"] = df.groupby(["Season", "City"])["AQI_Bucket"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Moderate")
)
df["AQI_Bucket"] = df["AQI_Bucket"].fillna("Moderate")
print(f"  Remaining nulls: {df.isnull().sum().sum()}")

# ============================================================
# 5. COMPOSITE SCORE (MinMax normalized)
# ============================================================
print("\nSTEP 5: Computing Composite Score...")
weights = {"PM2.5": 0.35, "PM10": 0.20, "NO2": 0.15, "CO": 0.10, "SO2": 0.10, "O3": 0.10}
pollutant_cols = list(weights.keys())

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[pollutant_cols] = scaler.fit_transform(df[pollutant_cols])
df["Composite_Score"] = sum(df_scaled[col] * weights[col] for col in pollutant_cols)

# Save the scaler for Streamlit app to use
joblib.dump(scaler, "models/minmax_scaler.pkl")
print("  Saved: models/minmax_scaler.pkl")

# ============================================================
# 6. ENCODE FEATURES
# ============================================================
print("\nSTEP 6: Encoding features...")
le = LabelEncoder()
df["Season_Encoded"] = le.fit_transform(df["Season"])
joblib.dump(le, "models/season_encoder.pkl")
print("  Saved: models/season_encoder.pkl")

# City one-hot
city_dummies = pd.get_dummies(df["City"], prefix="City")
df = pd.concat([df, city_dummies], axis=1)

# AQI Bucket ordinal encoding
bucket_order = {"Good": 0, "Satisfactory": 1, "Moderate": 2, "Poor": 3, "Very Poor": 4, "Severe": 5}
bucket_reverse = {v: k for k, v in bucket_order.items()}
df["AQI_Bucket_Encoded"] = df["AQI_Bucket"].map(bucket_order)

# Save mappings
joblib.dump(bucket_order, "models/bucket_order.pkl")
joblib.dump(bucket_reverse, "models/bucket_reverse.pkl")

# Save city list for Streamlit
city_list = sorted(df["City"].unique().tolist())
joblib.dump(city_list, "models/city_list.pkl")

# ============================================================
# 7. PREPARE FEATURES
# ============================================================
print("\nSTEP 7: Preparing features...")
drop_cols = ["AQI", "AQI_Bucket", "City", "Season", "Date", "Month", "AQI_Bucket_Encoded"]
existing_drops = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=existing_drops).select_dtypes(include=[np.number])
y_reg = df["AQI"]
y_clf = df["AQI_Bucket_Encoded"]

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/feature_names.pkl")
print(f"  Features ({len(feature_names)}): {feature_names[:10]}...")

# Train-test split (same random state as notebook)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# ============================================================
# 8. TRAIN XGBoost REGRESSOR
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Training XGBoost Regressor...")
xgb_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.07,
    max_depth=6,
    random_state=42,
)
xgb_reg.fit(X_train, y_reg_train)

y_pred_reg = xgb_reg.predict(X_test)
r2 = r2_score(y_reg_test, y_pred_reg)
mae = mean_absolute_error(y_reg_test, y_pred_reg)

print(f"  R2 Score: {r2:.4f}")
print(f"  MAE:      {mae:.4f}")

joblib.dump(xgb_reg, "models/xgb_regressor.pkl")
print("  Saved: models/xgb_regressor.pkl")

# ============================================================
# 9. TRAIN Random Forest CLASSIFIER
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=80,  # Reduced from 200 to keep file size under 100MB for GitHub
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train, y_clf_train)

y_pred_clf = rf_clf.predict(X_test)
acc = accuracy_score(y_clf_test, y_pred_clf)

print(f"  Accuracy: {acc:.4f}")
print(f"\n{classification_report(y_clf_test, y_pred_clf, target_names=[bucket_reverse[i] for i in sorted(bucket_reverse.keys())])}")

joblib.dump(rf_clf, "models/rf_classifier.pkl")
print("  Saved: models/rf_classifier.pkl")

# ============================================================
# 10. SAVE METRICS
# ============================================================
import json
metrics = {
    "xgb_regressor": {"r2": round(r2, 4), "mae": round(mae, 4)},
    "rf_classifier": {"accuracy": round(acc, 4)},
    "feature_count": len(feature_names),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("ALL DONE! Models saved in models/ folder:")
for f in os.listdir("models"):
    size = os.path.getsize(f"models/{f}")
    print(f"  {f:30s} {size/1024:.0f} KB")
