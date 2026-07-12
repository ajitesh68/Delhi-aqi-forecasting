"""
Orchestrator: Train models using src/ modules and save .pkl files.
Run: python train_and_save_models.py
"""
import joblib, json, os
from src.data_loader import load_city_day
from src.feature_engineering import (
    add_date_features, add_season, compute_composite_score, 
    encode_features, prepare_regression_data, prepare_classification_data
)
from src.preprocessing import full_cleaning_pipeline
from src.modeling import (
    split_data, train_xgb_regressor, train_rf_classifier,
    evaluate_regression, evaluate_classification
)


# 1. Load
df = load_city_day(data_dir="data")
# 2. Date + Season features
df = add_date_features(df)
df = add_season(df)
# 3. Full cleaning (drop cols + impute + outliers)
df = full_cleaning_pipeline(df)
# 4. Composite Score
df,scaler = compute_composite_score(df)
# 5. Encode
df,season_encoder = encode_features(df)
# 6. Prepare X, y
X_reg, y_reg = prepare_regression_data(df)
X_clf, y_clf = prepare_classification_data(df)
# 7. Split
X_train, X_test, y_reg_train, y_reg_test = split_data(X_reg, y_reg)
_, _, y_clf_train, y_clf_test = split_data(X_clf, y_clf)
# 8. Train + Evaluate
xgb = train_xgb_regressor(X_train, y_reg_train, n_estimators=300, learning_rate=0.07)
evaluate_regression(y_reg_test, xgb.predict(X_test), "XGBoost Regressor")
rf = train_rf_classifier(X_train, y_clf_train, n_estimators=80)
evaluate_classification(y_clf_test, rf.predict(X_test), "RF Classifier")





# 9. Save all artifacts
joblib.dump(xgb, "models/xgb_regressor.pkl")
joblib.dump(rf, "models/rf_classifier.pkl")
joblib.dump(season_encoder, "models/season_encoder.pkl")
joblib.dump(scaler, "models/minmax_scaler.pkl")
joblib.dump(X_reg.columns.tolist(), "models/feature_names.pkl")
joblib.dump(sorted(df["City"].unique().tolist()), "models/city_list.pkl")
joblib.dump({0: "Good", 1: "Satisfactory", 2: "Moderate", 3: "Poor", 4: "Very Poor", 5: "Severe"}, "models/bucket_reverse.pkl")
print("DONE!")