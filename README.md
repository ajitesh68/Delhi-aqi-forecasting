# 🏭 India Air Quality Index (AQI) — Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/rohanrao/air-quality-data-in-india)

> **End-to-end data science project** analyzing air quality across 26 Indian cities (2015–2020), building predictive models for AQI values and AQI bucket classification, with SHAP-based model explainability.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Key Findings](#-key-findings)
- [Methodology](#-methodology)
- [Model Results](#-model-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## 🎯 Problem Statement

Air pollution is one of India's most critical environmental challenges. This project aims to:

1. **Analyze** pollutant trends across 26 Indian cities over 5+ years
2. **Identify** the most and least polluted cities, seasonal patterns, and COVID-19's impact
3. **Predict** AQI values using regression models (Random Forest, XGBoost)
4. **Classify** AQI into health-risk buckets (Good → Severe)
5. **Explain** model predictions using SHAP for actionable insights

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Air Quality Data in India](https://www.kaggle.com/rohanrao/air-quality-data-in-india) |
| **Time Period** | January 2015 — July 2020 |
| **Cities** | 26 Indian cities |
| **Records** | 29,531 daily observations |
| **Features** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket |

---

## 🔍 Key Findings

- **Most polluted cities:** Delhi, Patna, Gurugram, Lucknow, Ahmedabad
- **Cleanest cities:** Aizawl, Shillong, Coimbatore, Thiruvananthapuram
- **Winter pollution is 2x Monsoon:** Winter AQI avg = 220 vs Monsoon = 115
- **COVID-19 lockdown impact:** Delhi's AQI dropped ~40% in March–April 2020 vs 2019
- **PM2.5 is the strongest predictor** of AQI across all models
- **Composite Pollution Score** (custom weighted metric) shows 0.90+ correlation with official AQI

---

## 🔬 Methodology

```
Data Loading → EDA → Data Cleaning → Feature Engineering → Visualization → Modeling → Explainability
```

### Data Cleaning
- Dropped high-nullity columns: Xylene (61%), Toluene (27%), NH3 (35%)
- Imputed missing values using **Season × City median** (context-aware)
- Handled AQI outliers (Ahmedabad > 500 capped using seasonal medians)

### Feature Engineering
- Season mapping (Winter/Spring/Monsoon/Autumn — Indian context)
- Composite Pollution Score (weighted: PM2.5=35%, PM10=20%, NO2=15%, CO=10%, SO2=10%, O3=10%)
- One-hot encoding for cities, label encoding for seasons

### Models Built
- Random Forest Regressor & Classifier
- XGBoost Regressor & Classifier (with GridSearchCV tuning)
- SHAP Explainability

---

## 📈 Model Results

### Regression — AQI Prediction

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Random Forest | 16.68 | 29.86 | 0.9217 |
| XGBoost | 16.73 | 29.40 | 0.9241 |
| **XGBoost (Tuned)** | — | — | **0.9280** |

### Classification — AQI Bucket Prediction

| Model | Accuracy |
|-------|----------|
| Random Forest Classifier | 83.14% |
| XGBoost Classifier | 83.13% |

---

## 📁 Project Structure

```
india-aqi-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # Original Kaggle CSVs
│   └── processed/              # Cleaned & engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_visualization.ipynb
│   ├── 05_modeling_regression.ipynb
│   ├── 06_modeling_classification.ipynb
│   └── 07_model_explainability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
│
├── models/                     # Saved trained models
├── reports/
│   └── figures/                # Saved plot images
└── app/                        # (Future) Dashboard app
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/india-aqi-analysis.git
cd india-aqi-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Data
Download the dataset from [Kaggle](https://www.kaggle.com/rohanrao/air-quality-data-in-india) and place CSV files in `data/raw/`.

### 4. Run Notebooks
Execute notebooks in order (01 → 07) from the `notebooks/` directory.

---

## 🔮 Future Scope

- [ ] **Interactive Dashboard** — Streamlit/Flask web app for city-wise AQI visualization
- [ ] **Time Series Forecasting** — LSTM / Facebook Prophet for AQI prediction
- [ ] **Real-time AQI API** — Integration with government AQI APIs
- [ ] **Geographic Heatmap** — Folium/Plotly map visualization
- [ ] **Health Impact Calculator** — WHO-based health risk estimation
- [ ] **Extended Dataset** — Integrate 2015–2023 data for broader analysis

---

## 👤 Author

**Ajitesh**  
- GitHub: [@ajitesh68](https://github.com/ajitesh68)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
