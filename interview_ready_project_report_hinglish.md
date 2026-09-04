# Delhi Multivariate AQI Forecasting
## Interview-Ready Complete Project Report

**Project type:** End-to-end Machine Learning and Streamlit deployment project  
**Model:** Multivariate LSTM  
**Forecast:** Next 72 hours  
**Input history:** Previous 336 hourly records, yani 14 days  
**Locations:** Anand Vihar, Connaught Place, Dwarka, IGI Airport, Okhla Phase III, Rohini  
**Main goal:** Future pollutant concentrations predict karke AQI aur health category dikhana

---

## 1. Project ko ek line mein kaise explain karna hai

"Maine Delhi ke six locations ke liye ek multivariate time-series forecasting system banaya hai jo pichhle 14 din ke pollution, weather aur seasonal features ko use karke agle 72 ghanton ke PM2.5, PM10, CO aur NO2 predict karta hai. In predictions ko CPCB-style breakpoint formula se AQI mein convert karke Streamlit dashboard mein forecast, pollutant trends, health advisory aur backtesting dikhaya gaya hai."

---

## 2. Business problem

Air pollution ka impact health, travel, outdoor work aur city planning par directly padta hai. Sirf current AQI dekhna enough nahi hai; agar agle 1-3 din ka trend pata ho toh log outdoor activity, mask use aur travel planning better kar sakte hain.

Problem ko ML terms mein:

- **Input:** Historical hourly pollution + weather + calendar information.
- **Output:** Future 72 hours ke four pollutant concentrations.
- **Final user output:** AQI number, category, dominant pollutant aur health advisory.

Direct AQI predict karne ke bajay pollutants predict kiye gaye hain, kyunki pollutant-level output zyada interpretable hai. User dekh sakta hai ki AQI high hone ka reason PM2.5 hai ya PM10.

---

## 3. Project ka complete flow

```text
Raw CSV / Open-Meteo API
          |
          v
Data cleaning and datetime alignment
          |
          v
Feature engineering
          |
          v
Scaling with saved MinMaxScaler
          |
          v
336-hour input sequence
          |
          v
Location-specific LSTM
          |
          v
72 x 4 future pollutant values
          |
          v
Inverse scaling
          |
          v
CPCB-style AQI calculation
          |
          v
Streamlit cards, tables, charts, advisory, backtest
```

Training aur inference mein same feature order aur same scaler use karna bahut important hai. Agar training ke time feature order alag ho aur live app mein alag, model technically run karega par prediction wrong ho sakti hai.

---

## 4. Repository structure aur har file ka role

### `app.py`

Main Streamlit application. Is file mein UI, API data fetching, model loading, preprocessing for live data, prediction, calibration, AQI display aur backtesting hai.

### `src/prepare_lstm_data.py`

Training data preparation file. Raw CSV read karti hai, missing values handle karti hai, features banati hai, scaling karti hai, sequences banati hai aur `.npy`/`.pkl` artifacts save karti hai.

### `src/lstm_model.py`

LSTM architecture aur training script. Har location ke liye model train karta hai. Current version mein `--force`, `--epochs` aur `--batch-size` command-line options hain. Training ke baad `models/lstm_metrics.json` save hota hai.

### `src/aqi_formula.py`

PM2.5, PM10, CO aur NO2 ke concentration ko sub-index mein map karti hai. Final AQI maximum sub-index hota hai. Isi file mein category aur health advisory functions hain.

### `scripts/download_2026_data.py`

Open-Meteo historical weather aur air-quality API se data download karke raw CSV banata hai. Windows ke liye hourly time formatting fix ki gayi hai.

### `models/`

Location-specific `.h5` LSTM models, feature scalers, target scalers aur metrics file.

### `data/raw/`

Original downloaded CSV files.

### `data/processed/`

Training ke liye generated `X_train`, `X_test`, `y_train`, `y_test` NumPy arrays.

### `reports/`

Project analysis, user guide aur interview-ready documentation.

---

## 5. Data kya use ho raha hai

Features ko do groups mein divide kiya gaya hai.

### Pollution features

```text
pm2_5, pm10, co, no2
```

### Weather features

```text
temp_c, humidity, pressure_mb, windspeed_kph
```

In 8 base features ke alawa six engineered features hain:

```text
hour_sin, hour_cos,
month_sin, month_cos,
is_weekend,
days_to_diwali
```

Total input features:

```text
8 base features + 6 engineered features = 14 features
```

### Data source ka important point

Current system Open-Meteo air-quality estimates use karta hai. Yeh useful development source hai, lekin CPCB station observation ke exactly equal nahi hota. Isi wajah se app ka AQI aur kisi website/station ka AQI different aa sakta hai.

Production-level accuracy ke liye CPCB station data ko ground truth banana chahiye. Open-Meteo weather data input ke liye use kiya ja sakta hai, par pollution labels ideally same station ke hone chahiye jisse actual comparison kar rahe hain.

---

## 6. Data preparation ka logic

### Step 1: Raw files load karna

`load_and_combine()` raw directory mein `delhi-weather-aqi*.csv` files read karta hai. Files combine karke location aur datetime ke basis par sort hoti hain.

### Step 2: Datetime banana

Data mein date aur time alag columns ho sakte hain. Code unko combine karke pandas datetime banata hai. Agar timestamp already ISO format mein hai toh direct parse hota hai.

Datetime time-series projects ka backbone hai. Galat timezone ya galat ordering se model ka input aur target mismatch ho sakta hai.

### Step 3: Missing values

Pollution aur weather columns mein missing values ko linear interpolation se fill kiya jaata hai. Uske baad agar required features mein values missing rah jaati hain toh rows drop hoti hain.

Interpolation tab reasonable hai jab missing gap chhota ho. Bahut long missing gaps ke liye interpolation artificial trend bana sakti hai; future version mein missingness flag aur station-quality checks useful honge.

### Step 4: Time features

Hour aur month ko directly integer ki tarah dene par cyclic nature lost ho jaata hai. 23:00 aur 00:00 actually paas-paas hain, lekin integer encoding mein 23 aur 0 door lagte hain. Isliye sine/cosine encoding use hoti hai:

```text
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
month_sin = sin(2 * pi * month / 12)
month_cos = cos(2 * pi * month / 12)
```

Sine/cosine pair model ko periodic pattern samajhne mein help karta hai.

### Step 5: Weekend feature

Saturday/Sunday ko `is_weekend = 1`, baaki din `0`. Traffic aur human activity ki wajah se weekend pollution pattern different ho sakta hai.

### Step 6: Diwali feature

`days_to_diwali` nearest known Diwali date tak signed day distance capture karta hai. Delhi mein Diwali/firecracker season pollution spike la sakta hai. Yeh feature seasonal event ko model ke liye visible banata hai.

Limitation: Diwali feature pollution ka complete explanation nahi hai. Weather inversion, crop burning, wind direction aur traffic bhi important factors hain.

---

## 7. Scaling ka logic

Neural networks ko different ranges wale features dena difficult hota hai. Example:

- humidity: 0-100
- pressure: around 1000
- CO: thousands
- cyclical feature: -1 to 1

Isliye `MinMaxScaler` use hota hai. Current corrected pipeline mein:

```python
target_values = target_scaler.fit_transform(loc_df[POLLUTANTS])
feature_values = scaler.fit_transform(loc_df[feature_cols])
```

Feature scaler input ke 14 columns ke liye save hota hai. Target scaler sirf four pollutant target columns ke liye save hota hai.

Inference mein live input feature scaler se transform hota hai. Model output target scaler ke `inverse_transform()` se original pollutant units mein aata hai.

### Scaler leakage ka rule

Ideal practice hai ki scaler sirf training period par fit ho aur validation/test/future data par sirf transform ho. Current pipeline mein scaler location data ke full range par fit hota hai; future improvement mein isko train-only fit karna chahiye, taaki test information indirectly training mein na aaye.

---

## 8. Sequence creation

Model ek single row nahi dekhta. Usko previous 336 hourly rows ka context diya jaata hai.

```text
Input shape = (336, 14)
```

Target next 72 hours ke four pollutants hain:

```text
Target shape = (72, 4)
Flattened output = 72 * 4 = 288
```

Sequence sliding-window se banti hai:

```text
X[i] = data[i : i + 336]
y[i] = target[i + 336 : i + 336 + 72]
```

Iska matlab model ko past 14 din dekar immediately next 3 din ka target diya jaata hai.

Location data ko chronological order mein 85% train aur 15% test mein split kiya jaata hai. Time-series ke liye chronological split random split se better hai, kyunki future rows ko past training mein mix nahi karna chahiye.

---

## 9. LSTM architecture

Current architecture:

```text
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(64, activation="relu")
Dense(288, activation="linear")
```

### Layer explanation

- First LSTM 64 units: 336-hour sequence se temporal patterns learn karta hai aur har time step ka representation next LSTM ko deta hai.
- `return_sequences=True`: second LSTM ko poori sequence output milti hai.
- First Dropout 0.2: overfitting reduce karne ke liye 20% activations randomly drop hoti hain.
- Second LSTM 32 units: sequence ko compact summary mein convert karta hai.
- Second Dropout: additional regularization.
- Dense 64 ReLU: nonlinear combination of learned patterns.
- Final Dense 288 linear: future 72 x 4 pollutant values.

### Training settings

- Optimizer: Adam
- Initial learning rate: 0.001
- Loss: Mean Squared Error
- Metric: Mean Absolute Error
- Batch size: 64
- Maximum epochs: 100
- Early stopping patience: 10
- ReduceLROnPlateau: validation improvement slow hone par learning rate reduce

MSE training ke liye useful hai, lekin AQI interpretation ke liye pollutant MAE aur final AQI MAE zyada meaningful hain.

---

## 10. Training script ka detailed logic

`train_location(loc_tag)` location ke four NumPy arrays load karta hai. `y` ko flatten kiya jaata hai:

```text
(samples, 72, 4) -> (samples, 288)
```

Input shape dynamically `X_train.shape` se li jaati hai, isliye code hard-coded sample count par dependent nahi hai.

Model train hone ke baad `.h5` file save hoti hai. Test loss aur MAE print hote hain.

`train_all()` processed directory mein available locations detect karta hai. Default mode existing models ko skip karta hai. Updated model banane ke liye:

```bash
python src/lstm_model.py --force --epochs 100 --batch-size 64
```

`--force` existing `.h5` ko skip nahi hone deta. Training ke baad `models/lstm_metrics.json` location-wise scaled test MSE/MAE save karta hai.

### Checkpoint limitation

Current script final model save karta hai, per-epoch checkpoint nahi. Long Colab session disconnect hone par training progress lose ho sakti hai. Future improvement mein `ModelCheckpoint` aur Google Drive save add karna chahiye.

---

## 11. AQI calculation ka logic

Har pollutant ke liye concentration breakpoint table defined hai. General interpolation formula:

```text
sub_index = ((I_high - I_low) / (C_high - C_low))
             * (concentration - C_low) + I_low
```

Jahan:

- `C_low`, `C_high`: concentration breakpoint limits
- `I_low`, `I_high`: AQI index limits

Four sub-indices calculate hote hain:

```text
PM2.5 index
PM10 index
CO index
NO2 index
```

Final AQI:

```text
AQI = max(all pollutant sub-indices)
```

Dominant pollutant wahi hota hai jiska sub-index maximum ho.

### AQI mismatch ka important explanation

Agar PM2.5 thoda high hai toh final AQI PM2.5 index se high ho sakta hai. Isliye pollutant predictions dekhna zaroori hai, sirf final AQI nahi.

Official comparison ke liye concentration units, averaging duration, breakpoint boundaries aur missing pollutant handling CPCB source ke saath verify karna chahiye.

---

## 12. Streamlit application ka logic

User location select karta hai aur Run Forecast click karta hai.

### Model loading

`load_forecast_model()` selected location ke naam ko filename tag mein convert karta hai, `.h5` load karta hai aur Streamlit resource cache use karta hai.

### Live data fetching

Weather ke liye archive API se historical data aur forecast API se current-day data liya jaata hai. Pollution ke liye Open-Meteo air-quality API use hoti hai. Dono dataframes datetime par inner merge hote hain.

### Live sequence

Live dataframe par wahi time features add kiye jaate hain jo training mein use hue the. Saved feature scaler se transform karke last 336 rows input ban jaati hain.

### Prediction

Model output 288 scaled values deta hai. Reshape ke baad target scaler inverse-transform karta hai:

```text
(288,) -> (72, 4) -> original pollutant units
```

Negative pollutant predictions ko zero se clip kiya jaata hai.

### Calibration

Latest 24-hour pollutant median ka limited blend use hota hai:

```text
final = 0.75 * model_prediction + 0.25 * recent_median
```

Yeh short-term distribution shift ko thoda reduce kar sakta hai. Yeh actual CPCB mismatch ka permanent solution nahi hai aur iski value backtesting se validate karni chahiye.

### Dashboard output

- 3-day average AQI cards
- Category: Good/Satisfactory/Moderate/Poor/Very Poor/Severe
- Dominant pollutant
- Health advisory
- Har din ki hourly table
- 72-hour pollutant line charts
- Past 7-day predicted vs actual Open-Meteo backtest
- Average AQI error metric

---

## 13. Accuracy issue ka real diagnosis

Current 90 vs 150-160 ka difference dekhkar sirf "LSTM weak hai" bolna incomplete hoga. Likely causes:

1. **Source mismatch:** Open-Meteo gridded estimate vs CPCB station measurement.
2. **Location mismatch:** Coordinate station ke exact sensor location ka proxy ho sakta hai.
3. **Time mismatch:** Hourly forecast vs daily/rolling official AQI.
4. **Unit mismatch:** Especially CO units verify karne honge.
5. **AQI formula mismatch:** Breakpoints/averaging rules exactly match karne honge.
6. **Multi-step uncertainty:** 72-hour direct output mein Day 2/Day 3 error naturally badh sakta hai.
7. **Data distribution shift:** Training period aur current weather/pollution regime different ho sakta hai.
8. **Scaler fitting:** Full location data par scaler fit karna leakage risk create karta hai.
9. **Validation mismatch:** Current backtest Open-Meteo values ko actual label maanta hai; CPCB truth ke against nahi.

### Scaling correction ka honest result

Target scaling ko explicit separate banaya gaya hai. Yeh implementation ko correct aur understandable banata hai, lekin same pollutant columns ke MinMax ranges ki wajah se is dataset mein isse automatic huge accuracy jump expect nahi karna chahiye.

---

## 14. Evaluation kaise karni chahiye

Scaled MSE/MAE alone user-facing AQI quality nahi batate. Evaluation mein yeh metrics honi chahiye:

### Pollutant level

- PM2.5 MAE/RMSE
- PM10 MAE/RMSE
- CO MAE/RMSE
- NO2 MAE/RMSE

### AQI level

- Day 1 AQI MAE
- Day 2 AQI MAE
- Day 3 AQI MAE
- RMSE
- Mean signed bias
- Percentage predictions within +/-10 AQI
- Percentage within +/-20 AQI
- Category accuracy

### Baseline comparison

LSTM ko kam se kam in baselines se compare karo:

1. Last observed value.
2. Last 24-hour average.
3. Same hour previous day.
4. Seasonal moving average.
5. XGBoost/Random Forest.

Agar LSTM baseline se better nahi hai, toh architecture ko blindly complex karna useful nahi hoga.

### Fair validation

CPCB station data ka exact table banao:

```text
datetime, station, pm2_5, pm10, co, no2, official_aqi
```

Phir API/model forecast ko same datetime, same station aur same averaging window par join karke error nikalo. Isi comparison se 90 vs 150 ka real reason clear hoga.

---

## 15. Contributions

Interview mein personal contribution is tarah explain kar sakte ho:

- End-to-end air-quality forecasting pipeline design ki.
- Six Delhi locations ke liye separate models organize kiye.
- Pollution aur weather features integrate kiye.
- Cyclic time encoding aur Diwali seasonal feature add kiya.
- Sliding-window time-series dataset banaya.
- LSTM architecture, dropout, early stopping aur learning-rate reduction implement kiya.
- CPCB-style AQI calculation aur dominant pollutant logic banaya.
- Streamlit dashboard mein live API inference integrate ki.
- 72-hour forecast visualization aur health advisories add ki.
- Backtesting view add kiya.
- Input/target scaling ko explicit separate pipeline mein clean kiya.
- Windows-compatible data downloader fix kiya.
- Colab ke liye force retraining aur metrics export workflow banaya.
- Documentation aur reproducible run instructions create ki.

---

## 16. Challenges aur unke solutions

### Challenge 1: Long sequence training slow thi

14 days x hourly data aur 72-hour output ke saath LSTM CPU par expensive hai. Solution: Colab GPU workflow, early stopping aur configurable batch size.

### Challenge 2: Existing models automatically skip ho rahe the

Training script existing `.h5` milne par skip karti thi. Isliye updated preprocessing ke baad bhi purane models reh sakte the. Solution: `--force` option.

### Challenge 3: Actual AQI se large difference

Initial assumption model error thi, lekin deeper issue source/time/aggregation mismatch ho sakta hai. Solution: pollutant-level comparison aur same-source CPCB validation.

### Challenge 4: Feature cyclicity

Hour 23 aur hour 0 ko simple integer encoding wrong distance samjha sakta tha. Solution: sine/cosine encoding.

### Challenge 5: Future seasonal spike

Diwali pollution spike normal trend se alag ho sakti hai. Solution: `days_to_diwali` feature.

### Challenge 6: Windows compatibility

Unix-style `%-H` formatting Windows par fail ho sakti thi. Solution: pandas datetime hour ko string mein convert karna.

### Challenge 7: Model output interpretation

Model direct AQI nahi, pollutant concentrations predict karta hai. Solution: inverse scaling ke baad AQI formula aur dominant pollutant display.

---

## 17. Limitations

- Open-Meteo pollution labels CPCB station observations nahi hain.
- AQI exact official reading ke roop mein claim nahi karna chahiye.
- Current weather inputs aur pollution inputs same source reliability nahi rakhte.
- Direct 72-hour output long horizon par uncertain hai.
- Confidence intervals nahi dikhaye gaye.
- Current scaler fitting train-only nahi hai.
- Current backtest Open-Meteo actual values par based hai.
- API unavailable hone par live forecast fail ho sakta hai.
- Model checkpoints aur resume training workflow abhi limited hai.
- `models/metrics.json` historical non-LSTM metrics ho sakta hai; new LSTM metrics `lstm_metrics.json` mein save hote hain.

---

## 18. Future improvements

Priority order:

1. CPCB station observations ko ground truth banao.
2. Same station/time/averaging comparison automate karo.
3. Train-only scalers fit karo.
4. AQI unit tests add karo.
5. LSTM ko persistence aur XGBoost baselines se compare karo.
6. Day-wise metrics aur confidence range show karo.
7. ModelCheckpoint + Google Drive/Cloud storage add karo.
8. Wind direction, boundary-layer height, rainfall, traffic aur satellite/fire data add karo.
9. Direct AQI model ko pollutant model ke saath compare karo.
10. One global model vs six location models evaluate karo.
11. Probabilistic forecasting ya quantile loss use karo.
12. Model metadata save karo: training date, data range, features, scaler version, code version.

Expected user-facing format future mein:

```text
Expected AQI: 96
Likely range: 80-118
Confidence: Medium
Dominant pollutant: PM2.5
```

Single exact number false confidence de sakta hai.

---

## 19. Local run instructions

### Install

```powershell
cd D:\india-aqi-analysis
pip install -r requirements.txt
```

### Streamlit

```powershell
streamlit run app.py
```

Browser:

```text
http://localhost:8501
```

### Local retraining

```powershell
python src/prepare_lstm_data.py
python src/lstm_model.py --force --epochs 100 --batch-size 64
```

CPU par six models ko roughly 1-4 hours ya usse zyada lag sakta hai. Training ke liye Google Colab GPU better hai.

---

## 20. Google Colab training instructions

1. Project folder ko zip karo. `.git`, `.venv`, `__pycache__` exclude karo.
2. Colab kholo.
3. Runtime type mein T4 GPU select karo.
4. Zip upload karo:

```python
from google.colab import files
files.upload()
```

5. Extract karo:

```python
import zipfile
with zipfile.ZipFile("india-aqi-analysis.zip") as zip_ref:
    zip_ref.extractall("/content")
%cd /content/india-aqi-analysis
```

6. Install karo:

```python
!pip install -r requirements.txt
```

7. Data prepare karo:

```python
!python src/prepare_lstm_data.py
```

8. Old model skip na karne ke liye force retrain:

```python
!python src/lstm_model.py --force --epochs 100 --batch-size 64
```

9. Metrics dekho:

```python
import json
with open("models/lstm_metrics.json") as file:
    print(json.load(file))
```

10. Models download karo:

```python
from google.colab import files
import shutil
shutil.make_archive("trained_models", "zip", "models")
files.download("trained_models.zip")
```

11. Local `models/` folder mein new artifacts extract karo.
12. Local Streamlit run karo.

Colab runtime permanent nahi hota. Download kiye bina trained models lose ho sakte hain.

---

## 21. Interview questions aur short answers

### Q1. LSTM kyon use kiya?

AQI hourly time-series hai aur previous pollution/weather context future value ko affect karta hai. LSTM sequential dependencies aur longer context capture kar sakta hai.

### Q2. Simple regression kyon nahi?

Regression baseline ho sakta hai, lekin sequence order aur temporal memory naturally handle nahi karta. Phir bhi fair project mein regression/XGBoost baseline compare karna chahiye.

### Q3. 336 aur 72 numbers kahan se aaye?

336 = 14 days x 24 hours. 72 = next 3 days x 24 hours.

### Q4. Output 288 kyon hai?

72 future timestamps aur 4 pollutants: 72 x 4 = 288.

### Q5. AQI kaise calculate hota hai?

Har pollutant ka breakpoint interpolation sub-index nikalta hai aur maximum sub-index final AQI hota hai.

### Q6. Dropout ka purpose?

Overfitting reduce karna. Training examples par memorization kam karne ke liye activations ka fraction randomly drop hota hai.

### Q7. Early stopping kyon?

Validation loss improve na ho toh unnecessary epochs avoid karta hai aur best weights restore karta hai.

### Q8. Validation loss low hone ke baad bhi AQI wrong kyon?

Scaled pollutant loss low ho sakta hai, lekin AQI nonlinear breakpoint maximum par depend karta hai. Saath hi data source, time aggregation aur units mismatch ho sakte hain.

### Q9. Current biggest limitation?

Training/validation source aur user ke actual CPCB comparison source ka mismatch.

### Q10. Model ko kaise improve karoge?

CPCB station data, train-only scaling, fair baselines, AQI-level metrics, richer meteorological/traffic features aur uncertainty intervals.

### Q11. Model production-ready hai?

Abhi research/demo level. Production-ready banane ke liye ground-truth validation, monitoring, data-quality checks, model versioning aur retraining schedule chahiye.

### Q12. Model perfect hoga?

Nahi. Goal exact number nahi, reliable error range aur decision-useful forecast banana hai.

---

## 22. Final conclusion

Yeh project sirf ek LSTM file nahi hai; yeh data ingestion, time-series preprocessing, deep-learning training, AQI domain logic, live API inference, dashboarding aur evaluation ka complete pipeline hai.

Strong parts:

- End-to-end working structure.
- Six location-specific models.
- Multivariate pollution + weather input.
- Cyclic and seasonal feature engineering.
- 72-hour multi-output forecasting.
- CPCB-style AQI conversion.
- Streamlit user interface.
- Backtesting section.
- Colab retraining support.

Sabse important improvement direction:

```text
CPCB station ground truth
-> exact time/aggregation alignment
-> train-only scaling
-> baseline comparison
-> retrain with --force
-> AQI-level evaluation
-> deploy with confidence range
```

Interview mein project ko honest tareeke se present karna best hoga: architecture aur engineering decisions confidently explain karo, lekin accuracy ko overclaim mat karo. Current project useful prototype hai; reliable real-world product banne ke liye ground-truth CPCB validation sabse bada next step hai.
