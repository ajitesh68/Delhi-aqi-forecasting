---
title: Delhi Air
emoji: 🌫️
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.63.0
app_file: app.py
pinned: false
license: mit
short_description: Live CPCB station AQI for Delhi, with a 24-hour PM2.5 forecast
---

# Delhi Air

Live air quality from the CPCB / DPCC ground-monitoring network across
Delhi, with a 24-hour PM2.5 forecast that is measured against the baseline
it has to beat rather than presented on trust.

The headline number most dashboards show is a city average. Delhi does not
have one air quality — on an ordinary afternoon the network reads 42 at
Cantonment Area and 214 at Sonia Vihar. That spread is the most useful
thing the data has to say, and this dashboard leads with it.

## What the data actually is

| Layer | Source | Covers | Lag |
|---|---|---|---|
| Live | data.gov.in CAAQMS feed | 44 Delhi stations, 7 pollutants | ~1 hour |
| History | OpenAQ CPCB archive | 7 stations, hourly, 19 months | ~5 days |
| Weather | Open-Meteo | observed + 24 h forecast | live |

Nothing here is a satellite or chemistry-transport model. An earlier version
of this project served Open-Meteo CAMS reanalysis while calling it CPCB
data; measured at Anand Vihar the two correlate **0.401**, which is why the
whole data layer was replaced.

Open-Meteo is still used, but only for weather — where it is genuinely good.

## AQI is computed the way CPCB defines it

Sub-indices use the mandated rolling windows — 24 hours for
PM2.5/PM10/NO2/SO2/NH3, 8 hours for CO/O3 — and an AQI is published only
when at least three sub-indices are available including a PM one. A
pollutant whose "24-hour average" is built from too few hours is dropped
rather than quietly averaged. `tests/test_aqi.py` pins this behaviour.

CO is excluded: the feed reports it in units that cannot be resolved from
the values themselves, and guessing wrong is a 1000x error.

## The forecast, and what it is worth

A pooled Conv1D → LSTM reads 168 hours of history for one station plus the
next 24 hours of forecast weather, and predicts PM2.5 for each of the next
24 hours.

Scored on January 2026, a month held out of training entirely:

| | Model | Persistence |
|---|---|---|
| MAE | **49.4 µg/m³** | 51.2 µg/m³ |
| Dirtiest 10% of hours | **107.3 µg/m³** | 126.7 µg/m³ |

Persistence — "tomorrow looks like today" — is a strong baseline at this
horizon, which is why it is the bar rather than a formality. The model
clears it by 3.4% overall and by 15.3% on the hours that matter.

Two design choices did that work. The model predicts a *departure from*
persistence rather than a level, so it starts level with the baseline and
can only add. And it reads forecast weather — wind, temperature, boundary
layer height — which is the only thing it knows that persistence does not.

One honest limitation: it is trained on a single Delhi winter, so it
under-predicts the sharpest peaks.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the committed parquet store, so it works offline apart from
the live feed. No API key is needed — data.gov.in's public demo key is the
fallback. Set `DATA_GOV_IN_API_KEY` for a personal one.

Rebuilding from scratch:

```bash
python scripts/download_cpcb_history.py   # OpenAQ backfill -> archive
python scripts/download_weather.py        # Open-Meteo per station
python scripts/prepare_cpcb_data.py       # gap-aware windows -> sequences
python scripts/train_cpcb.py              # pooled model + scorecard
```

`scripts/collect_snapshot.py` appends the current hour to the rolling store
and runs hourly in CI. It is what closes the archive's lag: until the
rolling window covers a continuous week, the forecast can only start where
the archive ends, and the app says so rather than implying otherwise.
