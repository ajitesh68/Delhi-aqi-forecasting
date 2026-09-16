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

A live air-quality dashboard for Delhi, built on the CPCB / DPCC
ground-monitoring network — the same stations behind the official
bulletin — with a 24-hour PM2.5 forecast.

Most dashboards show one number for the whole city. Delhi does not have
one air quality: on an ordinary afternoon the network reads 42 at
Cantonment Area and 214 at Sonia Vihar. This one leads with that spread.

## What it does

**Live map.** Every reporting station, ranked and mapped, coloured by
current AQI. Hover any station for its pollutant breakdown.

**When to go out.** The cleanest hours of the day for your area, from the
hour-by-hour profile of what has actually been measured there.

**Tomorrow.** A 24-hour PM2.5 forecast for the selected station, shown
next to the score it earned against a persistence baseline, so you can
see how much to trust it.

**History.** The last 24 hours, 7 days or 30 days of collected readings.
Hours the network never reported are drawn dotted and marked estimated.

**Your exposure.** What the air costs you personally — cigarette
equivalent, activity safety for your sensitivity group, and commute
exposure between two areas.

## Using it

Pick your station in the sidebar; everything on the page follows it. Set
"Who is this for?" to match the person you are asking about — a child,
an elderly person and an asthmatic get different advice from the same
AQI.

The AQI shown is CPCB's, computed CPCB's way: rolling averages (24 hours
for PM2.5, PM10, NO₂, SO₂, NH₃; 8 hours for CO and O₃), and at least
three pollutants including a PM one. A station that cannot meet that
shows no AQI rather than a number built from whatever reported.

## Running it locally

Python 3.11–3.13. **Not 3.14** — TensorFlow publishes no wheels for it
yet, and the resulting `No matching distribution found for tensorflow-cpu`
looks like a version-range problem but is the interpreter. On a managed
host, set the Python version to 3.13 before deploying.

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key needed — the public data.gov.in demo key is the fallback. That
key is shared by every anonymous caller and rate-limits often, so for
anything beyond a quick look, register a free key at
[data.gov.in](https://data.gov.in/user/register) and set it:

```bash
export DATA_GOV_IN_API_KEY=your_key_here
```

## How the data is kept

The live feed serves only the current hour — there is no history endpoint.
So `scripts/collect_snapshot.py` runs hourly in GitHub Actions, appending
each hour to a rolling parquet file that keeps the last 40 days. That file
is what every chart reads.

A separate archive of older hours (`data/store/cpcb_hourly.parquet`) is
kept for training the forecasting model and is not used by the charts.

## The forecast

A pooled Conv1D → LSTM reads 168 hours of history for one station plus the
next 24 hours of forecast weather, and predicts PM2.5 for each of the next
24 hours. Scored on a held-out January:

| | Model | Persistence |
|---|---|---|
| MAE | **49.4 µg/m³** | 51.2 µg/m³ |
| Dirtiest 10% of hours | **107.3 µg/m³** | 126.7 µg/m³ |

Persistence — "tomorrow looks like today" — is a strong baseline at this
horizon, which is why it is the bar. The model clears it by 3.4% overall
and 15.3% on the hours that matter. It is trained on a single Delhi
winter, so it under-predicts the sharpest peaks.

Rebuilding the model from scratch:

```bash
python scripts/download_cpcb_history.py   # archive backfill
python scripts/download_weather.py        # per-station weather
python scripts/prepare_cpcb_data.py       # windows -> sequences
python scripts/train_cpcb.py              # model + scorecard
```

## Layout

```
app.py                  Streamlit entry and tabs
src/
  aqi.py                CPCB AQI: breakpoints, rolling windows
  data_loader.py        the app's only data entry point
  forecast.py           model loading and inference
  analysis.py           daily roll-ups from observed hours
  metrics.py            exposure, activity safety, best window
  visualizations.py     Plotly charts
  maps.py               Pydeck maps
  sources/              data.gov.in, OpenAQ, parquet store
scripts/                backfill, training, hourly collector
tests/                  AQI correctness
```
