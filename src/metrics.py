"""Health translations of air quality: dose, cigarettes, safety, timing."""

import numpy as np
import pandas as pd

from src.aqi import get_category

PM25_PER_CIGARETTE = 22.0
WHO_DAILY_PM25 = 15.0

SENSITIVITY = {
    "General adult": 1.00,
    "Runner / cyclist": 1.85,
    "Child": 1.45,
    "Elderly": 1.40,
    "Asthma / COPD": 1.70,
    "Pregnant": 1.35,
}

ACTIVITY_ADVICE = [
    (8.5, "Go ahead", "Air is fine for this. No precautions needed."),
    (6.5, "Mostly fine", "Reasonable. Keep intense effort under an hour."),
    (4.5, "Take it easy", "Shorten it, lower the intensity, avoid main roads."),
    (2.5, "Move indoors", "Not worth it outdoors today. Train inside."),
    (0.0, "Stay in", "Keep windows shut and run a purifier if you have one."),
]


def cigarette_equivalent(pm25, hours=24.0):
    """Cigarettes-equivalent for a PM2.5 exposure over `hours`."""
    if pm25 is None or not np.isfinite(pm25) or pm25 < 0:
        return None
    return round((float(pm25) * float(hours) / 24.0) / PM25_PER_CIGARETTE, 2)


def who_multiple(pm25):
    """How many times the WHO 24-hour guideline this level is."""
    if pm25 is None or not np.isfinite(pm25):
        return None
    return round(float(pm25) / WHO_DAILY_PM25, 1)


def activity_safety_score(aqi, group="General adult"):
    """1-10 safety rating for outdoor activity. 10 is safe, 1 is not."""
    if aqi is None or not np.isfinite(aqi):
        return None
    base = float(np.interp(aqi, [0, 50, 100, 200, 300, 400, 500],
                                [10, 9.2, 7.8, 5.0, 2.8, 1.4, 1.0]))
    factor = SENSITIVITY.get(group, 1.0)
    score = 1.0 + (base - 1.0) / factor
    return round(float(np.clip(score, 1.0, 10.0)), 1)


def activity_verdict(score):
    if score is None:
        return "No data", "Not enough readings to judge."
    for threshold, headline, detail in ACTIVITY_ADVICE:
        if score >= threshold:
            return headline, detail
    return "Stay in", ""


def exposure_dose(segments):
    """Total PM2.5 dose across a day described as (hours, pm25) segments."""
    segments = [(float(h), float(p)) for h, p in segments
                if h and p is not None and np.isfinite(p)]
    if not segments:
        return None
    total_hours = sum(h for h, _ in segments)
    if total_hours <= 0:
        return None
    ug_hours = sum(h * p for h, p in segments)
    mean_pm25 = ug_hours / total_hours
    return {
        "hours": round(total_hours, 1),
        "mean_pm25": round(mean_pm25, 1),
        "ug_hours": round(ug_hours, 1),
        "cigarettes": cigarette_equivalent(mean_pm25, total_hours),
        "who_multiple": who_multiple(mean_pm25),
    }


def commute_exposure(home_pm25, work_pm25, commute_pm25, hours_home=14.0,
                     hours_work=8.0, hours_commute=2.0, indoor_factor=0.55):
    """Daily dose for a home / commute / work routine."""
    segments = [
        (hours_home, (home_pm25 or 0) * indoor_factor),
        (hours_work, (work_pm25 or 0) * indoor_factor),
        (hours_commute, commute_pm25 if commute_pm25 is not None else 0),
    ]
    result = exposure_dose(segments)
    if result:
        result["breakdown"] = {
            "Home (indoor)": round(hours_home * (home_pm25 or 0) * indoor_factor, 1),
            "Work (indoor)": round(hours_work * (work_pm25 or 0) * indoor_factor, 1),
            "Commute (outdoor)": round(hours_commute * (commute_pm25 or 0), 1),
        }
    return result


def best_outdoor_window(hourly, duration=2, aqi_col="aqi", time_col="datetime",
                        earliest=None, latest=None):
    """Cleanest contiguous `duration`-hour block in an hourly frame."""
    if hourly is None or len(hourly) == 0 or aqi_col not in hourly.columns:
        return None
    df = hourly[[time_col, aqi_col]].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    if len(df) < duration:
        return None

    hours = df[time_col].dt.hour
    mask = pd.Series(True, index=df.index)
    if earliest is not None:
        mask &= hours >= earliest
    if latest is not None:
        mask &= hours <= latest

    candidates = []
    for i in range(len(df) - duration + 1):
        block = df.iloc[i:i + duration]
        spans = (block[time_col].diff().dropna() == pd.Timedelta(hours=1)).all()
        if not spans or not mask.iloc[i:i + duration].all():
            continue
        candidates.append({
            "start": block[time_col].iloc[0],
            "end": block[time_col].iloc[-1],
            "mean_aqi": round(float(block[aqi_col].mean())),
            "max_aqi": round(float(block[aqi_col].max())),
        })
    if not candidates:
        return None

    candidates.sort(key=lambda c: c["mean_aqi"])
    best = dict(candidates[0])
    best["category"] = get_category(best["mean_aqi"])
    best["safety"] = activity_safety_score(best["mean_aqi"])
    worst = max(candidates, key=lambda c: c["mean_aqi"])
    best["saving_vs_worst"] = worst["mean_aqi"] - best["mean_aqi"]
    best["all_windows"] = candidates
    return best
