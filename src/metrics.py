import numpy as np

from src.aqi import get_category

# Activity sensitivity factors based on inhalation rate and vulnerability
SENSITIVITY = {
    "General adult": 1.00,
    "Runner / cyclist": 1.85,
    "Child": 1.45,
    "Elderly": 1.40,
    "Asthma / COPD": 1.70,
    "Pregnant": 1.35,
}

# Thresholds for activity recommendations (Score out of 10)
ACTIVITY_ADVICE = [
    (8.5, "Go ahead", "Air is fine for this. No precautions needed."),
    (6.5, "Mostly fine", "Reasonable. Keep intense effort under an hour."),
    (4.5, "Take it easy", "Shorten it, lower the intensity, avoid main roads."),
    (2.5, "Move indoors", "Not worth it outdoors today. Train inside."),
    (0.0, "Stay in", "Keep windows shut and run a purifier if you have one."),
]

# Official EPA/CPCB AQI Breakpoints: (Concentration Low, Concentration High, AQI Low, AQI High)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def activity_safety_score(aqi: float, group: str = "General adult") -> float | None:
    """Calculates a personalized 1.0 to 10.0 safety score for outdoor activity."""
    if aqi is None or not np.isfinite(aqi):
        return None

    # Map AQI to a 10 (clean) down to 1 (severe) baseline score
    base_score = float(np.interp(
        aqi,
        [0, 50, 100, 200, 300, 400, 500],
        [10.0, 9.2, 7.8, 5.0, 2.8, 1.4, 1.0],
    ))

    sensitivity_factor = SENSITIVITY.get(group, 1.0)
    adjusted_score = 1.0 + (base_score - 1.0) / sensitivity_factor
    return round(float(np.clip(adjusted_score, 1.0, 10.0)), 1)


def activity_verdict(score: float | None) -> tuple[str, str]:
    """Returns a headline and actionable advice based on the activity safety score."""
    if score is None:
        return "No data", "Not enough readings to judge."
    for threshold, headline, detail in ACTIVITY_ADVICE:
        if score >= threshold:
            return headline, detail
    return "Stay in", ""


def aqi_to_pm25_concentration(aqi_val: float) -> float:
    """
    Inverts the piecewise AQI formula back to PM2.5 mass concentration (ug/m3).
    AQI is a non-linear index, so we interpolate backwards through the breakpoint table.
    """
    if aqi_val is None or not np.isfinite(aqi_val) or aqi_val <= 0:
        return 0.0

    for conc_low, conc_high, aqi_low, aqi_high in PM25_BREAKPOINTS:
        if aqi_val <= aqi_high:
            if aqi_high == aqi_low:
                return float(conc_high)
            fraction = (aqi_val - aqi_low) / (aqi_high - aqi_low)
            return float(conc_low + fraction * (conc_high - conc_low))

    # Extrapolate beyond AQI 500 linearly
    return float(500.4 + (aqi_val - 500) * 0.5)


def cigarette_equivalent(aqi_val: float) -> float:
    """
    Berkeley Earth Rule: 22 ug/m3 of PM2.5 over 24 hours equals smoking 1 cigarette.
    First inverts AQI to PM2.5 concentration (ug/m3), then divides by 22.
    """
    if aqi_val is None or not np.isfinite(aqi_val) or aqi_val <= 0:
        return 0.0
    concentration_ugm3 = aqi_to_pm25_concentration(aqi_val)
    return round(float(concentration_ugm3 / 22.0), 1)
